#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
症状追问模块 - 智能去重版
解决重复提问问题
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from .llm_client import MedicalLLMClient


class SymptomClarifier:
    """智能症状追问系统 - 去重版"""

    def __init__(self, max_rounds: int = 5):
        """
        初始化症状追问系统

        Args:
            max_rounds: 最大追问轮数（上限）
        """
        self.max_rounds = max_rounds
        self.llm_client = MedicalLLMClient()
        self.conversation_history = []
        self.symptom_context = {}
        self.current_round = 0
        self.asked_topics: Set[str] = set()
        self.asked_questions: Set[str] = set()  # 记录已问过的问题文本
        self.key_symptoms: List[str] = []
        self.is_emergency = False
        self.emergency_keywords = [
            '昏迷', '晕倒', '撞人', '事故', '车祸', '出血', '呼吸困难', 
            '胸痛', '心跳加速', '抽搐', '高热', '意识模糊', '休克',
            '120', '急救', '紧急', '危险', '危及生命'
        ]

    def _detect_emergency(self, text: str) -> bool:
        """检测紧急情况关键词"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.emergency_keywords)

    def _extract_key_symptoms(self, text: str) -> List[str]:
        """提取关键症状"""
        symptom_keywords = {
            '头痛': ['头疼', '头痛', '头晕', '眩晕'],
            '胸痛': ['胸痛', '胸闷', '胸口疼', '心脏痛'],
            '腹痛': ['腹痛', '肚子痛', '胃疼', '腹部不适'],
            '发热': ['发烧', '发热', '体温高', '高烧'],
            '呼吸困难': ['呼吸', '喘气', '胸闷', '气短'],
            '外伤': ['受伤', '流血', '骨折', '撞伤'],
            '意识问题': ['昏迷', '晕倒', '意识', '清醒'],
            '事故': ['撞人', '车祸', '事故', '摔倒']
        }
        
        detected = []
        text_lower = text.lower()
        for category, keywords in symptom_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(category)
        return detected

    def _is_duplicate_question(self, new_question: str) -> bool:
        """检测新问题是否与历史问题重复或高度相似"""
        new_q_lower = new_question.lower().strip()
        
        # 完全重复检测
        if new_q_lower in self.asked_questions:
            return True
        
        # 相似问题检测（检查关键词重复）
        new_q_words = set(new_q_lower.replace('？', '').replace('吗', '').split())
        
        for old_q in self.asked_questions:
            old_q_lower = old_q.lower().replace('？', '').replace('吗', '')
            old_q_words = set(old_q_lower.split())
            
            # 如果两个问题有60%以上的关键词相同，则视为重复
            common_words = new_q_words & old_q_words
            if len(common_words) >= 0.6 * len(new_q_words) and len(common_words) >= 2:
                return True
        
        return False

    def start_clarification(self, initial_symptom: str) -> Dict[str, Any]:
        """
        开始症状梳理对话
        """
        self.conversation_history = []
        self.symptom_context = {
            "initial_symptom": initial_symptom
        }
        self.current_round = 0
        self.asked_topics = set()
        self.asked_questions = set()
        self.key_symptoms = self._extract_key_symptoms(initial_symptom)
        self.is_emergency = self._detect_emergency(initial_symptom)

        self.conversation_history.append({
            "role": "patient",
            "content": initial_symptom
        })

        return self._generate_next_question()

    def continue_clarification(self, patient_answer: str) -> Dict[str, Any]:
        """
        继续追问对话
        """
        self.conversation_history.append({
            "role": "patient",
            "content": patient_answer
        })

        self.current_round += 1

        if self._detect_emergency(patient_answer):
            self.is_emergency = True
        
        new_symptoms = self._extract_key_symptoms(patient_answer)
        self.key_symptoms.extend([s for s in new_symptoms if s not in self.key_symptoms])

        if self.current_round >= self.max_rounds:
            return self._finish_clarification("追问轮数已达上限")

        return self._generate_next_question()

    def finish_early(self, reason: str = "用户主动结束") -> Dict[str, Any]:
        """
        提前结束追问
        """
        return self._finish_clarification(reason)

    def _extract_topic_from_question(self, question: str) -> str:
        """从问题中提取主题，避免重复"""
        question_lower = question.lower()

        if "多久" in question or "时间" in question or "持续" in question or "开始" in question:
            return "duration"
        elif "部位" in question or "位置" in question or "哪里" in question:
            return "location"
        elif "怎么" in question or "如何" in question or "情况" in question:
            return "how"
        elif "伴随" in question or "其他" in question or "还有" in question:
            return "associated_symptoms"
        elif "严重" in question or "程度" in question or "厉害" in question:
            return "severity"
        elif "原因" in question or "为什么" in question:
            return "cause"
        elif "处理" in question or "做什么" in question:
            return "action_taken"

        return f"topic_{self.current_round}"

    def _generate_next_question(self) -> Dict[str, Any]:
        """生成下一轮追问问题 - 智能去重版"""
        context = self._build_context_for_llm()
        previous_question = self.conversation_history[-2]["content"] if len(self.conversation_history) > 1 else ""
        previous_answer = self.conversation_history[-1]["content"]

        if previous_question:
            topic = self._extract_topic_from_question(previous_question)
            self.asked_topics.add(topic)
            self.asked_questions.add(previous_question)

        symptom_type = ", ".join(self.key_symptoms) if self.key_symptoms else "未明确"

        if self.current_round == 0:
            prompt = f"""你是一位专业的急诊医生，正在快速了解患者情况。

患者描述：{self.symptom_context.get('initial_symptom', '')}

分析：{'【紧急情况】' if self.is_emergency else '【常规问诊】'}
已识别症状类型：{symptom_type}

请根据以下原则提出最重要的第一个问题：
1. 如果是紧急情况，优先确认时间、地点、当前状态
2. 如果是常规症状，先问持续时间或发生时间
3. 问题要直接、明确，帮助快速了解病情
4. 避免假设，只基于患者已说的内容提问
5. 问题要与患者描述直接相关

示例（紧急情况）：
- "您说人昏迷了，这种情况发生多久了？"
- "您在哪里？需要立即拨打120吗？"

示例（常规症状）：
- "您说头痛，这种情况大概持续多久了？"
- "您说胃痛，是今天才开始的吗？"

请直接输出1个问题，不要其他内容。"""
        else:
            asked_topics_list = ", ".join(self.asked_topics) if self.asked_topics else "无"
            prompt = f"""你是一位专业的急诊医生，正在快速了解患者情况。

=== 对话历史 ===
{context}
=== 对话结束 ===

上一轮你问了：{previous_question}
患者最新回答：{previous_answer}

分析：{'【紧急情况】' if self.is_emergency else '【常规问诊】'}
已识别症状类型：{symptom_type}
已问过的话题类型：{asked_topics_list}

请根据以下原则决定下一步：
1. 如果是紧急情况，优先问关键信息：时间、地点、当前状态、已采取措施
2. 如果患者回答与之前话题完全无关（如从蚊子叮咬突然转到撞人），立即跟进新话题
3. 如果信息足够明确，可以输出"[可以了]"直接结束追问
4. 问题必须与患者当前描述高度相关
5. 绝对不要问与之前重复或相似的问题！

紧急情况优先询问顺序：
1. 发生时间（多久了？什么时候发生的？）
2. 当前位置和状况（人现在怎么样？在哪里？）
3. 已采取措施（有没有打120？做了什么急救？）
4. 具体情况（怎么发生的？什么原因？）

常规症状询问顺序：
1. 持续时间（多久了？）
2. 严重程度（疼得厉害吗？）
3. 伴随症状（还有其他不舒服吗？）
4. 诱发/缓解因素（什么情况下加重/好转？）

请直接输出问题或"[可以了]"，不要其他内容。"""

        # 最多尝试3次生成不重复的问题
        max_attempts = 3
        for attempt in range(max_attempts):
            response = self.llm_client.generate(prompt)
            response = response.strip()

            if "[可以了]" in response or "[结束]" in response or "可以了" in response:
                return self._finish_clarification("关键信息已收集")

            # 检查是否重复
            if self._is_duplicate_question(response):
                if attempt < max_attempts - 1:
                    # 添加提示让LLM避免重复
                    prompt += f"\n\n注意：你刚才问的问题与之前重复了，请换一个不同的问题！"
                    continue
                else:
                    # 多次尝试后仍然重复，直接结束追问
                    return self._finish_clarification("关键信息已收集")
            else:
                # 问题不重复，接受这个问题
                break

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return {
            "continue": True,
            "question": response,
            "round": self.current_round + 1,
            "max_rounds": self.max_rounds,
            "progress": f"{self.current_round + 1}/{self.max_rounds}",
            "symptom_type": symptom_type,
            "is_emergency": self.is_emergency
        }

    def _finish_clarification(self, reason: str) -> Dict[str, Any]:
        """结束追问，生成汇总"""
        context = self._build_context_for_llm()

        summary_prompt = f"""你是一位医生助手，将患者的描述整理成清晰的症状摘要。

=== 对话记录 ===
{context}
=== 对话结束 ===

请根据上面的对话记录，整理患者的症状。不要编造信息，只写患者明确说过的话。

输出格式：
【症状总结】
主要症状：
发病时间：
关键信息：

请用简洁的语言整理，只写患者明确说过的内容。"""

        summary = self.llm_client.generate(summary_prompt)

        return {
            "continue": False,
            "reason": reason,
            "summary": summary,
            "conversation": self.conversation_history,
            "symptom_context": self.symptom_context,
            "total_rounds": self.current_round,
            "asked_topics": list(self.asked_topics),
            "asked_questions": list(self.asked_questions),
            "key_symptoms": self.key_symptoms,
            "is_emergency": self.is_emergency,
            "can_proceed_to_triage": True
        }

    def _build_context_for_llm(self) -> str:
        """构建供LLM使用的上下文"""
        lines = []
        lines.append(f"患者最初描述：{self.symptom_context.get('initial_symptom', '')}")

        q_count = 0
        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                q_count += 1
                lines.append(f"医生问{q_count}：{msg['content']}")
            else:
                lines.append(f"患者回答：{msg['content']}")

        return "\n".join(lines)

    def get_clarified_symptom(self) -> str:
        """获取梳理后的完整症状描述"""
        if not self.conversation_history:
            return ""

        lines = [f"【完整对话记录】\n患者描述：{self.symptom_context.get('initial_symptom', '')}"]

        q_count = 0
        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                q_count += 1
                lines.append(f"\n医生问{q_count}：{msg['content']}")
            else:
                lines.append(f"\n患者回答：{msg['content']}")

        return "\n".join(lines)


def create_clarifier(max_rounds: int = 5) -> SymptomClarifier:
    """创建症状追问器"""
    return SymptomClarifier(max_rounds=max_rounds)


if __name__ == "__main__":
    print("=" * 70)
    print("症状梳理对话 - 智能去重版")
    print("=" * 70)

    clarifier = SymptomClarifier(max_rounds=5)

    print("\n👋 您好！我来帮您理清一下症状。")
    print("请描述您哪里不舒服：")

    initial = input("\n📝 ")

    print(f"\n好的，我听到了。让我帮您理清一下...\n")

    result = clarifier.start_clarification(initial)

    print(f"\n💬 {result['question']}")
    print(f"   [{result['progress']}]")
    if result.get('is_emergency'):
        print("   ⚠️ 检测到紧急情况")

    for i in range(2, 7):
        answer = input(f"\n📝 您的回答：")
        result = clarifier.continue_clarification(answer)

        if not result["continue"]:
            print(f"\n✅ 追问结束，共问了 {result['total_rounds']} 个问题")
            print(f"已问问题：{result['asked_questions']}")
            break

        print(f"\n💬 {result['question']}")
        print(f"   [{result['progress']}]")
        if result.get('is_emergency'):
            print("   ⚠️ 检测到紧急情况")

    print("\n" + "=" * 70)
    print("【症状总结】")
    print("=" * 70)
    print(result["summary"])