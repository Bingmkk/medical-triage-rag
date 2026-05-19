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

import re
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
        self.clarified_summary: str = ""
        self.known_facts: Dict[str, str] = {}
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
            '事故': ['撞人', '车祸', '事故', '摔倒'],
            '风湿关节': ['风湿', '关节痛', '关节炎', '类风湿', '痛风', '腰腿痛'],
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

    def _get_all_patient_text(self) -> str:
        """合并患者全部表述（初始描述 + 各轮回答）"""
        parts = [self.symptom_context.get("initial_symptom", "")]
        for msg in self.conversation_history:
            if msg["role"] == "patient":
                parts.append(msg["content"])
        return " ".join(p for p in parts if p).strip()

    def _extract_known_facts(self) -> Dict[str, str]:
        """从患者已说的话中提取「已知信息」，用于禁止重复追问"""
        text = self._get_all_patient_text()
        if not text:
            return {}

        facts: Dict[str, str] = {}

        duration_patterns = [
            r"\d+\s*[天日周月年]",
            r"[一二三四五六七八九十百千两]+[天日周月年]",
            r"最近|今天|昨天|今早|昨晚|刚才|一直|多年|很久|刚开始",
        ]
        if any(re.search(p, text) for p in duration_patterns):
            facts["duration"] = "发病/持续时间"

        severity_words = [
            "厉害", "很严重", "特别疼", "疼得厉害", "难受", "加重", "更明显",
            "受不了", "剧烈", "严重", "厉害多了",
        ]
        if any(w in text for w in severity_words):
            facts["severity"] = "严重程度或疼痛程度"

        trigger_words = [
            "下雨", "雨天", "潮湿", "阴雨天", "天冷", "受凉", "风吹",
            "劳累", "运动后", "活动后", "走路", "久坐", "弯腰",
            "吃了", "喝了", "熬夜", "情绪激动", "因为",
        ]
        matched_triggers = [w for w in trigger_words if w in text]
        if matched_triggers:
            facts["aggravating_factors"] = (
                "诱发或加重因素（如：" + "、".join(matched_triggers[:4]) + "）"
            )

        relieve_words = ["休息后", "吃药后", "热敷", "缓解", "好转", "减轻", "舒服些"]
        if any(w in text for w in relieve_words):
            facts["relieving_factors"] = "缓解因素"

        location_words = [
            "左边", "右边", "两侧", "单侧", "双侧", "膝盖", "手腕", "手指",
            "肩膀", "腰", "背", "颈", "头", "胸", "腹", "脚", "踝", "肘",
        ]
        if any(w in text for w in location_words):
            facts["location"] = "不适部位"

        associated_words = [
            "发烧", "发热", "恶心", "呕吐", "麻木", "肿胀", "僵硬",
            "无力", "皮疹", "瘙痒", "出血",
        ]
        if any(w in text for w in associated_words):
            facts["associated_symptoms"] = "部分伴随症状"

        if re.search(r"风湿|关节|类风湿|痛风", text):
            facts["disease_context"] = "风湿/关节相关描述"

        return facts

    def _sync_asked_topics_from_facts(self) -> None:
        """患者已说清的信息，标记为已覆盖话题"""
        self.known_facts = self._extract_known_facts()
        topic_map = {
            "duration": "duration",
            "severity": "severity",
            "aggravating_factors": "aggravating_factors",
            "relieving_factors": "relieving_factors",
            "location": "location",
            "associated_symptoms": "associated_symptoms",
        }
        for fact_key, topic in topic_map.items():
            if fact_key in self.known_facts:
                self.asked_topics.add(topic)

    def _format_known_facts_block(self) -> str:
        if not self.known_facts:
            return "（暂无，需从患者描述中获取）"
        return "\n".join(f"- {v}" for v in self.known_facts.values())

    def _format_forbidden_examples(self) -> str:
        examples = []
        if "aggravating_factors" in self.known_facts or "severity" in self.known_facts:
            examples.append(
                '- 患者已说明「下雨/潮湿等会让症状加重或更厉害」时，禁止再问：'
                '「什么时候更严重」「什么情况下会加重」「疼得厉害吗」'
            )
        if "duration" in self.known_facts:
            examples.append(
                '- 患者已说明时间/多久时，禁止再问：「持续多久」「什么时候开始的」'
            )
        if "location" in self.known_facts:
            examples.append('- 患者已说明部位时，禁止再问：「哪里不舒服」「哪个部位」')
        if not examples:
            examples.append("- 不要换种说法重复问患者已经回答过的内容")
        return "\n".join(examples)

    def _extract_topic_from_question(self, question: str) -> str:
        """从问题中提取主题，避免重复"""
        question_lower = question.lower()

        if ("什么时候" in question or "何时" in question or "什么情况下" in question) and (
            "严重" in question or "加重" in question or "厉害" in question or "更明显" in question
        ):
            return "aggravating_factors"
        if "多久" in question or "时间" in question or "持续" in question or "开始" in question:
            return "duration"
        elif "部位" in question or "位置" in question or "哪里" in question or "哪儿" in question:
            return "location"
        elif "伴随" in question or "其他" in question or "还有" in question or "同时" in question:
            return "associated_symptoms"
        elif "严重" in question or "程度" in question or "厉害" in question or "多疼" in question:
            return "severity"
        elif (
            "加重" in question or "诱发" in question or "缓解" in question
            or "什么情况" in question or "什么原因" in question
        ):
            return "aggravating_factors"
        elif "原因" in question or "为什么" in question:
            return "cause"
        elif "处理" in question or "做什么" in question or "用药" in question:
            return "action_taken"
        elif "怎么" in question or "如何" in question or "什么样" in question:
            return "how"

        return f"topic_{self.current_round}"

    def _is_redundant_with_known_facts(self, question: str) -> bool:
        """问题是否在追问患者已经说过的信息"""
        topic = self._extract_topic_from_question(question)
        if topic in self.asked_topics and topic != f"topic_{self.current_round}":
            return True
        if topic in self.known_facts:
            return True
        if topic == "severity" and "aggravating_factors" in self.known_facts:
            return True
        if topic == "aggravating_factors" and (
            "aggravating_factors" in self.known_facts or "severity" in self.known_facts
        ):
            return True
        if topic == "cause" and "aggravating_factors" in self.known_facts:
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

    def _generate_next_question(self) -> Dict[str, Any]:
        """生成下一轮追问问题 - 智能去重版"""
        self._sync_asked_topics_from_facts()

        context = self._build_context_for_llm()
        previous_question = self.conversation_history[-2]["content"] if len(self.conversation_history) > 1 else ""
        previous_answer = self.conversation_history[-1]["content"]

        if previous_question:
            topic = self._extract_topic_from_question(previous_question)
            self.asked_topics.add(topic)
            self.asked_questions.add(previous_question)

        symptom_type = ", ".join(self.key_symptoms) if self.key_symptoms else "未明确"
        known_block = self._format_known_facts_block()
        forbidden_block = self._format_forbidden_examples()
        asked_topics_list = ", ".join(sorted(self.asked_topics)) if self.asked_topics else "无"

        listening_rules = f"""
【倾听原则 — 必须遵守】
1. 先完整阅读患者已说的每一句话，把已知信息当作「已回答」，不得换说法再问一遍。
2. 患者已提供的信息（禁止重复追问）：
{known_block}
3. 禁止重复追问示例：
{forbidden_block}
4. 只问「上面列表里还没有」且对分诊有帮助的信息；若关键信息已够，输出「[可以了]」。
5. 已问过或已覆盖的话题类型（禁止再问同类）：{asked_topics_list}
"""

        if self.current_round == 0:
            prompt = f"""你是一位专业的急诊医生，正在快速了解患者情况。

患者描述：{self.symptom_context.get('initial_symptom', '')}

分析：{'【紧急情况】' if self.is_emergency else '【常规问诊】'}
已识别症状类型：{symptom_type}
{listening_rules}

请提出第一个问题：
- 紧急情况：优先问尚未说明的时间、地点、当前状态（仅问缺失项）
- 常规症状：不要机械地问「持续多久」；若患者已说明时间、加重因素、严重程度，改问其他缺失信息，例如：
  · 具体哪些关节/部位不适
  · 是否肿胀、晨僵、活动受限
  · 是否用药、既往诊断
- 风湿/关节痛且已提到下雨、潮湿、天冷等加重：禁止再问「什么时候更严重」「什么情况下加重」

请直接输出1个问题，或信息已足够时输出「[可以了]」。"""
        else:
            prompt = f"""你是一位专业的急诊医生，正在快速了解患者情况。

=== 对话历史 ===
{context}
=== 对话结束 ===

上一轮你问了：{previous_question}
患者最新回答：{previous_answer}

分析：{'【紧急情况】' if self.is_emergency else '【常规问诊】'}
已识别症状类型：{symptom_type}
{listening_rules}

请决定下一步：
1. 紧急情况：只追问尚未说明的关键信息（时间、地点、状态、已采取措施）
2. 患者补充了新话题时，跟进新信息，但仍不要重复问已知内容
3. 信息足够分诊时，输出「[可以了]」
4. 禁止用不同措辞重复同一信息（例如患者已说下雨后风湿加重，不要再问何时更严重）

仍可追问的维度（仅当患者尚未说明时）：
- 具体部位 / 是否对称
- 伴随症状（发热、肿胀、麻木等）
- 是否用药及效果
- 既往类似发作

请直接输出一个问题，或「[可以了]」。"""

        max_attempts = 4
        response = ""
        for attempt in range(max_attempts):
            response = self.llm_client.generate(prompt).strip()

            if "[可以了]" in response or "[结束]" in response:
                return self._finish_clarification("关键信息已收集")

            redundant_reason = None
            if self._is_duplicate_question(response):
                redundant_reason = "与之前的问题措辞重复"
            elif self._is_redundant_with_known_facts(response):
                topic = self._extract_topic_from_question(response)
                redundant_reason = f"患者已说明相关内容（话题：{topic}）"

            if redundant_reason:
                if attempt < max_attempts - 1:
                    prompt += (
                        f"\n\n【系统提示】你刚才的问题不合格：{redundant_reason}。"
                        f"请改问患者尚未说明的其他维度，或输出「[可以了]」。"
                    )
                    continue
                return self._finish_clarification("关键信息已收集")
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
        self.clarified_summary = summary.strip()

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
        """获取梳理后的症状描述（优先返回症状摘要，供分诊与科室推荐使用）"""
        if self.clarified_summary:
            return self.clarified_summary
        if not self.conversation_history:
            return self.symptom_context.get("initial_symptom", "")

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