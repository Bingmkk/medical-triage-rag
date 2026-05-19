"""
医学分诊Agent核心模块
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from .llm_client import MedicalLLMClient
from .rag_engine import RAGEngine
from .knowledge_graph import MedicalKnowledgeGraph
from .vector_store import MedicalVectorStore
from .hospital_loader import HospitalDataLoader
from config import TRIAGE_CONFIG


class UrgencyLevel(Enum):
    """紧急程度枚举"""
    CRITICAL = 1
    EMERGENT = 2
    LESS_URGENT = 3
    NON_URGENT = 4


@dataclass
class TriageResult:
    """分诊结果数据类"""
    symptom_description: str
    urgency_level: UrgencyLevel
    urgency_name: str
    urgency_color: str
    urgency_description: str
    possible_diseases: List[str]
    recommended_department: str
    recommended_doctors: List[str]
    advice: str
    reasoning: str
    confidence: float


class MedicalTriageAgent:
    """医学分诊Agent"""

    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        knowledge_graph: Optional[MedicalKnowledgeGraph] = None,
        vector_store: Optional[MedicalVectorStore] = None,
        llm_client: Optional[MedicalLLMClient] = None,
        hospital_loader: Optional[HospitalDataLoader] = None
    ):
        self.llm_client = llm_client or MedicalLLMClient()
        self.rag_engine = rag_engine
        if self.rag_engine is None:
            self.rag_engine = RAGEngine()
            self.rag_engine.initialize()
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.hospital_loader = hospital_loader or HospitalDataLoader()
        self.hospital_loader.load_data()
        self.conversation_history = []

    def triage(self, symptom_description: str) -> TriageResult:
        """
        执行医学分诊

        Args:
            symptom_description: 患者的症状描述

        Returns:
            TriageResult: 分诊结果
        """
        self.conversation_history.append({
            "role": "user",
            "content": symptom_description
        })

        triage_data = self.rag_engine.triage(symptom_description)

        urgency_info = self._parse_urgency(triage_data["analysis"])

        possible_diseases = self._extract_diseases(triage_data["analysis"])

        recommended_department = self._extract_department(triage_data["analysis"], symptom_description)

        recommended_doctors = self._extract_doctors(triage_data["analysis"], recommended_department, urgency_info["level"])

        advice = self._extract_advice(triage_data["analysis"])

        reasoning = self._generate_reasoning(triage_data["analysis"], possible_diseases, urgency_info)

        result = TriageResult(
            symptom_description=symptom_description,
            urgency_level=UrgencyLevel(urgency_info["level"]),
            urgency_name=urgency_info["name"],
            urgency_color=urgency_info["color"],
            urgency_description=self._get_urgency_description(urgency_info["level"]),
            possible_diseases=possible_diseases,
            recommended_department=recommended_department,
            recommended_doctors=recommended_doctors,
            advice=advice,
            reasoning=reasoning,
            confidence=urgency_info.get("confidence", 0.8)
        )

        self.conversation_history.append({
            "role": "assistant",
            "content": self._format_triage_result(result)
        })

        return result

    def _parse_urgency(self, analysis: str) -> Dict[str, Any]:
        """解析紧急程度"""
        analysis_lower = analysis.lower()

        if any(keyword in analysis_lower for keyword in ["危急", "立即", "紧急电话", "拨打120", "危及生命", "死亡风险"]):
            return {"level": 1, "name": "危急", "color": "red", "confidence": 0.95}
        elif any(keyword in analysis_lower for keyword in ["紧急", "尽快", "严重", "威胁健康"]):
            return {"level": 2, "name": "紧急", "color": "orange", "confidence": 0.85}
        elif any(keyword in analysis_lower for keyword in ["次紧急", "不紧急", "可以等待"]):
            return {"level": 3, "name": "次紧急", "color": "yellow", "confidence": 0.75}
        else:
            return {"level": 4, "name": "非紧急", "color": "green", "confidence": 0.70}

    def _extract_diseases(self, analysis: str) -> List[str]:
        """提取可能的疾病"""
        diseases = []

        disease_keywords = [
            "可能是", "可能为", "考虑", "怀疑", "诊断", "常见于",
            "高血压", "糖尿病", "冠心病", "肺炎", "心肌梗死", "脑卒中",
            "支气管炎", "哮喘", "胃炎", "肝炎", "甲状腺", "痛风"
        ]

        lines = analysis.split("\n")
        for line in lines:
            for keyword in disease_keywords:
                if keyword in line:
                    disease = line.strip()
                    if len(disease) < 50:
                        diseases.append(disease)
                        break

        return list(set(diseases))[:5]

    def _extract_department(self, analysis: str, symptom_description: str = "") -> str:
        """提取建议科室（与推荐引擎对齐，保证名称存在于本院数据）"""
        urgency_info = self._parse_urgency(analysis) if analysis else {"level": 4}
        recommended = self.hospital_loader.recommend_departments(
            symptom_text=symptom_description,
            analysis=analysis,
            urgency_level=urgency_info.get("level", 4),
            limit=1,
        )
        if recommended:
            return recommended[0]["name"]
        if self._is_child(symptom_description):
            return self._get_pediatric_department()
        return "消化内科"
    
    def _is_child(self, text: str) -> bool:
        """判断是否是儿童（14岁以下）"""
        child_keywords = ["孩子", "小孩", "儿童", "宝宝", "婴儿", "幼儿", "小儿", "小朋友", "女童", "男童"]
        
        for keyword in child_keywords:
            if keyword in text:
                return True
        
        age_pattern = r'(\d+)\s*(岁|周岁|岁半|个月)'
        match = re.search(age_pattern, text)
        if match:
            age = int(match.group(1))
            unit = match.group(2)
            if unit == "个月":
                return age <= 168
            return age < 14
        
        return False
    
    def _get_pediatric_department(self) -> str:
        """获取儿科科室名称"""
        dept_info = self.hospital_loader.get_department_info("儿科普通门诊")
        if dept_info:
            return "儿科普通门诊"
        
        dept_info = self.hospital_loader.get_department_info("儿科")
        if dept_info:
            return "儿科"
        
        return "儿科普通门诊"

    def _extract_doctors(self, analysis: str, department_name: str = "", urgency_level: int = 4) -> List[str]:
        """提取推荐医生（根据紧急程度差异化推荐）"""
        import re
        from datetime import datetime
        
        current_time = datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        
        is_work_hours = 8 <= current_hour < 17 and current_weekday < 5
        
        if urgency_level <= 2:
            emergency_info = self.hospital_loader.get_department_info("急诊科")
            if emergency_info and emergency_info.get("doctors"):
                return [doc["name"] for doc in emergency_info["doctors"][:3]]
        
        available_doctors = []
        
        if department_name:
            dept_info = self.hospital_loader.get_department_info(department_name)
            if dept_info and dept_info.get("doctors"):
                for doctor in dept_info["doctors"]:
                    schedule = doctor.get("schedule", "")
                    if self._is_doctor_available_now(schedule, current_time):
                        available_doctors.append(doctor["name"])
                    elif self._is_doctor_available_soon(schedule, current_time):
                        available_doctors.append(doctor["name"])
        
        if urgency_level == 3 and available_doctors:
            emergency_info = self.hospital_loader.get_department_info("急诊科")
            if emergency_info and emergency_info.get("doctors"):
                emergency_doctors = [doc["name"] for doc in emergency_info["doctors"][:2]]
                return list(dict.fromkeys(available_doctors[:2] + emergency_doctors))
        
        if available_doctors:
            return list(dict.fromkeys(available_doctors[:3]))
        
        if urgency_level == 3:
            emergency_info = self.hospital_loader.get_department_info("急诊科")
            if emergency_info and emergency_info.get("doctors"):
                return [doc["name"] for doc in emergency_info["doctors"][:3]]
        
        if department_name:
            dept_info = self.hospital_loader.get_department_info(department_name)
            if dept_info and dept_info.get("doctors"):
                return [doc["name"] for doc in dept_info["doctors"][:3]]
        
        return []
    
    def _is_doctor_available_now(self, schedule: str, current_time) -> bool:
        """判断医生当前是否出诊"""
        current_weekday = current_time.weekday()
        current_hour = current_time.hour
        
        if "全天" in schedule or "24小时" in schedule:
            if 8 <= current_hour < 17:
                return True
        elif "上午" in schedule:
            if 8 <= current_hour < 12:
                return True
        elif "下午" in schedule:
            if 13 <= current_hour < 17:
                return True
        
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
        for day_char, day_num in weekday_map.items():
            if day_char in schedule and current_weekday == day_num:
                if "上午" in schedule:
                    return 8 <= current_hour < 12
                elif "下午" in schedule:
                    return 13 <= current_hour < 17
                else:
                    return 8 <= current_hour < 17
        
        return False
    
    def _is_doctor_available_soon(self, schedule: str, current_time) -> bool:
        """判断医生最近是否有排班"""
        current_weekday = current_time.weekday()
        current_hour = current_time.hour
        
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
        
        for day_char, day_num in weekday_map.items():
            if day_char in schedule:
                days_ahead = (day_num - current_weekday) % 7
                if days_ahead <= 2:
                    return True
        
        return False

    def _extract_advice(self, analysis: str) -> str:
        """提取建议"""
        advice_lines = []

        lines = analysis.split("\n")
        for line in lines:
            if any(keyword in line for keyword in ["建议", "注意", "应该", "立即", "尽快"]):
                advice_lines.append(line.strip())

        if advice_lines:
            return "；".join(advice_lines[:3])

        return "建议及时就医，遵循医生的治疗方案。"

    def _generate_reasoning(self, analysis: str, diseases: List[str], urgency: Dict[str, Any]) -> str:
        """生成推理过程"""
        reasoning_parts = []

        if diseases:
            reasoning_parts.append(f"根据症状分析，可能涉及的疾病包括：{', '.join(diseases[:3])}。")

        reasoning_parts.append(f"综合评估后，紧急程度判定为：{urgency['name']}。")

        if urgency["level"] == 1:
            reasoning_parts.append("由于症状表明可能存在危及生命的情况，需要立即进行医疗干预。")
        elif urgency["level"] == 2:
            reasoning_parts.append("症状表明情况较为严重，需要在较短时间内得到专业诊治。")
        elif urgency["level"] == 3:
            reasoning_parts.append("症状表明需要医疗关注，但暂无生命危险。")
        else:
            reasoning_parts.append("症状表明可以在常规门诊时间进行诊治。")

        return " ".join(reasoning_parts)

    def _get_urgency_description(self, level: int) -> str:
        """获取紧急程度描述"""
        descriptions = {
            1: "情况危急，可能危及生命，请立即拨打120急救电话或前往最近的急诊科！",
            2: "情况紧急，需要尽快就医，建议立即前往医院急诊或尽快预约专科门诊。",
            3: "情况次紧急，建议在24小时内就医，或先在门诊进行初步检查。",
            4: "情况非紧急，可以预约门诊或在方便时前往医院就诊。"
        }
        return descriptions.get(level, "")

    def _format_triage_result(self, result: TriageResult) -> str:
        """格式化分诊结果"""
        doctors_line = f"\n推荐医生：{', '.join(result.recommended_doctors)}" if result.recommended_doctors else ""
        
        return f"""
【医学分诊结果】

紧急程度：{result.urgency_name} ({result.urgency_level.value}/4)
{result.urgency_description}

可能疾病：{', '.join(result.possible_diseases) if result.possible_diseases else '待进一步检查确定'}

建议科室：{result.recommended_department}{doctors_line}

处置建议：{result.advice}

分析依据：{result.reasoning}
""".strip()

    def batch_triage(self, symptom_descriptions: List[str]) -> List[TriageResult]:
        """批量分诊"""
        results = []
        for description in symptom_descriptions:
            result = self.triage(description)
            results.append(result)
        return results

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history

    def clear_conversation(self) -> None:
        """清除对话历史"""
        self.conversation_history = []

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.conversation_history:
            return {
                "total_queries": 0,
                "urgency_distribution": {}
            }

        urgency_counts = {level.value: 0 for level in UrgencyLevel}

        for entry in self.conversation_history:
            if entry["role"] == "assistant" and "urgency_level" in entry["content"]:
                for level in UrgencyLevel:
                    if level.name in entry["content"]:
                        urgency_counts[level.value] += 1
                        break

        return {
            "total_queries": len([e for e in self.conversation_history if e["role"] == "user"]),
            "urgency_distribution": urgency_counts
        }


class MultiTurnTriageAgent(MedicalTriageAgent):
    """多轮对话分诊Agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_triage = None

    def ask_followup(self, question: str) -> str:
        """追问"""
        if not self.conversation_history:
            return "请先描述您的症状。"

        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-6:]
        ])

        prompt = f"""基于以下对话历史，回答用户的追问。

对话历史：
{context}

用户追问：{question}

请提供简洁、有针对性的回答。如果追问超出医学分诊范围，请礼貌地引导回分诊话题。
"""

        answer = self.llm_client.generate(prompt)

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def refine_triage(self, additional_info: str) -> TriageResult:
        """基于补充信息重新分诊"""
        full_description = self.conversation_history[0]["content"] + f"\n\n补充信息：{additional_info}"

        return self.triage(full_description)
