"""
科室推荐引擎：以 LLM 分诊分析中的科室为主，叠加四级急诊分级制度
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from config import TRIAGE_CONFIG
except ImportError:
    TRIAGE_CONFIG = {
        "urgency_levels": [
            {"level": 1, "name": "危急", "color": "red"},
            {"level": 2, "name": "紧急", "color": "orange"},
            {"level": 3, "name": "次紧急", "color": "yellow"},
            {"level": 4, "name": "非紧急", "color": "green"},
        ]
    }

# LLM 常用科室名 -> 本院标准科室名
DEPARTMENT_ALIASES: Dict[str, str] = {
    "神经内科": "神经科",
    "神经外科": "神经科",
    "心血管内科": "心内科",
    "心脏内科": "心内科",
    "呼吸科": "呼吸内科",
    "消化科": "消化内科",
    "内分泌": "内分泌科",
    "骨科门诊": "骨科",
    "创伤骨科": "骨科",
    "急诊": "急诊科",
    "急诊医学科": "急诊科",
    "儿科门诊": "儿科普通门诊",
    "小儿科": "儿科普通门诊",
    "儿童科": "儿科普通门诊",
    "皮肤科门诊": "皮肤科",
    "耳鼻喉": "耳鼻喉科",
    "眼科门诊": "眼科",
    "口腔": "口腔科",
    "泌尿科": "泌尿外科",
    "肾脏内科": "肾内科",
    "风湿科": "风湿免疫科",
    "肿瘤科": "血液科",
}

CHILD_KEYWORDS = [
    "孩子", "小孩", "儿童", "宝宝", "婴儿", "幼儿", "小儿", "小朋友", "女童", "男童",
]

LLM_DEPT_PATTERNS = (
    r"科室[：:]\s*([^（(\n]+)",
    r"建议(?:就诊)?科室[：:]\s*([^（(\n]+)",
    r"推荐科室[：:]\s*([^（(\n]+)",
)

# 四级急诊：在「分析科室」之上如何叠加急诊路径
URGENCY_TIER_POLICY: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "危急",
        "emergency_first": True,
        "include_emergency": True,
        "emergency_reason": "分级急诊：危及生命，须立即急诊处置",
        "secondary_reason": "分诊分析建议的专科（病情稳定后进一步诊治）",
        "visit_guidance": "请立即拨打120或前往急诊科，不要自行拖延。",
    },
    2: {
        "name": "紧急",
        "emergency_first": True,
        "include_emergency": True,
        "emergency_reason": "分级急诊：需尽快急诊评估",
        "secondary_reason": "分诊分析建议的就诊科室",
        "visit_guidance": "请尽快前往急诊科就诊；稳定后可至专科门诊随访。",
    },
    3: {
        "name": "次紧急",
        "emergency_first": False,
        "include_emergency": False,
        "primary_reason": "分诊分析建议的就诊科室",
        "visit_guidance": "建议尽快预约专科门诊；若症状加重，请及时急诊。",
    },
    4: {
        "name": "非紧急",
        "emergency_first": False,
        "include_emergency": False,
        "primary_reason": "分诊分析建议的就诊科室",
        "visit_guidance": "可预约门诊就诊，注意观察症状变化。",
    },
}


class DepartmentRecommender:
    """科室推荐：分析结果科室 + 四级急诊制度"""

    def __init__(self, hospital_loader):
        self.loader = hospital_loader
        self._dept_by_id: Dict[str, Dict[str, Any]] = {}
        self._name_to_id: Dict[str, str] = {}

    def recommend(
        self,
        symptom_text: str = "",
        analysis: str = "",
        urgency_level: int = 4,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.loader.hospital_data:
            self.loader.load_data()

        self._build_dept_index()
        urgency_level = max(1, min(4, int(urgency_level or 4)))
        policy = URGENCY_TIER_POLICY.get(urgency_level, URGENCY_TIER_POLICY[4])

        combined = f"{symptom_text or ''}\n{analysis or ''}".strip()
        is_child = self._is_child(combined)
        emergency_id = "pediatric_emergency" if is_child else "emergency"

        analysis_dept_names = self.parse_all_llm_department_names(analysis)
        analysis_dept_ids = self._names_to_ids(analysis_dept_names)

        ordered_ids: List[str] = []
        seen: set = set()

        def add_id(dept_id: str) -> None:
            if dept_id and dept_id in self._dept_by_id and dept_id not in seen:
                seen.add(dept_id)
                ordered_ids.append(dept_id)

        if policy.get("include_emergency") and policy.get("emergency_first"):
            add_id(emergency_id)
            for dept_id in analysis_dept_ids:
                if dept_id != emergency_id:
                    add_id(dept_id)
        else:
            for dept_id in analysis_dept_ids:
                add_id(dept_id)

        if not ordered_ids and policy.get("include_emergency"):
            add_id(emergency_id)

        if not ordered_ids:
            inferred = self._infer_departments_from_analysis(analysis)
            for dept_id in inferred:
                add_id(dept_id)

        results: List[Dict[str, Any]] = []
        for priority, dept_id in enumerate(ordered_ids[:limit], start=1):
            payload = self._dept_payload(dept_id)
            payload["visit_priority"] = priority
            payload["triage_tier"] = urgency_level
            payload["triage_tier_name"] = policy["name"]
            payload["recommend_reason"] = self._build_tier_reason(
                dept_id, emergency_id, analysis_dept_ids, policy, priority
            )
            if priority == 1:
                payload["visit_guidance"] = policy.get("visit_guidance", "")
            results.append(payload)

        return results

    def parse_llm_department_name(self, analysis: str) -> Optional[str]:
        names = self.parse_all_llm_department_names(analysis)
        return names[0] if names else None

    def parse_all_llm_department_names(self, analysis: str) -> List[str]:
        """从分诊分析中解析全部推荐科室（保持出现顺序、去重）"""
        if not analysis:
            return []

        found: List[str] = []
        for pattern in LLM_DEPT_PATTERNS:
            for match in re.finditer(pattern, analysis):
                raw = match.group(1).strip()
                for part in re.split(r"[、,，/及与和]", raw):
                    part = part.strip()
                    if not part:
                        continue
                    resolved = self.resolve_department_name(part)
                    if resolved and resolved not in found:
                        found.append(resolved)
        return found

    def resolve_department_name(self, name: str) -> Optional[str]:
        if not name:
            return None

        cleaned = re.sub(r"[（(].*?[）)]", "", name).strip()
        cleaned = cleaned.split("、")[0].split(",")[0].strip()
        if not cleaned:
            return None

        if cleaned in DEPARTMENT_ALIASES:
            cleaned = DEPARTMENT_ALIASES[cleaned]

        self._build_dept_index()
        if cleaned in self._name_to_id:
            return cleaned

        for dept_name in self._name_to_id:
            if cleaned == dept_name or cleaned in dept_name or dept_name in cleaned:
                return dept_name
        return None

    def _infer_departments_from_analysis(self, analysis: str) -> List[str]:
        """分析正文未写「科室：」时，尝试匹配本院科室名"""
        if not analysis:
            return []

        self._build_dept_index()
        matched: List[str] = []
        for dept_name in sorted(self._name_to_id.keys(), key=len, reverse=True):
            if dept_name in analysis and dept_name not in matched:
                matched.append(dept_name)
        return self._names_to_ids(matched)[:2]

    def _names_to_ids(self, names: List[str]) -> List[str]:
        self._build_dept_index()
        ids: List[str] = []
        for name in names:
            dept_id = self._name_to_id.get(name)
            if dept_id and dept_id not in ids:
                ids.append(dept_id)
        return ids

    def _build_tier_reason(
        self,
        dept_id: str,
        emergency_id: str,
        analysis_dept_ids: List[str],
        policy: Dict[str, Any],
        priority: int,
    ) -> str:
        if dept_id == emergency_id:
            return policy.get("emergency_reason", "分级急诊推荐")
        if dept_id in analysis_dept_ids:
            if priority == 1 and not policy.get("emergency_first"):
                return policy.get("primary_reason", "与分诊分析建议科室一致")
            return policy.get("secondary_reason", "与分诊分析建议科室一致")
        return policy.get("primary_reason", "分诊分析推荐")

    def _build_dept_index(self) -> None:
        if self._dept_by_id:
            return

        for dept_category in self.loader.hospital_data.get("departments", []):
            for sub_dept in dept_category.get("sub_departments", []):
                self._register_department(
                    sub_dept,
                    category=dept_category.get("name", ""),
                    parent_id=None,
                    parent_name=None,
                )
                for specialty in sub_dept.get("sub_specialties", []):
                    self._register_department(
                        specialty,
                        category=dept_category.get("name", ""),
                        parent_id=sub_dept.get("id"),
                        parent_name=sub_dept.get("name"),
                    )

    def _register_department(
        self,
        dept: Dict[str, Any],
        category: str,
        parent_id: Optional[str],
        parent_name: Optional[str],
    ) -> None:
        dept_id = dept["id"]
        name = dept.get("name", "")
        self._dept_by_id[dept_id] = {
            "id": dept_id,
            "name": name,
            "description": dept.get("description", ""),
            "category": category,
            "parent_id": parent_id,
            "parent_name": parent_name,
            "location": dept.get("location", ""),
            "room_number": dept.get("room_number", ""),
            "beds": dept.get("beds", 0),
            "doctors": list(dept.get("doctors", [])),
            "schedule_info": dept.get("schedule_info", {}),
        }
        self._name_to_id[name] = dept_id

    def _dept_payload(self, dept_id: str) -> Dict[str, Any]:
        entry = self._dept_by_id[dept_id]
        payload = {
            "name": entry["name"],
            "description": entry.get("description", ""),
            "category": entry.get("category", ""),
            "location": entry.get("location", ""),
            "room_number": entry.get("room_number", ""),
            "beds": entry.get("beds", 0),
            "doctors": entry.get("doctors", []),
            "schedule_info": entry.get("schedule_info", {}),
        }
        if entry.get("parent_name"):
            payload["parent"] = entry["parent_name"]
        return payload

    def _is_child(self, text: str) -> bool:
        for keyword in CHILD_KEYWORDS:
            if keyword in text:
                return True
        match = re.search(r"(\d+)\s*(岁|周岁|岁半|个月)", text)
        if match:
            age = int(match.group(1))
            unit = match.group(2)
            if unit == "个月":
                return age <= 168
            return age < 14
        return False


def enrich_departments_with_referral(
    symptom_text: str,
    departments: List[Dict[str, Any]],
    smart_referral_tool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """为推荐科室附加智能分流信息"""
    if not smart_referral_tool or not departments:
        return departments, None

    last_referral = None
    for dept in departments:
        dept_name = dept.get("name", "")
        referral_result = smart_referral_tool.analyze_and_recommend(
            symptom_description=symptom_text,
            department_name=dept_name,
        )
        if referral_result.get("success"):
            dept["smart_referral"] = {
                "recommended_doctor": referral_result.get("recommended_doctor"),
                "available_doctors_now": referral_result.get("available_doctors_now", [])[:3],
                "current_time": referral_result.get("current_time"),
                "referral_message": referral_result.get("referral_message", ""),
            }
            last_referral = referral_result

    return departments, last_referral


def get_urgency_tier_meta(level: int) -> Dict[str, Any]:
    """返回 config 与推荐策略一致的急诊分级元数据"""
    level = max(1, min(4, int(level or 4)))
    base = next(
        (u for u in TRIAGE_CONFIG.get("urgency_levels", []) if u.get("level") == level),
        {},
    )
    policy = URGENCY_TIER_POLICY.get(level, URGENCY_TIER_POLICY[4])
    return {**base, "visit_guidance": policy.get("visit_guidance", "")}
