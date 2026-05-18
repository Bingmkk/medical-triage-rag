"""
智能分流工具
根据排班时间、医生专业和当前时间自动推荐最合适的医生
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, time
from .hospital_loader import HospitalDataLoader


class ScheduleAnalyzer:
    """排班分析器"""

    def __init__(self, hospital_loader: Optional[HospitalDataLoader] = None):
        self.hospital_loader = hospital_loader or HospitalDataLoader()
        self.hospital_loader.load_data()

    def get_current_time_info(self) -> Dict[str, Any]:
        """获取当前时间信息"""
        now = datetime.now()
        
        weekday_map = {
            0: "周一",
            1: "周二", 
            2: "周三",
            3: "周四",
            4: "周五",
            5: "周六",
            6: "周日"
        }
        
        return {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": weekday_map[now.weekday()],
            "weekday_num": now.weekday(),
            "hour": now.hour,
            "minute": now.minute,
            "is_weekend": now.weekday() >= 5,
            "is_morning": 8 <= now.hour < 12,
            "is_afternoon": 12 <= now.hour < 18,
            "is_evening": now.hour >= 18 or now.hour < 8
        }

    def parse_schedule(self, schedule_str: str) -> List[Dict[str, Any]]:
        """解析排班字符串"""
        schedules = []
        
        weekday_map = {
            "周一": 0, "周二": 1, "周三": 2, "周四": 3, 
            "周五": 4, "周六": 5, "周日": 6
        }
        
        parts = schedule_str.split("、")
        
        for part in parts:
            part = part.strip()
            
            for day_name, day_num in weekday_map.items():
                if day_name in part:
                    schedule_entry = {
                        "weekday": day_num,
                        "weekday_name": day_name,
                        "time_slot": "unknown"
                    }
                    
                    if "上午" in part:
                        schedule_entry["time_slot"] = "morning"
                        schedule_entry["start_hour"] = 8
                        schedule_entry["end_hour"] = 12
                    elif "下午" in part:
                        schedule_entry["time_slot"] = "afternoon"
                        schedule_entry["start_hour"] = 13
                        schedule_entry["end_hour"] = 17
                    elif "全天" in part:
                        schedule_entry["time_slot"] = "allday"
                        schedule_entry["start_hour"] = 8
                        schedule_entry["end_hour"] = 17
                    
                    schedules.append(schedule_entry)
                    break
        
        return schedules

    def is_doctor_available(self, schedule_str: str, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """判断医生当前是否出诊"""
        if current_time is None:
            current_time = datetime.now()
        
        schedules = self.parse_schedule(schedule_str)
        
        current_weekday = current_time.weekday()
        current_hour = current_time.hour
        
        for schedule in schedules:
            if schedule["weekday"] == current_weekday:
                time_slot = schedule.get("time_slot", "unknown")
                
                if time_slot == "allday":
                    if 8 <= current_hour < 17:
                        return True, f"今日全天出诊 (8:00-17:00)"
                elif time_slot == "morning":
                    if 8 <= current_hour < 12:
                        return True, f"今日上午出诊 (8:00-12:00)"
                elif time_slot == "afternoon":
                    if 13 <= current_hour < 17:
                        return True, f"今日下午出诊 (13:00-17:00)"
        
        return False, "当前时间不在出诊时段"

    def get_next_available_time(self, schedule_str: str, current_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """获取下次出诊时间"""
        if current_time is None:
            current_time = datetime.now()
        
        schedules = self.parse_schedule(schedule_str)
        
        if not schedules:
            return None
        
        current_weekday = current_time.weekday()
        current_hour = current_time.hour
        
        weekday_order = list(range(current_weekday, 7)) + list(range(0, current_weekday))
        
        for day_offset, day_num in enumerate(weekday_order):
            for schedule in schedules:
                if schedule["weekday"] == day_num:
                    time_slot = schedule.get("time_slot", "unknown")
                    
                    if day_offset == 0:
                        if time_slot == "morning" and current_hour < 12:
                            return {
                                "weekday": schedule["weekday_name"],
                                "time_slot": "上午",
                                "days_from_now": 0,
                                "message": f"今日上午 (8:00-12:00)"
                            }
                        elif time_slot == "afternoon" and current_hour < 17:
                            return {
                                "weekday": schedule["weekday_name"],
                                "time_slot": "下午",
                                "days_from_now": 0,
                                "message": f"今日下午 (13:00-17:00)"
                            }
                    else:
                        time_msg = "全天" if time_slot == "allday" else ("上午" if time_slot == "morning" else "下午")
                        return {
                            "weekday": schedule["weekday_name"],
                            "time_slot": time_msg,
                            "days_from_now": day_offset,
                            "message": f"{schedule['weekday_name']}{time_msg}"
                        }
        
        return None

    def match_specialty(self, doctor_specialty: str, symptom_keywords: List[str]) -> float:
        """匹配医生专业与症状关键词"""
        if not symptom_keywords:
            return 0.0
        
        doctor_specialty_lower = doctor_specialty.lower()
        match_count = 0
        
        specialty_keywords = {
            "冠心病": ["胸痛", "心绞痛", "心脏", "心悸", "胸痛"],
            "高血压": ["高血压", "头痛", "头晕", "血压"],
            "心力衰竭": ["心力衰竭", "呼吸困难", "水肿", "心衰"],
            "心律失常": ["心律失常", "心悸", "心跳", "心动过速"],
            "心肌病": ["心肌病", "心脏扩大", "心衰"],
            "脑血管病": ["脑梗", "中风", "偏瘫", "头痛", "头晕"],
            "帕金森": ["帕金森", "震颤", "僵硬", "运动障碍"],
            "癫痫": ["癫痫", "抽搐", "发作"],
            "肺部感染": ["肺炎", "咳嗽", "发热", "呼吸困难"],
            "哮喘": ["哮喘", "喘息", "呼吸困难", "咳嗽"],
            "慢阻肺": ["慢阻肺", "copd", "呼吸困难", "咳嗽"],
            "肺癌": ["肺癌", "咳嗽", "咯血", "胸痛"],
            "胃肠疾病": ["胃痛", "腹痛", "腹泻", "便秘", "消化不良"],
            "肝病": ["肝病", "肝炎", "黄疸", "肝功能"],
            "糖尿病": ["糖尿病", "血糖", "多饮", "多尿"],
            "甲状腺": ["甲状腺", "甲亢", "甲减", "脖子粗"],
            "骨折": ["骨折", "外伤", "疼痛", "肿胀"],
            "关节置换": ["关节", "关节炎", "关节痛"],
            "脊柱": ["脊柱", "腰椎", "颈椎", "腰痛", "背痛"],
            "创伤": ["外伤", "骨折", "伤口", "出血"],
            "小儿呼吸": ["儿童咳嗽", "小儿肺炎", "儿童发热"],
            "小儿消化": ["儿童腹泻", "小儿腹痛", "儿童消化不良"],
            "白内障": ["白内障", "视力下降", "视物模糊"],
            "青光眼": ["青光眼", "眼压", "视力下降"],
            "中耳炎": ["中耳炎", "耳痛", "听力下降"],
            "鼻炎": ["鼻炎", "鼻塞", "流涕", "打喷嚏"],
            "咽喉": ["咽喉", "嗓子痛", "声音嘶哑"],
            "皮肤": ["皮疹", "瘙痒", "湿疹", "痤疮"],
            "抑郁": ["抑郁", "情绪低落", "失眠"],
            "焦虑": ["焦虑", "紧张", "心慌"],
            "肿瘤": ["肿瘤", "癌症", "肿块"],
            "感染": ["发热", "感染", "炎症"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if specialty in doctor_specialty_lower:
                for keyword in symptom_keywords:
                    if keyword in keywords:
                        match_count += 1
        
        return match_count / len(symptom_keywords) if symptom_keywords else 0.0


class SmartReferralTool:
    """智能分流工具"""

    def __init__(self, hospital_loader: Optional[HospitalDataLoader] = None):
        self.hospital_loader = hospital_loader or HospitalDataLoader()
        self.hospital_loader.load_data()
        self.schedule_analyzer = ScheduleAnalyzer(self.hospital_loader)

    def analyze_and_recommend(
        self,
        symptom_description: str,
        department_name: str,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """分析并推荐最合适的医生"""
        
        if current_time is None:
            current_time = datetime.now()
        
        time_info = self.schedule_analyzer.get_current_time_info()
        
        department_info = self.hospital_loader.get_department_info(department_name)
        
        if not department_info:
            return {
                "success": False,
                "error": f"未找到科室: {department_name}",
                "current_time": time_info
            }
        
        symptom_keywords = self._extract_symptom_keywords(symptom_description)
        
        doctors = department_info.get("doctors", [])
        
        if not doctors:
            return {
                "success": True,
                "department": department_info,
                "current_time": time_info,
                "available_doctors": [],
                "recommended_doctor": None,
                "message": "该科室暂无医生排班信息"
            }
        
        analyzed_doctors = []
        
        for doctor in doctors:
            schedule_str = doctor.get("schedule", "")
            is_available, availability_msg = self.schedule_analyzer.is_doctor_available(
                schedule_str, current_time
            )
            
            next_time = self.schedule_analyzer.get_next_available_time(
                schedule_str, current_time
            )
            
            specialty_match = self.schedule_analyzer.match_specialty(
                doctor.get("specialty", ""), symptom_keywords
            )
            
            title_priority = self._get_title_priority(doctor.get("title", ""))
            
            analyzed_doctors.append({
                "name": doctor["name"],
                "title": doctor["title"],
                "specialty": doctor.get("specialty", ""),
                "schedule": schedule_str,
                "is_available_now": is_available,
                "availability_message": availability_msg,
                "next_available_time": next_time,
                "specialty_match_score": specialty_match,
                "title_priority": title_priority,
                "overall_score": self._calculate_overall_score(
                    is_available, specialty_match, title_priority
                )
            })
        
        analyzed_doctors.sort(key=lambda x: x["overall_score"], reverse=True)
        
        available_doctors = [d for d in analyzed_doctors if d["is_available_now"]]
        
        recommended_doctor = analyzed_doctors[0] if analyzed_doctors else None
        
        return {
            "success": True,
            "department": {
                "name": department_info["name"],
                "location": department_info.get("location", ""),
                "room_number": department_info.get("room_number", ""),
                "schedule_info": department_info.get("schedule_info", {})
            },
            "current_time": time_info,
            "symptom_keywords": symptom_keywords,
            "all_doctors": analyzed_doctors,
            "available_doctors_now": available_doctors,
            "recommended_doctor": recommended_doctor,
            "referral_message": self._generate_referral_message(
                recommended_doctor, available_doctors, time_info
            )
        }

    def _extract_symptom_keywords(self, symptom_description: str) -> List[str]:
        """提取症状关键词"""
        keywords = []
        
        symptom_patterns = [
            "胸痛", "头痛", "腹痛", "腰痛", "关节痛",
            "发热", "咳嗽", "呼吸困难", "心悸", "头晕",
            "恶心", "呕吐", "腹泻", "便秘",
            "高血压", "糖尿病", "骨折", "外伤",
            "视力下降", "听力下降", "皮疹", "瘙痒"
        ]
        
        desc_lower = symptom_description.lower()
        
        for pattern in symptom_patterns:
            if pattern in desc_lower:
                keywords.append(pattern)
        
        return keywords

    def _get_title_priority(self, title: str) -> int:
        """获取职称优先级"""
        title_priority = {
            "主任医师": 4,
            "副主任医师": 3,
            "主治医师": 2,
            "住院医师": 1
        }
        return title_priority.get(title, 0)

    def _calculate_overall_score(
        self,
        is_available: bool,
        specialty_match: float,
        title_priority: int
    ) -> float:
        """计算综合评分"""
        availability_score = 100 if is_available else 0
        specialty_score = specialty_match * 50
        title_score = title_priority * 10
        
        return availability_score + specialty_score + title_score

    def _generate_referral_message(
        self,
        recommended_doctor: Optional[Dict[str, Any]],
        available_doctors: List[Dict[str, Any]],
        time_info: Dict[str, Any]
    ) -> str:
        """生成分流建议消息"""
        if not recommended_doctor:
            return "暂无医生排班信息，请咨询导诊台"
        
        if recommended_doctor["is_available_now"]:
            return (
                f"推荐挂号: {recommended_doctor['name']} ({recommended_doctor['title']})\n"
                f"专业: {recommended_doctor['specialty']}\n"
                f"状态: {recommended_doctor['availability_message']}\n"
                f"匹配度: {recommended_doctor['specialty_match_score']*100:.0f}%"
            )
        else:
            next_time = recommended_doctor.get("next_available_time")
            if next_time:
                return (
                    f"推荐预约: {recommended_doctor['name']} ({recommended_doctor['title']})\n"
                    f"专业: {recommended_doctor['specialty']}\n"
                    f"下次出诊: {next_time['message']}\n"
                    f"匹配度: {recommended_doctor['specialty_match_score']*100:.0f}%"
                )
            else:
                return (
                    f"推荐医生: {recommended_doctor['name']} ({recommended_doctor['title']})\n"
                    f"专业: {recommended_doctor['specialty']}\n"
                    f"请咨询导诊台获取具体出诊时间"
                )


def create_smart_referral_tool(hospital_loader: Optional[HospitalDataLoader] = None) -> SmartReferralTool:
    """创建智能分流工具"""
    return SmartReferralTool(hospital_loader)


if __name__ == "__main__":
    print("=" * 70)
    print("智能分流工具测试")
    print("=" * 70)
    
    tool = create_smart_referral_tool()
    
    print("\n当前时间信息:")
    time_info = tool.schedule_analyzer.get_current_time_info()
    for key, value in time_info.items():
        print(f"  {key}: {value}")
    
    print("\n测试场景1: 胸痛患者推荐心内科医生")
    result = tool.analyze_and_recommend(
        symptom_description="我最近一周胸痛、呼吸困难，有时感觉心跳很快",
        department_name="心内科"
    )
    
    print("\n分流结果:")
    print(f"  科室: {result['department']['name']}")
    print(f"  位置: {result['department']['location']}")
    print(f"  门牌号: {result['department']['room_number']}")
    
    if result.get("recommended_doctor"):
        doc = result["recommended_doctor"]
        print(f"\n推荐医生: {doc['name']} ({doc['title']})")
        print(f"  专业: {doc['specialty']}")
        print(f"  排班: {doc['schedule']}")
        print(f"  当前是否出诊: {'是' if doc['is_available_now'] else '否'}")
        print(f"  专业匹配度: {doc['specialty_match_score']*100:.0f}%")
        print(f"  综合评分: {doc['overall_score']:.1f}")
    
    print("\n分流建议:")
    print(result["referral_message"])
    
    print("\n当前可挂号医生:")
    for doc in result.get("available_doctors_now", []):
        print(f"  - {doc['name']} ({doc['title']}): {doc['specialty']}")
    
    print("\n" + "=" * 70)
