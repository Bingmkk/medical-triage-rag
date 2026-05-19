"""
医院科室数据加载器
将虚构医院的科室分布加载到知识图谱中
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .knowledge_graph import MedicalKnowledgeGraph, MedicalEntity, MedicalRelation


class HospitalDataLoader:
    """医院数据加载器"""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "hospital_departments.json"

        self.data_path = data_path
        self.hospital_data = None
        self.entity_id_counter = 1000

    def load_data(self) -> Dict[str, Any]:
        """加载医院数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.hospital_data = json.load(f)
        return self.hospital_data

    def create_entities_from_hospital(self) -> List[MedicalEntity]:
        """从医院数据创建实体"""
        entities = []

        hospital_entity = MedicalEntity(
            entity_id="hospital_1",
            name=self.hospital_data["hospital_name"],
            entity_type="hospital",
            properties={
                "level": self.hospital_data["hospital_level"],
                "address": self.hospital_data["hospital_address"],
                "phone": self.hospital_data["hospital_phone"],
                "emergency_phone": self.hospital_data["emergency_phone"],
                "buildings": json.dumps(self.hospital_data.get("building_info", {}), ensure_ascii=False)
            }
        )
        entities.append(hospital_entity)

        for dept_category in self.hospital_data["departments"]:
            category_entity = MedicalEntity(
                entity_id=f"cat_{dept_category['id']}",
                name=dept_category["name"],
                entity_type="department_category",
                properties={
                    "description": dept_category.get("description", ""),
                    "location": dept_category.get("location", "")
                }
            )
            entities.append(category_entity)

            if "sub_departments" in dept_category:
                for sub_dept in dept_category["sub_departments"]:
                    sub_dept_entity = MedicalEntity(
                        entity_id=f"dept_{sub_dept['id']}",
                        name=sub_dept["name"],
                        entity_type="department",
                        properties={
                            "description": sub_dept.get("description", ""),
                            "location": sub_dept.get("location", ""),
                            "room_number": sub_dept.get("room_number", ""),
                            "beds": sub_dept.get("beds", 0),
                            "doctors_count": len(sub_dept.get("doctors", []))
                        }
                    )
                    entities.append(sub_dept_entity)

                    if "sub_specialties" in sub_dept:
                        for specialty in sub_dept["sub_specialties"]:
                            doctors_info = []
                            for doctor in specialty.get("doctors", []):
                                doctors_info.append({
                                    "name": doctor["name"],
                                    "title": doctor["title"],
                                    "specialty": doctor.get("specialty", ""),
                                    "schedule": doctor.get("schedule", "")
                                })

                            schedule_info = {
                                "weekday": specialty.get("schedule_info", {}).get("weekday", ""),
                                "weekend": specialty.get("schedule_info", {}).get("weekend", ""),
                                "emergency": specialty.get("schedule_info", {}).get("emergency", "")
                            }

                            specialty_entity = MedicalEntity(
                                entity_id=f"dept_{specialty['id']}",
                                name=specialty["name"],
                                entity_type="department",
                                properties={
                                    "description": specialty.get("description", ""),
                                    "location": specialty.get("location", ""),
                                    "room_number": specialty.get("room_number", ""),
                                    "beds": specialty.get("beds", 0),
                                    "doctors": json.dumps(doctors_info, ensure_ascii=False),
                                    "schedule_info": json.dumps(schedule_info, ensure_ascii=False),
                                    "parent": sub_dept["name"],
                                    "category": dept_category["name"]
                                }
                            )
                            entities.append(specialty_entity)

                            for doctor in specialty.get("doctors", []):
                                doctor_entity = MedicalEntity(
                                    entity_id=f"doc_{specialty['id']}_{doctor['name'].replace(' ', '_')}",
                                    name=doctor["name"],
                                    entity_type="doctor",
                                    properties={
                                        "title": doctor["title"],
                                        "specialty": doctor.get("specialty", ""),
                                        "schedule": doctor.get("schedule", ""),
                                        "department": specialty["name"]
                                    }
                                )
                                entities.append(doctor_entity)

        symptom_mapping = self.hospital_data.get("specialty_mapping", {})
        for symptom_key, dept_ids in symptom_mapping.items():
            symptom_name = symptom_key.replace("_", " ")
            symptom_entity = MedicalEntity(
                entity_id=f"symptom_{symptom_key}",
                name=symptom_name,
                entity_type="symptom",
                properties={"description": f"常见症状: {symptom_name}"}
            )
            entities.append(symptom_entity)

        return entities

    def create_relations_from_hospital(self, entities: List[MedicalEntity]) -> List[MedicalRelation]:
        """从医院数据创建关系"""
        relations = []
        entity_map = {e.name: e.id for e in entities}

        hospital_name = self.hospital_data["hospital_name"]
        if hospital_name in entity_map:
            hospital_id = entity_map[hospital_name]

            for dept_category in self.hospital_data["departments"]:
                category_name = dept_category["name"]
                if category_name in entity_map:
                    relations.append(MedicalRelation(
                        source_id=hospital_id,
                        target_id=entity_map[category_name],
                        relation_type="has_department_category"
                    ))

                if "sub_departments" in dept_category:
                    for sub_dept in dept_category["sub_departments"]:
                        sub_dept_name = sub_dept["name"]
                        if sub_dept_name in entity_map:
                            relations.append(MedicalRelation(
                                source_id=entity_map[category_name],
                                target_id=entity_map[sub_dept_name],
                                relation_type="has_sub_department"
                            ))

                        if "sub_specialties" in sub_dept:
                            for specialty in sub_dept["sub_specialties"]:
                                specialty_name = specialty["name"]
                                if specialty_name in entity_map:
                                    relations.append(MedicalRelation(
                                        source_id=entity_map[sub_dept_name],
                                        target_id=entity_map[specialty_name],
                                        relation_type="has_specialty"
                                    ))

                                for doctor in specialty.get("doctors", []):
                                    doctor_id = f"doc_{specialty['id']}_{doctor['name'].replace(' ', '_')}"
                                    if doctor_id in [e.id for e in entities]:
                                        relations.append(MedicalRelation(
                                            source_id=entity_map[specialty_name],
                                            target_id=doctor_id,
                                            relation_type="has_doctor"
                                        ))

        symptom_mapping = self.hospital_data.get("specialty_mapping", {})
        for symptom_key, dept_ids in symptom_mapping.items():
            symptom_name = symptom_key.replace("_", " ")
            if symptom_name in entity_map:
                symptom_id = entity_map[symptom_name]
                for dept_id in dept_ids:
                    dept_id_formatted = f"dept_{dept_id}"
                    if dept_id_formatted in entity_map.values():
                        for name, eid in entity_map.items():
                            if eid == dept_id_formatted:
                                relations.append(MedicalRelation(
                                    source_id=symptom_id,
                                    target_id=eid,
                                    relation_type="treatable_in"
                                ))
                                break

        return relations

    def build_knowledge_graph(self) -> MedicalKnowledgeGraph:
        """构建包含医院科室的知识图谱"""
        self.load_data()

        kg = MedicalKnowledgeGraph(name="hospital_medical_kg")

        entities = self.create_entities_from_hospital()
        for entity in entities:
            kg.add_entity(entity)

        relations = self.create_relations_from_hospital(entities)
        for relation in relations:
            try:
                kg.add_relation(relation)
            except ValueError:
                continue

        return kg

    def recommend_departments(
        self,
        symptom_text: str,
        analysis: str = "",
        urgency_level: int = 4,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """根据症状、分诊分析与紧急程度推荐科室（评分排序）"""
        from .department_recommender import DepartmentRecommender

        if not self.hospital_data:
            self.load_data()
        return DepartmentRecommender(self).recommend(
            symptom_text=symptom_text,
            analysis=analysis,
            urgency_level=urgency_level,
            limit=limit,
        )

    def get_department_by_symptom(self, symptom_name: str) -> List[Dict[str, Any]]:
        """根据症状推荐科室（兼容旧接口）"""
        return self.recommend_departments(symptom_name, limit=5)

    def _extract_chinese_keywords(self, text: str) -> List[str]:
        """从文本中提取中文关键词"""
        import re
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]+', text.lower())
        keywords = []
        for chars in chinese_chars:
            if len(chars) >= 2:
                keywords.append(chars)
        return keywords

    def get_all_departments(self) -> Dict[str, Any]:
        """获取所有科室信息"""
        if not self.hospital_data:
            self.load_data()

        return {
            "hospital_info": {
                "name": self.hospital_data["hospital_name"],
                "level": self.hospital_data["hospital_level"],
                "address": self.hospital_data["hospital_address"],
                "phone": self.hospital_data["hospital_phone"],
                "emergency_phone": self.hospital_data["emergency_phone"],
                "buildings": self.hospital_data.get("building_info", {})
            },
            "departments": self.hospital_data["departments"],
            "notice": {
                "emergency": self.hospital_data.get("emergency_notice", ""),
                "appointment": self.hospital_data.get("appointment_notice", "")
            }
        }

    def get_doctor_info(self, doctor_name: str) -> Optional[Dict[str, Any]]:
        """获取医生详细信息"""
        if not self.hospital_data:
            self.load_data()

        for dept_category in self.hospital_data["departments"]:
            if "sub_departments" in dept_category:
                for sub_dept in dept_category["sub_departments"]:
                    if "sub_specialties" in sub_dept:
                        for specialty in sub_dept["sub_specialties"]:
                            for doctor in specialty.get("doctors", []):
                                if doctor["name"] == doctor_name:
                                    return {
                                        "name": doctor["name"],
                                        "title": doctor["title"],
                                        "specialty": doctor.get("specialty", ""),
                                        "schedule": doctor.get("schedule", ""),
                                        "department": specialty["name"],
                                        "location": specialty.get("location", ""),
                                        "room_number": specialty.get("room_number", "")
                                    }
        return None

    def get_department_info(self, department_name: str) -> Optional[Dict[str, Any]]:
        """获取科室详细信息"""
        if not self.hospital_data:
            self.load_data()

        from .department_recommender import DepartmentRecommender

        resolved_name = DepartmentRecommender(self).resolve_department_name(department_name)
        if resolved_name:
            department_name = resolved_name

        for dept_category in self.hospital_data["departments"]:
            if "sub_departments" in dept_category:
                for sub_dept in dept_category["sub_departments"]:
                    if sub_dept["name"] == department_name:
                        return {
                            "name": sub_dept["name"],
                            "description": sub_dept.get("description", ""),
                            "category": dept_category["name"],
                            "location": sub_dept.get("location", ""),
                            "room_number": sub_dept.get("room_number", ""),
                            "beds": sub_dept.get("beds", 0),
                            "doctors": sub_dept.get("doctors", []),
                            "schedule_info": sub_dept.get("schedule_info", {})
                        }

                    if "sub_specialties" in sub_dept:
                        for specialty in sub_dept["sub_specialties"]:
                            if specialty["name"] == department_name:
                                return {
                                    "name": specialty["name"],
                                    "description": specialty.get("description", ""),
                                    "category": dept_category["name"],
                                    "parent": sub_dept["name"],
                                    "location": specialty.get("location", ""),
                                    "room_number": specialty.get("room_number", ""),
                                    "beds": specialty.get("beds", 0),
                                    "doctors": specialty.get("doctors", []),
                                    "schedule_info": specialty.get("schedule_info", {})
                                }
        return None


def load_hospital_knowledge_graph() -> MedicalKnowledgeGraph:
    """加载医院科室知识图谱"""
    loader = HospitalDataLoader()
    return loader.build_knowledge_graph()


def get_recommended_departments(symptom_name: str) -> List[Dict[str, Any]]:
    """根据症状获取推荐科室"""
    loader = HospitalDataLoader()
    return loader.get_department_by_symptom(symptom_name)


if __name__ == "__main__":
    print("=" * 70)
    print("医院科室数据加载测试")
    print("=" * 70)

    loader = HospitalDataLoader()
    kg = loader.build_knowledge_graph()

    print(f"\n✓ 知识图谱创建成功")
    print(f"  - 实体数量: {len(kg.graph.nodes)}")
    print(f"  - 关系数量: {len(kg.graph.edges)}")

    print("\n科室实体示例：")
    clinical = kg.get_entity_by_name("心内科")
    if clinical:
        print(f"  - {clinical['name']} (ID: {clinical['id']})")
        print(f"    位置: {clinical.get('properties', {}).get('location', 'N/A')}")
        print(f"    门牌号: {clinical.get('properties', {}).get('room_number', 'N/A')}")
        print(f"    床位数: {clinical.get('properties', {}).get('beds', 0)}")

    print("\n医生实体示例：")
    doctor = kg.get_entity_by_name("张明华")
    if doctor:
        print(f"  - {doctor['name']} (ID: {doctor['id']})")
        print(f"    职称: {doctor.get('properties', {}).get('title', 'N/A')}")
        print(f"    专业: {doctor.get('properties', {}).get('specialty', 'N/A')}")
        print(f"    排班: {doctor.get('properties', {}).get('schedule', 'N/A')}")

    print("\n根据症状推荐科室测试：")
    test_symptoms = ["胸痛", "头痛", "腹痛", "糖尿病"]

    for symptom in test_symptoms:
        print(f"\n  症状: {symptom}")
        departments = loader.get_department_by_symptom(symptom)
        if departments:
            for dept in departments[:3]:
                print(f"    → {dept['name']}: {dept['location']}")
                print(f"       门牌号: {dept.get('room_number', 'N/A')}")
                if dept.get('doctors'):
                    print(f"       医生:")
                    for doc in dept['doctors'][:2]:
                        print(f"         - {doc['name']} ({doc['title']}): {doc['specialty']}, 排班: {doc['schedule']}")
        else:
            print(f"    → 未找到对应科室")

    hospital_info = loader.get_all_departments()
    print(f"\n\n医院信息：")
    print(f"  名称: {hospital_info['hospital_info']['name']}")
    print(f"  等级: {hospital_info['hospital_info']['level']}")
    print(f"  地址: {hospital_info['hospital_info']['address']}")
    print(f"  总机: {hospital_info['hospital_info']['phone']}")
    print(f"  急诊: {hospital_info['hospital_info']['emergency_phone']}")
    print(f"  建筑分布: {hospital_info['hospital_info']['buildings']}")

    print("\n获取医生信息测试：")
    doc_info = loader.get_doctor_info("张明华")
    if doc_info:
        print(f"  医生: {doc_info['name']}")
        print(f"  职称: {doc_info['title']}")
        print(f"  专业: {doc_info['specialty']}")
        print(f"  科室: {doc_info['department']}")
        print(f"  位置: {doc_info['location']}")
        print(f"  门牌号: {doc_info['room_number']}")
        print(f"  排班: {doc_info['schedule']}")

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)
