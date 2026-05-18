"""
医学知识图谱构建和管理模块
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import networkx as nx
from pathlib import Path
import re
from config import KG_STORE_DIR


class MedicalEntity:
    """医学实体类"""

    def __init__(self, entity_id: str, name: str, entity_type: str, properties: Dict[str, Any] = None):
        self.id = entity_id
        self.name = name
        self.type = entity_type
        self.properties = properties or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MedicalEntity':
        return cls(
            entity_id=data["id"],
            name=data["name"],
            entity_type=data["type"],
            properties=data.get("properties", {})
        )


class MedicalRelation:
    """医学关系类"""

    def __init__(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None):
        self.source = source_id
        self.target = target_id
        self.type = relation_type
        self.properties = properties or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MedicalRelation':
        return cls(
            source_id=data["source"],
            target_id=data["target"],
            relation_type=data["type"],
            properties=data.get("properties", {})
        )


class MedicalKnowledgeGraph:
    """医学知识图谱"""

    def __init__(self, name: str = "medical_kg"):
        self.name = name
        self.graph = nx.DiGraph()
        self.entity_index = {}  # 实体名称到ID的映射
        self.entity_types = [
            "disease", "symptom", "drug", "treatment", "department",
            "body_part", "test", "procedure", "contagion"
        ]

    def add_entity(self, entity: MedicalEntity) -> None:
        """添加实体"""
        self.graph.add_node(entity.id, **entity.to_dict())
        self.entity_index[entity.name.lower()] = entity.id

        if entity.name not in self.entity_index:
            self.entity_index[entity.name] = entity.id

    def add_relation(self, relation: MedicalRelation) -> None:
        """添加关系"""
        if relation.source not in self.graph or relation.target not in self.graph:
            raise ValueError("关系两端必须在图中存在")

        self.graph.add_edge(
            relation.source,
            relation.target,
            **relation.to_dict()
        )

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取实体信息"""
        if entity_id in self.graph:
            return self.graph.nodes[entity_id]
        return None

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称查找实体"""
        entity_id = self.entity_index.get(name.lower())
        if entity_id:
            return self.get_entity(entity_id)
        return None

    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取邻居节点"""
        if entity_id not in self.graph:
            return []

        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.edges[entity_id, neighbor]
            if relation_type is None or edge_data.get("type") == relation_type:
                neighbors.append({
                    "entity": self.graph.nodes[neighbor],
                    "relation": edge_data
                })
        return neighbors

    def find_paths(self, source_id: str, target_id: str, max_length: int = 3) -> List[List[str]]:
        """查找两个实体间的路径"""
        try:
            return list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_length))
        except nx.NetworkXNoPath:
            return []

    def query_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """根据类型查询实体"""
        return [
            self.graph.nodes[node]
            for node in self.graph.nodes
            if self.graph.nodes[node].get("type") == entity_type
        ]

    def get_disease_relations(self, disease_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """获取疾病的所有关联信息"""
        relations = {
            "symptoms": [],
            "treatments": [],
            "drugs": [],
            "departments": [],
            "tests": [],
            "contagions": []
        }

        for neighbor in self.graph.neighbors(disease_id):
            edge_data = self.graph.edges[disease_id, neighbor]
            neighbor_data = self.graph.nodes[neighbor]
            relation_type = edge_data.get("type", "")

            if "symptom" in relation_type:
                relations["symptoms"].append(neighbor_data)
            elif "treatment" in relation_type:
                relations["treatments"].append(neighbor_data)
            elif "drug" in relation_type:
                relations["drugs"].append(neighbor_data)
            elif "department" in relation_type:
                relations["departments"].append(neighbor_data)
            elif "test" in relation_type:
                relations["tests"].append(neighbor_data)
            elif "contagion" in relation_type:
                relations["contagions"].append(neighbor_data)

        return relations

    def save(self, filepath: Optional[Path] = None) -> None:
        """保存知识图谱"""
        if filepath is None:
            filepath = KG_STORE_DIR / f"{self.name}.json"

        data = {
            "name": self.name,
            "entities": [self.graph.nodes[node] for node in self.graph.nodes],
            "relations": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    **self.graph.edges[edge]
                }
                for edge in self.graph.edges
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'MedicalKnowledgeGraph':
        """加载知识图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        kg = cls(name=data["name"])

        for entity in data.get("entities", []):
            kg.add_entity(MedicalEntity.from_dict(entity))

        for relation in data.get("relations", []):
            kg.add_relation(MedicalRelation.from_dict(relation))

        return kg

    def load_graphml(self, filepath: str) -> None:
        """从GraphML文件加载知识图谱

        Args:
            filepath: GraphML文件路径
        """
        import networkx as nx

        G = nx.read_graphml(filepath)

        for node_id, node_data in G.nodes(data=True):
            entity_type = node_data.get('type', 'unknown')
            entity_name = node_data.get('name', node_id)
            entity_desc = node_data.get('description', '')

            entity = MedicalEntity(
                entity_id=node_id,
                name=entity_name,
                entity_type=entity_type,
                properties=dict(node_data)
            )
            self.add_entity(entity)

        for u, v, edge_data in G.edges(data=True):
            relation_type = edge_data.get('relation', edge_data.get('label', 'related_to'))
            source_id = u
            target_id = v
            properties = dict(edge_data)

            relation = MedicalRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                properties=properties
            )
            try:
                self.add_relation(relation)
            except ValueError:
                continue

    def build_from_text(self, text: str, entity_extractor) -> None:
        """
        从文本构建知识图谱

        Args:
            text: 输入文本
            entity_extractor: 实体提取器（可调用对象）
        """
        entities, relations = entity_extractor.extract(text)

        for entity in entities:
            self.add_entity(entity)

        for relation in relations:
            try:
                self.add_relation(relation)
            except ValueError:
                continue


class MedicalEntityExtractor:
    """医学实体提取器（基于规则的简单实现）"""

    def __init__(self):
        self.entity_patterns = {
            "disease": [
                r"高血压", r"糖尿病", r"心脏病", r"肺炎", r"胃炎", r"肝炎",
                r"癌症", r"肿瘤", r"白血病", r"艾滋病", r"流感", r"新冠",
                r"支气管炎", r"哮喘", r"癫痫", r"脑卒中", r"心肌梗死"
            ],
            "symptom": [
                r"头痛", r"发热", r"咳嗽", r"胸痛", r"腹痛", r"腹泻",
                r"呕吐", r"恶心", r"头晕", r"乏力", r"呼吸困难", r"皮疹",
                r"出血", r"水肿", r"失眠", r"焦虑"
            ],
            "drug": [
                r"阿司匹林", r"布洛芬", r"青霉素", r"头孢", r"胰岛素",
                r"降压药", r"止痛药", r"抗生素", r"维生素", r"中药"
            ],
            "department": [
                r"内科", r"外科", r"儿科", r"妇产科", r"骨科", r"神经科",
                r"心内科", r"消化内科", r"呼吸科", r"急诊科"
            ],
            "treatment": [
                r"手术", r"化疗", r"放疗", r"药物治疗", r"物理治疗",
                r"介入治疗", r"中医治疗", r"康复治疗"
            ],
            "body_part": [
                r"心脏", r"肝脏", r"肾脏", r"肺", r"胃", r"肠",
                r"脑", r"血管", r"骨骼", r"肌肉", r"皮肤"
            ]
        }

    def extract(self, text: str) -> Tuple[List[MedicalEntity], List[MedicalRelation]]:
        """提取实体和关系"""
        entities = []
        entity_mentions = {}
        entity_id = 0

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    name = match.group()
                    if name not in entity_mentions:
                        entity_id += 1
                        entity = MedicalEntity(
                            entity_id=f"{entity_type}_{entity_id}",
                            name=name,
                            entity_type=entity_type
                        )
                        entities.append(entity)
                        entity_mentions[name] = entity.entity_id

        relations = self._extract_relations(text, entity_mentions)

        return entities, relations

    def _extract_relations(self, text: str, entity_mentions: Dict[str, str]) -> List[MedicalRelation]:
        """提取关系（简化版）"""
        relations = []
        relation_id = 0

        relation_keywords = {
            ("disease", "symptom"): ["引起", "导致", "表现为", "症状"],
            ("disease", "drug"): ["使用", "服用", "治疗", "用药"],
            ("disease", "treatment"): ["采用", "进行", "治疗"],
            ("disease", "department"): ["就诊", "科室", "看"],
            ("symptom", "body_part"): ["在", "位于"]
        }

        for (source_type, target_type), keywords in relation_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    for source_name, source_id in entity_mentions.items():
                        source_entity_type = source_name not in ["source", "target"] and source_type
                        if source_type == "disease":
                            source_patterns = self.entity_patterns.get("disease", [])
                            if source_name in [p.strip("()") for p in source_patterns]:
                                relation_id += 1
                                relations.append(MedicalRelation(
                                    source_id=source_id,
                                    target_id="unknown",
                                    relation_type=f"{source_type}_to_{target_type}"
                                ))

        return relations


def create_sample_knowledge_graph() -> MedicalKnowledgeGraph:
    """创建示例医学知识图谱"""
    kg = MedicalKnowledgeGraph(name="sample_medical_kg")

    diseases = [
        ("d1", "高血压", {"description": "以体循环动脉血压升高为主要表现的临床综合征"}),
        ("d2", "糖尿病", {"description": "以高血糖为特征的代谢性疾病"}),
        ("d3", "冠心病", {"description": "冠状动脉粥样硬化性心脏病的简称"}),
        ("d4", "肺炎", {"description": "肺部感染性疾病"}),
        ("d5", "急性心肌梗死", {"description": "心肌缺血性坏死"}),
    ]

    for entity_id, name, props in diseases:
        kg.add_entity(MedicalEntity(entity_id, name, "disease", props))

    symptoms = [
        ("s1", "头痛", {"description": "头部疼痛"}),
        ("s2", "胸痛", {"description": "胸部疼痛"}),
        ("s3", "呼吸困难", {"description": "感觉呼吸不顺畅"}),
        ("s4", "乏力", {"description": "身体疲倦无力"}),
        ("s5", "多饮多尿", {"description": "饮水量和尿量增多"}),
        ("s6", "发热", {"description": "体温升高"}),
        ("s7", "咳嗽", {"description": "咳嗽症状"}),
    ]

    for entity_id, name, props in symptoms:
        kg.add_entity(MedicalEntity(entity_id, name, "symptom", props))

    drugs = [
        ("dr1", "降压药", {"description": "用于降低血压的药物"}),
        ("dr2", "硝酸甘油", {"description": "用于缓解心绞痛"}),
        ("dr3", "胰岛素", {"description": "用于控制血糖"}),
        ("dr4", "抗生素", {"description": "用于抗菌治疗"}),
    ]

    for entity_id, name, props in drugs:
        kg.add_entity(MedicalEntity(entity_id, name, "drug", props))

    departments = [
        ("dept1", "心内科", {"description": "心血管疾病专科"}),
        ("dept2", "内分泌科", {"description": "内分泌疾病专科"}),
        ("dept3", "呼吸科", {"description": "呼吸系统疾病专科"}),
        ("dept4", "急诊科", {"description": "急诊急救"}),
    ]

    for entity_id, name, props in departments:
        kg.add_entity(MedicalEntity(entity_id, name, "department", props))

    treatments = [
        ("t1", "药物治疗", {"description": "使用药物进行治疗"}),
        ("t2", "介入治疗", {"description": "微创治疗方法"}),
        ("t3", "冠脉支架", {"description": "冠状动脉支架植入术"}),
    ]

    for entity_id, name, props in treatments:
        kg.add_entity(MedicalEntity(entity_id, name, "treatment", props))

    body_parts = [
        ("bp1", "心脏", {"description": "重要的循环器官"}),
        ("bp2", "血管", {"description": "血液流通的管道"}),
        ("bp3", "肺", {"description": "呼吸器官"}),
    ]

    for entity_id, name, props in body_parts:
        kg.add_entity(MedicalEntity(entity_id, name, "body_part", props))

    relations = [
        ("d1", "s1", "cause", {}),
        ("d1", "dr1", "treat_with", {}),
        ("d1", "dept1", "treated_in", {}),
        ("d2", "s4", "cause", {}),
        ("d2", "s5", "cause", {}),
        ("d2", "dr3", "treat_with", {}),
        ("d2", "dept2", "treated_in", {}),
        ("d3", "s2", "cause", {}),
        ("d3", "s3", "cause", {}),
        ("d3", "dr2", "treat_with", {}),
        ("d3", "dept1", "treated_in", {}),
        ("d3", "bp2", "located_in", {}),
        ("d4", "s6", "cause", {}),
        ("d4", "s7", "cause", {}),
        ("d4", "s3", "cause", {}),
        ("d4", "dr4", "treat_with", {}),
        ("d4", "dept3", "treated_in", {}),
        ("d5", "s2", "cause", {}),
        ("d5", "s3", "cause", {}),
        ("d5", "t3", "treat_with", {}),
        ("d5", "dept4", "treated_in", {}),
        ("d5", "bp1", "located_in", {}),
    ]

    for source, target, rel_type, props in relations:
        try:
            kg.add_relation(MedicalRelation(source, target, rel_type, props))
        except ValueError:
            continue

    return kg
