"""
医学书籍解析和导入工具
将医学教科书内容自动结构化后导入知识库
"""
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ParsedDisease:
    """解析的疾病信息"""
    name: str
    department: str
    symptoms: List[str]
    diagnosis: List[str]
    treatment: List[str]
    nursing: List[str]
    precautions: List[str]
    icd_code: Optional[str] = None
    source: Optional[str] = None


class MedicalBookParser:
    """医学书籍解析器"""

    def __init__(self):
        self.department_keywords = self._init_department_keywords()
        self.symptom_keywords = self._init_symptom_keywords()

    def _init_department_keywords(self) -> Dict[str, List[str]]:
        """初始化科室关键词"""
        return {
            "心内科": ["心脏", "心血管", "心律", "血压", "心肌", "冠心病", "心衰", "心绞痛", "心肌梗死"],
            "神经内科": ["脑", "神经", "头痛", "癫痫", "帕金森", "中风", "脑血管", "脑梗", "脑出血"],
            "呼吸内科": ["肺", "呼吸", "咳嗽", "肺炎", "哮喘", "慢阻肺", "支气管", "气胸", "胸腔"],
            "消化内科": ["胃", "肠", "肝", "胆", "胰腺", "消化", "腹痛", "腹泻", "呕吐", "黄疸"],
            "内分泌科": ["甲状腺", "糖尿病", "肥胖", "内分泌", "激素", "甲亢", "甲减", "肾上腺"],
            "肾内科": ["肾", "肾炎", "肾病", "尿毒症", "透析", "肾衰竭", "肾结石"],
            "血液内科": ["血", "贫血", "白血病", "淋巴", "骨髓", "凝血", "血小板"],
            "风湿免疫科": ["风湿", "类风湿", "红斑狼疮", "免疫", "关节炎", "强直性脊柱炎"],
            "普外科": ["疝", "阑尾", "胆囊", "甲状腺", "乳腺", "胃", "肠", "肛肠"],
            "骨科": ["骨折", "骨", "关节", "脊柱", "腰椎", "颈椎", "骨质", "骨病"],
            "神经外科": ["脑肿瘤", "颅脑", "脊髓", "脑血管畸形", "脑外伤"],
            "泌尿外科": ["肾", "膀胱", "前列腺", "尿路", "结石", "泌尿"],
            "心外科": ["心脏手术", "搭桥", "瓣膜", "先天性心脏病"],
            "胸外科": ["肺", "食管", "纵隔", "胸壁", "肺癌", "食管癌"],
            "妇产科": ["子宫", "卵巢", "阴道", "妊娠", "分娩", "妇科", "产科", "月经", "不孕"],
            "儿科": ["儿童", "小儿", "婴儿", "幼儿", "新生儿", "先天"],
            "眼科": ["眼", "视力", "白内障", "青光眼", "角膜", "视网膜"],
            "耳鼻喉科": ["耳", "鼻", "喉", "咽喉", "鼻炎", "中耳炎", "扁桃体"],
            "口腔科": ["牙", "口腔", "牙齿", "牙周", "口腔黏膜"],
            "皮肤科": ["皮肤", "皮疹", "瘙痒", "湿疹", "痤疮", "银屑病", "疱疹"],
            "急诊科": ["急", "危重", "休克", "中毒", "创伤", "急救", "心肺复苏"],
            "ICU": ["重症", "监护", "危重", "多器官"],
            "感染科": ["感染", "发热", "病毒", "细菌", "肝炎", "艾滋病", "结核"]
        }

    def _init_symptom_keywords(self) -> List[str]:
        """初始化症状关键词"""
        return [
            "胸痛", "腹痛", "头痛", "腰痛", "关节痛", "肌肉痛",
            "发热", "寒战", "出汗",
            "咳嗽", "咳痰", "呼吸困难", "胸闷", "喘息",
            "恶心", "呕吐", "腹泻", "便秘", "腹胀", "食欲不振",
            "黄疸", "水肿", "浮肿",
            "头晕", "眩晕", "晕厥", "意识障碍",
            "心悸", "心慌", "心跳快", "心跳慢",
            "多饮", "多尿", "多食", "消瘦",
            "皮疹", "瘙痒", "出血", "瘀斑",
            "视力下降", "听力下降", "言语不清", "肢体麻木",
            "抽搐", "痉挛", "震颤",
            "焦虑", "抑郁", "失眠", "嗜睡"
        ]

    def parse_text_file(self, file_path: str) -> List[ParsedDisease]:
        """解析文本格式的医学书籍"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        diseases = []

        disease_sections = self._split_into_diseases(content)

        for section in disease_sections:
            disease = self._parse_disease_section(section)
            if disease:
                diseases.append(disease)

        return diseases

    def _split_into_diseases(self, content: str) -> List[str]:
        """将书籍内容分割成疾病章节"""
        patterns = [
            r'第[一二三四五六七八九十百千\d]+[章节条款部篇].*?\n',
            r'\n\d+\.\s+[^\n]+\n',
            r'\n【[^\]]+】\n',
            r'\n#{1,3}\s+[^\n]+\n'
        ]

        sections = []
        current_pos = 0

        for pattern in patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                    section = content[start:end].strip()
                    if len(section) > 100:
                        sections.append(section)
                break

        if not sections:
            paragraphs = content.split('\n\n')
            current_disease = ""
            for para in paragraphs:
                if self._looks_like_disease_title(para):
                    if current_disease:
                        sections.append(current_disease.strip())
                    current_disease = para
                else:
                    current_disease += "\n" + para
            if current_disease:
                sections.append(current_disease.strip())

        return sections

    def _looks_like_disease_title(self, text: str) -> bool:
        """判断文本是否像疾病标题"""
        disease_indicators = [
            '定义', '病因', '病理', '临床表现', '诊断', '鉴别诊断',
            '治疗', '护理', '预防', '预后'
        ]

        text_lower = text.lower()

        if any(indicator in text for indicator in disease_indicators):
            return True

        if len(text) < 50 and not text.endswith('。'):
            return True

        return False

    def _parse_disease_section(self, section: str) -> Optional[ParsedDisease]:
        """解析疾病章节"""
        lines = section.split('\n')
        title = self._extract_title(lines)

        if not title:
            return None

        department = self._detect_department(section)
        symptoms = self._extract_symptoms(section)
        diagnosis = self._extract_diagnosis(section)
        treatment = self._extract_treatment(section)
        nursing = self._extract_nursing(section)
        precautions = self._extract_precautions(section)

        return ParsedDisease(
            name=title,
            department=department,
            symptoms=symptoms,
            diagnosis=diagnosis,
            treatment=treatment,
            nursing=nursing,
            precautions=precautions,
            source="医学教科书"
        )

    def _extract_title(self, lines: List[str]) -> Optional[str]:
        """提取疾病名称"""
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 50:
                if not any(end in line for end in ['。', '：', ':', '；']):
                    return line
                if re.match(r'^[一二三四五六七八九十\d]+[.、]\s*\S+', line):
                    return re.sub(r'^[一二三四五六七八九十\d]+[.、]\s*', '', line)
        return None

    def _detect_department(self, text: str) -> str:
        """检测所属科室"""
        max_score = 0
        detected_dept = "内科"

        for dept, keywords in self.department_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > max_score:
                max_score = score
                detected_dept = dept

        return detected_dept

    def _extract_symptoms(self, text: str) -> List[str]:
        """提取症状信息"""
        symptoms = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if any(indicator in line for indicator in ['症状', '临床表现', '主要表现', '典型症状']):
                found_symptoms = self._extract_list_items(line, self.symptom_keywords)
                symptoms.extend(found_symptoms)

        for symptom in self.symptom_keywords:
            if symptom in text and symptom not in symptoms:
                symptoms.append(symptom)

        return list(set(symptoms))[:10]

    def _extract_diagnosis(self, text: str) -> List[str]:
        """提取诊断要点"""
        diagnosis_items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if '诊断' in line:
                continue
            if any(keyword in line for keyword in ['检查', '检验', '实验室', '影像', '心电图', 'X线', 'CT', 'MRI', '超声']):
                if len(line) < 200:
                    diagnosis_items.append(line)

        return diagnosis_items[:8]

    def _extract_treatment(self, text: str) -> List[str]:
        """提取治疗方案"""
        treatment_items = []
        lines = text.split('\n')

        in_treatment_section = False
        for line in lines:
            line = line.strip()
            if '治疗' in line and len(line) < 20:
                in_treatment_section = True
                continue
            if in_treatment_section and line:
                if any(keyword in line for keyword in ['护理', '预防', '预后', '出院']):
                    in_treatment_section = False
                if len(line) < 150:
                    treatment_items.append(line)

        return treatment_items[:8]

    def _extract_nursing(self, text: str) -> List[str]:
        """提取护理要点"""
        nursing_items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if '护理' in line:
                nursing_items.append(line)

        return nursing_items[:5]

    def _extract_precautions(self, text: str) -> List[str]:
        """提取注意事项"""
        precautions = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['注意', '禁忌', '预防', '不宜', '不可']):
                if len(line) < 100:
                    precautions.append(line)

        return precautions[:5]

    def _extract_list_items(self, text: str, keywords: List[str]) -> List[str]:
        """提取列表项"""
        items = []
        patterns = [
            r'[①②③④⑤⑥⑦⑧⑨⑩]\s*([^，,。]+)',
            r'\d+[.、]\s*([^，,。]+)',
            r'[-*]\s*([^，,。]+)',
            r'([^，,。]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if any(kw in match for kw in keywords):
                    items.append(match.strip())

        return items


class KnowledgeBaseImporter:
    """知识库导入器"""

    def __init__(self):
        self.knowledge_base = {
            "diseases": [],
            "symptoms": [],
            "drugs": [],
            "examinations": []
        }
        self.entity_id = 1000

    def import_diseases(self, diseases: List[ParsedDisease]) -> Dict[str, Any]:
        """导入疾病到知识库"""
        for disease in diseases:
            disease_entity = {
                "id": f"d_{self.entity_id}",
                "name": disease.name,
                "type": "disease",
                "properties": {
                    "department": disease.department,
                    "symptoms": disease.symptoms,
                    "diagnosis": disease.diagnosis,
                    "treatment": disease.treatment,
                    "nursing": disease.nursing,
                    "precautions": disease.precautions,
                    "icd_code": disease.icd_code or "",
                    "source": disease.source or "医学教科书"
                },
                "relations": []
            }

            for symptom in disease.symptoms:
                disease_entity["relations"].append({
                    "target": f"s_{self.entity_id}",
                    "type": "cause",
                    "description": symptom
                })

            disease_entity["relations"].append({
                "target": disease.department,
                "type": "treated_in",
                "description": f"在{disease.department}治疗"
            })

            self.knowledge_base["diseases"].append(disease_entity)
            self.entity_id += 1

        return {
            "imported_count": len(diseases),
            "total_diseases": len(self.knowledge_base["diseases"])
        }

    def save_knowledge_base(self, output_path: str):
        """保存知识库到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)

    def get_statistics(self) -> Dict[str, int]:
        """获取知识库统计信息"""
        return {
            "疾病总数": len(self.knowledge_base["diseases"]),
            "症状总数": len(self.knowledge_base["symptoms"]),
            "药物总数": len(self.knowledge_base["drugs"]),
            "检查项目总数": len(self.knowledge_base["examinations"])
        }


def import_medical_book(book_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """导入医学书籍的便捷函数"""
    parser = MedicalBookParser()
    importer = KnowledgeBaseImporter()

    diseases = parser.parse_text_file(book_path)

    result = importer.import_diseases(diseases)

    if output_path:
        importer.save_knowledge_base(output_path)
        result["output_file"] = output_path

    result["statistics"] = importer.get_statistics()

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("医学书籍解析和导入工具")
    print("=" * 70)

    print("\n使用方法:")
    print("1. 将医学教科书内容整理为文本格式（.txt）")
    print("2. 每章包含一个疾病，内容应包含：")
    print("   - 疾病名称（作为标题）")
    print("   - 临床表现/症状")
    print("   - 诊断要点")
    print("   - 治疗方案")
    print("   - 护理要点（如有）")
    print("   - 注意事项（如有）")
    print("\n3. 调用 import_medical_book() 函数导入")

    print("\n" + "=" * 70)
    print("支持的书籍格式")
    print("=" * 70)
    print("""
优先支持的格式：
1. TXT文本文件 - 最简单，只需纯文本
2. Markdown文件 - 支持结构化格式
3. JSON文件 - 结构化数据

书籍内容示例（TXT格式）：

内科学

第一章 高血压

定义
以体循环动脉血压增高为主要特征的临床综合征

临床表现
症状：头痛、头晕、心悸、胸闷
常见体征：血压升高

诊断要点
1. 多次测量血压超过140/90mmHg
2. 排除继发性高血压
3. 心电图检查
4. 血脂、血糖检查

治疗原则
1. 生活方式干预
2. 药物治疗
3. 定期监测

护理要点
1. 监测血压变化
2. 低盐低脂饮食
3. 适度运动

注意事项
1. 坚持长期服药
2. 定期复查
3. 避免情绪激动
    """)

    print("\n" + "=" * 70)
    print("导入流程")
    print("=" * 70)
    print("""
步骤1: 准备书籍文本
    将书籍内容保存为 .txt 文件

步骤2: 解析书籍
    parser = MedicalBookParser()
    diseases = parser.parse_text_file("path/to/book.txt")

步骤3: 导入知识库
    importer = KnowledgeBaseImporter()
    importer.import_diseases(diseases)

步骤4: 保存知识库
    importer.save_knowledge_base("knowledge_base.json")
    """)

    print("\n" + "=" * 70)
    print("支持的医学教科书类型")
    print("=" * 70)
    print("""
推荐导入的书籍（按优先级）：

高优先级：
1. 《内科学》- 涵盖各内科疾病
2. 《外科学》- 涵盖各外科疾病
3. 《诊断学》- 症状诊断要点
4. 《急诊医学》- 急症处理

中优先级：
5. 《药理学》- 药物信息
6. 《儿科学》- 儿童疾病
7. 《妇产科学》- 妇科产科疾病
8. 《神经病学》- 神经系统疾病

低优先级：
9. 《皮肤性病学》- 皮肤疾病
10. 《眼科学》- 眼科疾病
11. 《耳鼻喉科学》- 五官科疾病
12. 《影像诊断学》- 检查诊断
    """)

    print("\n" + "=" * 70)
