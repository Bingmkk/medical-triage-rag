#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医院信息RAG检索模块
将科室、医生信息向量化，通过RAG方式检索
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss


class HospitalRAG:
    """医院信息RAG检索系统"""

    def __init__(self):
        self.index = None
        self.department_embeddings = []
        self.department_info = []
        self.model = None
        self._initialized = False

    def initialize(self):
        """初始化嵌入模型"""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            print("加载医院信息嵌入模型...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self._initialized = True
            print("[V] 医院信息嵌入模型加载成功")
        except Exception as e:
            print(f"[X] 医院信息嵌入模型加载失败: {e}")
            raise

    def load_hospital_data(self, data_path: str = 'data/hospital_departments.json') -> bool:
        """加载医院数据并构建向量索引"""
        try:
            data_file = Path(data_path)
            if not data_file.exists():
                print(f"[X] 医院数据文件不存在: {data_path}")
                return False

            with open(data_file, 'r', encoding='utf-8') as f:
                hospital_data = json.load(f)

            print("处理医院科室数据...")
            self.department_info = []
            symptom_by_dept = self._build_symptom_index(
                hospital_data.get('specialty_mapping', {})
            )

            for dept_category in hospital_data.get('departments', []):
                for main_dept in dept_category.get('sub_departments', []):
                    if 'sub_specialties' in main_dept:
                        for sub_dept in main_dept['sub_specialties']:
                            dept_info = self._process_department(sub_dept)
                            if dept_info:
                                dept_info['symptoms'] = symptom_by_dept.get(
                                    sub_dept.get('id', ''), []
                                )
                                self.department_info.append(dept_info)
                    else:
                        dept_info = self._process_department(main_dept)
                        if dept_info:
                            dept_info['symptoms'] = symptom_by_dept.get(
                                main_dept.get('id', ''), []
                            )
                            self.department_info.append(dept_info)

            print(f"[V] 共处理 {len(self.department_info)} 个科室")

            self.initialize()
            self._build_index()

            return True

        except Exception as e:
            print(f"[X] 加载医院数据失败: {e}")
            return False

    def _build_symptom_index(self, specialty_mapping: Dict) -> Dict[str, List[str]]:
        """根据 specialty_mapping 反查每个科室对应的症状关键词"""
        index: Dict[str, List[str]] = {}
        for symptom, dept_ids in specialty_mapping.items():
            for dept_id in dept_ids:
                index.setdefault(dept_id, [])
                if symptom not in index[dept_id]:
                    index[dept_id].append(symptom)
        return index

    def _process_department(self, dept: Dict) -> Dict:
        """处理单个科室信息"""
        info = {
            'id': dept.get('id', ''),
            'name': dept.get('name', ''),
            'description': dept.get('description', ''),
            'location': dept.get('location', ''),
            'room_number': dept.get('room_number', ''),
            'beds': dept.get('beds', 0),
            'doctors': dept.get('doctors', []),
            'schedule_info': dept.get('schedule_info', {}),
            'symptoms': []
        }
        return info

    def _build_index(self):
        """构建向量索引"""
        if not self._initialized:
            self.initialize()

        print("生成科室嵌入向量...")
        self.department_embeddings = []

        for dept in self.department_info:
            text = self._create_dept_text(dept)
            embedding = self.model.encode(text)
            self.department_embeddings.append(embedding)

        embeddings_array = np.array(self.department_embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)

        print(f"[V] 医院信息索引构建完成，包含 {self.index.ntotal} 个向量")

    def _create_dept_text(self, dept: Dict) -> str:
        """创建科室文本描述用于向量化"""
        parts = []
        parts.append(dept['name'])
        if dept['description']:
            parts.append(dept['description'])
        if dept.get('symptoms'):
            parts.append(f"擅长症状: {', '.join(dept['symptoms'][:12])}")
        for doctor in dept.get('doctors', [])[:4]:
            specialty = doctor.get('specialty', '')
            if specialty:
                parts.append(f"诊治: {specialty}")
        if dept['location']:
            parts.append(f"位置: {dept['location']}")
        if dept['room_number']:
            parts.append(f"房间号: {dept['room_number']}")
        if dept['doctors']:
            doctor_names = [d['name'] for d in dept['doctors'][:3]]
            parts.append(f"医生: {', '.join(doctor_names)}")
        return ' '.join(parts)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关科室"""
        if not self._initialized or self.index is None:
            return []

        query_embedding = self.model.encode(query)
        query_array = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_array, top_k)

        results = []
        for i in range(top_k):
            if indices[0][i] >= 0 and indices[0][i] < len(self.department_info):
                dept = self.department_info[indices[0][i]]
                results.append({
                    'department': dept,
                    'score': float(distances[0][i]),
                    'match_type': 'vector'
                })

        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        """获取医院信息上下文"""
        results = self.search(query, top_k)
        
        if not results:
            return ""

        context_parts = []
        for result in results:
            dept = result['department']
            parts = []
            parts.append(f"科室: {dept['name']}")
            if dept['description']:
                parts.append(f"描述: {dept['description']}")
            if dept['location']:
                parts.append(f"位置: {dept['location']}")
            if dept['room_number']:
                parts.append(f"房间: {dept['room_number']}")
            if dept['doctors']:
                doc_info = []
                for doc in dept['doctors'][:3]:
                    doc_info.append(f"{doc['name']} ({doc['title']}，{doc.get('specialty', '')})")
                parts.append(f"推荐医生: {'; '.join(doc_info)}")
            if dept['schedule_info']:
                schedule = dept['schedule_info']
                schedule_parts = []
                if schedule.get('weekday'):
                    schedule_parts.append(f"工作日: {schedule['weekday']}")
                if schedule.get('weekend'):
                    schedule_parts.append(f"周末: {schedule['weekend']}")
                if schedule.get('emergency'):
                    schedule_parts.append(f"急诊: {schedule['emergency']}")
                if schedule_parts:
                    parts.append(f"时间: {' '.join(schedule_parts)}")
            
            context_parts.append('\n'.join(parts))

        return '\n\n'.join(context_parts)


def create_hospital_rag() -> HospitalRAG:
    """创建医院RAG检索系统"""
    rag = HospitalRAG()
    rag.load_hospital_data()
    return rag


if __name__ == "__main__":
    print("测试医院信息RAG检索...")
    
    hospital_rag = HospitalRAG()
    if hospital_rag.load_hospital_data():
        print("\n测试搜索: 头痛")
        results = hospital_rag.search("头痛", top_k=3)
        for i, result in enumerate(results, 1):
            dept = result['department']
            print(f"\n{i}. {dept['name']} (相似度: {result['score']:.3f})")
            print(f"   位置: {dept.get('location', '未知')}")
            if dept['doctors']:
                print(f"   医生: {dept['doctors'][0]['name']}")
        
        print("\n获取上下文:")
        context = hospital_rag.get_context("头痛")
        print(context)
    else:
        print("医院RAG初始化失败")