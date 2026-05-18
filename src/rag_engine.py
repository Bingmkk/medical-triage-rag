#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学RAG引擎 - 统一接口版
整合混合检索 + Rerank + 知识图谱 + 医院数据RAG
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from langchain.schema import Document
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from .knowledge_graph import MedicalKnowledgeGraph
from .llm_client import MedicalLLMClient
from .hospital_loader import HospitalDataLoader
from .hospital_rag import HospitalRAG, create_hospital_rag
from .hybrid_search_rerank import create_hybrid_search
from .prompts import TRIAGE_PROMPT_TEMPLATE, TRIAGE_PROMPT_NO_KG


class RAGEngine:
    """统一的医学RAG引擎"""

    def __init__(self):
        self.searcher = None
        self.knowledge_graph = None
        self.llm_client = None
        self.hospital_loader = None
        self.hospital_rag = None
        self._initialized = False
        self.qa_chain = None
        self.conversational_chain = None
        self.memory = None

    def initialize(self, load_knowledge_graph: bool = True) -> bool:
        """初始化RAG引擎"""
        print("=" * 70)
        print("初始化医学RAG引擎")
        print("=" * 70)

        try:
            print("\n[1/4] 加载混合检索系统...")
            self.searcher = create_hybrid_search()
            print("[V] 混合检索系统加载成功")

            if load_knowledge_graph:
                print("\n[2/4] 加载知识图谱...")
                kg_path = Path(__file__).parent.parent / 'data' / 'medical_kg.graphml'
                try:
                    if kg_path.exists():
                        self.knowledge_graph = MedicalKnowledgeGraph()
                        self.knowledge_graph.load_graphml(str(kg_path))
                        print(f"[V] 知识图谱加载成功，包含 {len(self.knowledge_graph.entity_index)} 个实体")
                    else:
                        print(f"[!] 知识图谱文件不存在: {kg_path}")
                except Exception as kg_error:
                    print(f"[!] 知识图谱加载失败，跳过: {kg_error}")
                    self.knowledge_graph = None

            print("\n[3/4] 加载医院数据RAG...")
            try:
                self.hospital_rag = HospitalRAG()
                if self.hospital_rag.load_hospital_data():
                    print("[V] 医院数据RAG加载成功")
                else:
                    print("[!] 医院数据RAG加载失败，降级使用传统方式")
                    self.hospital_loader = HospitalDataLoader()
                    hospital_info = self.hospital_loader.get_all_departments()
                    print(f"    医院: {hospital_info['hospital_info']['name']}")
            except Exception as hospital_error:
                print(f"[!] 医院数据加载失败: {hospital_error}")
                self.hospital_rag = None
                self.hospital_loader = None

            print("\n[4/4] 初始化LLM客户端...")
            self.llm_client = MedicalLLMClient()
            print("[V] LLM客户端初始化成功")

            self._initialized = True
            print("\n" + "=" * 70)
            print("✅ RAG引擎初始化完成！")
            print("=" * 70)

            return True

        except Exception as e:
            print(f"\n[X] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_hospital_context(self, symptom_or_disease: str) -> str:
        """获取医院相关上下文（通过RAG检索）"""
        if self.hospital_rag:
            context = self.hospital_rag.get_context(symptom_or_disease, top_k=3)
            if context:
                return context

        if self.hospital_loader:
            departments = self.hospital_loader.get_department_by_symptom(symptom_or_disease)
            if departments:
                context_parts = []
                hospital_info = self.hospital_loader.get_all_departments()
                context_parts.append(f"{hospital_info['hospital_info']['name']}")
                context_parts.append(f"急诊电话: {hospital_info['hospital_info']['emergency_phone']}")
                for i, dept in enumerate(departments[:3], 1):
                    context_parts.append(f"{i}. {dept['name']}")
                    if dept.get('location'):
                        context_parts.append(f"   位置: {dept['location']}")
                    if dept.get('doctors'):
                        doc_names = [d['name'] for d in dept['doctors'][:3]]
                        if doc_names:
                            context_parts.append(f"   医生: {', '.join(doc_names)}")
                return "\n".join(context_parts)

        return ""

    def query(self, question: str, use_knowledge_graph: bool = True, k: int = 5) -> Dict[str, Any]:
        """查询"""
        if not self._initialized:
            return {"success": False, "error": "RAG引擎未初始化"}

        try:
            print("\n正在检索相关知识...")
            context = self._retrieve_context(question, use_knowledge_graph, k)
            hospital_context = self._get_hospital_context(question)

            prompt = f"""根据以下医学知识信息回答问题。如果信息不足以完全回答，请说明你不知道。

医学知识信息：
{context}

{hospital_context}

问题：{question}

请给出专业、清晰、有帮助的回答。如果涉及具体科室和医生，请从本院信息中推荐。"""

            print("正在生成回答...")
            answer = self.llm_client.generate(prompt)

            return {
                "success": True,
                "question": question,
                "answer": answer,
                "context": context,
                "hospital_context": hospital_context
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def triage(self, symptom_description: str, k: int = 5) -> Dict[str, Any]:
        """医学分诊"""
        if not self._initialized:
            return {"success": False, "error": "RAG引擎未初始化"}

        try:
            print("\n正在检索相关知识...")
            context = self._retrieve_context(symptom_description, use_knowledge_graph=True, k=k)
            hospital_context = self._get_hospital_context(symptom_description)

            if context and context != "知识库中暂无相关信息，建议咨询专业医生获取准确诊断。":
                triage_prompt = TRIAGE_PROMPT_TEMPLATE.format(
                    context=context,
                    hospital_context=hospital_context,
                    symptom_description=symptom_description
                )
            else:
                triage_prompt = TRIAGE_PROMPT_NO_KG.format(
                    hospital_context=hospital_context,
                    symptom_description=symptom_description
                )

            print("正在生成分析...")
            answer = self.llm_client.generate(triage_prompt)

            urgency_level = self._extract_urgency_level(answer)

            return {
                "success": True,
                "symptom_description": symptom_description,
                "analysis": answer,
                "urgency_level": urgency_level,
                "context": context,
                "hospital_context": hospital_context
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _retrieve_context(self, query: str, use_knowledge_graph: bool = True, k: int = 5) -> str:
        """检索上下文"""
        context_parts = []

        if self.searcher:
            faiss_context = self.searcher.get_context(query, top_k=k, use_rerank=False)
            if faiss_context and faiss_context != "没有找到相关的医学知识信息。":
                context_parts.append(faiss_context)

        if use_knowledge_graph and self.knowledge_graph:
            kg_context = self._retrieve_from_knowledge_graph(query)
            if kg_context:
                context_parts.append(kg_context)

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)

    def _retrieve_from_knowledge_graph(self, query: str) -> str:
        """从知识图谱检索"""
        if not self.knowledge_graph:
            return ""

        query_lower = query.lower()
        keywords = []

        for entity_name in self.knowledge_graph.entity_index.keys():
            if entity_name in query_lower or query_lower in entity_name:
                keywords.append(entity_name)

        if not keywords:
            return ""

        context_parts = []
        for keyword in keywords[:2]:
            entity = self.knowledge_graph.get_entity_by_name(keyword)
            if entity and entity.get("type") == "disease":
                relations = self.knowledge_graph.get_disease_relations(entity["id"])

                parts = [f"关于{entity['name']}："]

                if relations.get("symptoms"):
                    symptoms = "、".join([s["name"] for s in relations["symptoms"]])
                    parts.append(f"常见症状：{symptoms}")

                if relations.get("drugs"):
                    drugs = "、".join([d["name"] for d in relations["drugs"]])
                    parts.append(f"常用药物：{drugs}")

                if relations.get("treatments"):
                    treatments = "、".join([t["name"] for t in relations["treatments"]])
                    parts.append(f"治疗方法：{treatments}")

                if relations.get("departments"):
                    depts = "、".join([d["name"] for d in relations["departments"]])
                    parts.append(f"就诊科室：{depts}")

                context_parts.append("；".join(parts))

        return "\n".join(context_parts) if context_parts else ""

    def _extract_urgency_level(self, answer: str) -> Dict[str, Any]:
        """提取紧急程度"""
        answer_lower = answer.lower()
        import re

        level_map = {
            "危急": {"level": 1, "name": "危急", "color": "red"},
            "紧急": {"level": 2, "name": "紧急", "color": "orange"},
            "次紧急": {"level": 3, "name": "次紧急", "color": "yellow"},
            "非紧急": {"level": 4, "name": "非紧急", "color": "green"},
        }

        for level_name, level_info in level_map.items():
            pattern = rf'【紧急程度】\s*{level_name}'
            if re.search(pattern, answer):
                return level_info

        for level_name, level_info in level_map.items():
            pattern = rf'紧急程度[：:]\s*{level_name}'
            if re.search(pattern, answer):
                return level_info

        危急_keywords = ["拨打120", "危及生命", "立即就医", "立即拨打", "需立即处理", "危及生命"]
        if any(keyword in answer_lower for keyword in 危急_keywords):
            if "不支持" not in answer_lower and "非危急" not in answer_lower:
                return {"level": 1, "name": "危急", "color": "red"}

        紧急_keywords = ["尽快就医", "严重", "需立即"]
        if any(keyword in answer_lower for keyword in 紧急_keywords):
            if "非紧急" not in answer_lower and "不需要" not in answer_lower:
                return {"level": 2, "name": "紧急", "color": "orange"}

        if "非紧急" in answer_lower or "不紧急" in answer_lower:
            return {"level": 4, "name": "非紧急", "color": "green"}

        if "次紧急" in answer_lower:
            return {"level": 3, "name": "次紧急", "color": "yellow"}

        return {"level": 4, "name": "非紧急", "color": "green"}

    def search_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """搜索医学知识"""
        if not self.searcher:
            return []
        return self.searcher.search(query, top_k=k, use_rerank=True)

    def get_recommended_departments(self, symptom: str) -> List[Dict[str, Any]]:
        """获取推荐科室（通过RAG检索）"""
        if self.hospital_rag:
            results = self.hospital_rag.search(symptom, top_k=5)
            return [r['department'] for r in results]
        elif self.hospital_loader:
            return self.hospital_loader.get_department_by_symptom(symptom)
        return []

    def create_qa_chain(self, prompt_template: Optional[str] = None) -> RetrievalQA:
        """创建问答链"""
        if self.searcher is None:
            raise ValueError("检索系统未初始化")

        if prompt_template is None:
            prompt_template = """根据以下医学知识信息回答问题。如果信息不足以回答，请说明你不知道。

医学知识信息：
{context}

问题：{question}

回答："""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        retriever = self.searcher.as_retriever(search_kwargs={"k": 5})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_client._get_client(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return self.qa_chain

    def create_conversational_chain(self, prompt_template: Optional[str] = None) -> ConversationalRetrievalChain:
        """创建对话链"""
        if self.searcher is None:
            raise ValueError("检索系统未初始化")

        if prompt_template is None:
            prompt_template = """你是一个专业的医学分诊助手。根据以下医学知识信息，结合对话历史，回答用户的问题。

医学知识信息：
{context}

对话历史：
{chat_history}

当前问题：{question}

回答："""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )

        retriever = self.searcher.as_retriever(search_kwargs={"k": 5})

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_client._get_client(),
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return self.conversational_chain

    def conversational_query(self, question: str) -> Dict[str, Any]:
        """对话查询"""
        if self.conversational_chain is None:
            self.create_conversational_chain()

        result = self.conversational_chain({"question": question})

        return {
            "question": question,
            "answer": result["answer"],
            "chat_history": self.memory.chat_memory.messages
        }

    def batch_query(self, questions: List[str], use_knowledge_graph: bool = True) -> List[Dict[str, Any]]:
        """批量查询"""
        results = []
        for question in questions:
            result = self.query(question, use_knowledge_graph)
            results.append(result)
        return results

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """获取相关文档"""
        if self.searcher is None:
            return []
        return self.searcher.similarity_search(query, k=k)

    def get_context_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """获取带分数的上下文"""
        if self.searcher is None:
            return []
        return self.searcher.similarity_search_with_score(query, k=k)


def create_rag_engine() -> RAGEngine:
    """创建并初始化RAG引擎"""
    engine = RAGEngine()
    success = engine.initialize()

    if success:
        return engine
    else:
        raise RuntimeError("RAG引擎初始化失败")


def create_optimized_rag_engine() -> RAGEngine:
    """创建优化版RAG引擎（兼容旧接口）"""
    return create_rag_engine()


if __name__ == "__main__":
    print("测试统一版RAG引擎...")

    engine = RAGEngine()

    if engine.initialize():
        print("\n测试查询...")
        result = engine.query("高血压有哪些症状？")

        if result.get("success"):
            print(f"\n问题: {result['question']}")
            print(f"\n回答:\n{result['answer']}")
        else:
            print(f"查询失败: {result.get('error')}")

        print("\n测试分诊...")
        result = engine.triage("我头疼，头晕，伴有恶心")

        if result.get("success"):
            print(f"\n症状: {result['symptom_description']}")
            print(f"\n分析:\n{result['analysis']}")
            print(f"\n紧急程度: {result['urgency_level']['name']}")
        else:
            print(f"分诊失败: {result.get('error')}")

        print("\n测试医院RAG检索...")
        depts = engine.get_recommended_departments("头痛")
        print(f"推荐科室: {[d['name'] for d in depts]}")
    else:
        print("RAG引擎初始化失败")