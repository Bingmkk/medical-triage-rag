"""
Faiss向量存储模块
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import pickle
import os
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import faiss
from config import VECTOR_STORE_DIR, EMBEDDING_CONFIG, FAISS_CONFIG


class MedicalEmbeddings(Embeddings):
    """医学领域嵌入模型（使用DashScope API）"""

    def __init__(self):
        import dashscope
        dashscope.api_key = EMBEDDING_CONFIG["api_key"]
        self.model_name = EMBEDDING_CONFIG["model_name"]
        self.dimension = EMBEDDING_CONFIG["dimension"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        from dashscope import TextEmbedding

        embeddings = []
        for text in texts:
            response = TextEmbedding.call(
                model=self.model_name,
                input=text
            )

            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                embeddings.append(embedding)
            else:
                embeddings.append([0.0] * self.dimension)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embed_documents([text])[0]


class LocalEmbeddings(Embeddings):
    """本地嵌入模型（使用Sentence Transformers）"""

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


class MedicalVectorStore:
    """医学向量存储管理器"""

    def __init__(self, embeddings: Optional[Embeddings] = None, store_name: str = "medical_index"):
        self.store_name = store_name
        self.embeddings = embeddings or self._create_default_embeddings()
        self.vector_store = None

    def _create_default_embeddings(self) -> Embeddings:
        """创建默认嵌入模型"""
        api_key = EMBEDDING_CONFIG.get("api_key")
        if api_key and api_key.strip():
            return MedicalEmbeddings()
        else:
            print("警告: 未设置API密钥，使用本地嵌入模型")
            return LocalEmbeddings()

    def create_vector_store(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> FAISS:
        """创建向量存储"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {"index": i}
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        return self.vector_store

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """添加文档到向量存储"""
        if self.vector_store is None:
            self.create_vector_store(texts, metadatas)
        else:
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else {"index": i}
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

            self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """相似度搜索"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建或加载向量存储")

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建或加载向量存储")

        return self.vector_store.similarity_search_with_score(query, k=k)

    def similarity_search_by_vector(self, embedding: List[float], k: int = 5) -> List[Document]:
        """通过向量进行相似度搜索"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建或加载向量存储")

        return self.vector_store.similarity_search_by_vector(embedding, k=k)

    def save(self, filepath: Optional[Path] = None) -> None:
        """保存向量存储"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，无法保存")

        if filepath is None:
            filepath = VECTOR_STORE_DIR / f"{self.store_name}"

        self.vector_store.save_local(str(filepath))
        print(f"向量存储已保存到: {filepath}")

    def load(self, filepath: Optional[Path] = None) -> None:
        """加载向量存储"""
        if filepath is None:
            filepath = VECTOR_STORE_DIR / f"{self.store_name}"

        if not Path(filepath).exists():
            raise FileNotFoundError(f"向量存储文件不存在: {filepath}")

        self.vector_store = FAISS.load_local(
            str(filepath),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"向量存储已从: {filepath} 加载")

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """转换为检索器"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")

        if search_kwargs is None:
            search_kwargs = {"k": 5}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


class HybridSearchEngine:
    """混合搜索引擎（向量搜索 + 关键词搜索）"""

    def __init__(self, vector_store: Optional[MedicalVectorStore] = None, knowledge_graph: Any = None):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """混合搜索"""
        results = []

        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                for doc in docs:
                    results.append({
                        "type": "document",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": "vector_search"
                    })
            except Exception as e:
                print(f"向量搜索出错: {e}")

        if self.knowledge_graph:
            kg_results = self._search_knowledge_graph(query)
            results.extend(kg_results)

        return self._deduplicate_and_rank(results)

    def _search_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """搜索知识图谱"""
        results = []

        query_lower = query.lower()
        entity = self.knowledge_graph.get_entity_by_name(query_lower)

        if entity:
            disease_relations = self.knowledge_graph.get_disease_relations(entity["id"])
            context_parts = []

            for symptom in disease_relations["symptoms"]:
                context_parts.append(f"症状: {symptom['name']}")

            for drug in disease_relations["drugs"]:
                context_parts.append(f"用药: {drug['name']}")

            for treatment in disease_relations["treatments"]:
                context_parts.append(f"治疗: {treatment['name']}")

            for dept in disease_relations["departments"]:
                context_parts.append(f"科室: {dept['name']}")

            if context_parts:
                results.append({
                    "type": "knowledge_graph",
                    "content": f"{entity['name']}: {', '.join(context_parts)}",
                    "metadata": entity,
                    "source": "knowledge_graph"
                })

        return results

    def _deduplicate_and_rank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和排序"""
        seen = set()
        unique_results = []

        for result in results:
            content = result["content"]
            if content not in seen:
                seen.add(content)
                unique_results.append(result)

        return unique_results


def create_sample_vector_store(embeddings: Optional[Embeddings] = None) -> Tuple[MedicalVectorStore, List[Document]]:
    """创建示例向量存储"""
    medical_texts = [
        "高血压是指以体循环动脉血压（收缩压和/或舒张压）增高为主要特征，可伴有心、脑、肾等器官的功能或器质性损害的临床综合征。常见症状包括头痛、头晕、疲劳等。治疗以降压药为主，常用药物包括钙通道阻滞剂、血管紧张素转换酶抑制剂等。",
        "糖尿病是一组以高血糖为特征的代谢性疾病。主要症状包括多饮、多尿、多食和体重下降。长期高血糖会导致各种组织，特别是眼、肾、心脏、血管、神经的慢性损害。治疗包括饮食控制、运动治疗、药物治疗和胰岛素治疗。常用药物有二甲双胍、磺脲类药物等。",
        "冠心病是冠状动脉粥样硬化性心脏病的简称，是指冠状动脉发生粥样硬化引起管腔狭窄或闭塞，导致心肌缺血缺氧或坏死而引起的心脏病。典型症状为胸痛，可放射至左臂、颈部或下颌。治疗包括药物治疗、介入治疗和外科手术。常用药物有硝酸甘油、阿司匹林等。",
        "肺炎是指终末气道、肺泡和肺间质的炎症，可由细菌、病毒、真菌、寄生虫等致病微生物，以及放射线、吸入性异物等理化因素引起。常见症状包括发热、咳嗽、咳痰、胸痛和呼吸困难。治疗以抗生素为主，根据病原菌选择合适的抗生素。",
        "急性心肌梗死是冠状动脉急性、持续性缺血缺氧所引起的心肌坏死。典型表现为持续性胸痛，常伴有烦躁不安、出汗、恐惧或濒死感。疼痛部位和性质与心绞痛相同，但程度更剧烈，持续时间更长。治疗包括紧急介入治疗、溶栓治疗和药物治疗。",
        "脑卒中俗称中风，是指脑血管突然破裂或血管阻塞导致血液不能流入大脑而引起脑组织损伤的一组疾病。包括缺血性和出血性脑卒中。症状包括突然出现的口眼歪斜、言语不利、半身不遂等。治疗包括溶栓治疗、抗血小板治疗和康复治疗。",
        "支气管哮喘简称哮喘，是一种以慢性气道炎症和气道高反应性为特征的疾病。表现为反复发作的喘息、气促、胸闷和咳嗽。治疗包括吸入性糖皮质激素、长效β2受体激动剂等药物控制气道炎症。",
        "胃炎是各种原因引起的胃黏膜炎症，分为急性和慢性两类。症状包括上腹痛、腹胀、嗳气、恶心等。治疗以抑酸药、胃黏膜保护剂为主，常用药物有奥美拉唑、铝碳酸镁等。",
        "甲状腺功能亢进症简称甲亢，是由于甲状腺合成释放过多的甲状腺激素，造成机体代谢亢进和交感神经兴奋。症状包括心悸、出汗、进食增多但体重下降、焦虑等。治疗包括抗甲状腺药物、放射性碘治疗和手术。",
        "痛风是一种由于嘌呤代谢紊乱及尿酸排泄减少所致的疾病。典型表现为关节红、肿、热、痛，常见于大脚趾关节。治疗包括饮食控制、碱化尿液和降尿酸药物，如别嘌醇、非布司他等。"
    ]

    medical_documents = [Document(page_content=text, metadata={"index": i}) for i, text in enumerate(medical_texts)]

    vector_store = MedicalVectorStore(embeddings=embeddings, store_name="sample_medical")
    vector_store.create_vector_store(medical_texts, [{"index": i} for i in range(len(medical_texts))])

    return vector_store, medical_documents
