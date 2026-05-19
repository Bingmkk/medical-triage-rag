#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索 + Rerank模块
Faiss向量检索 + BM25关键词匹配 + Rerank精排
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import numpy as np
import faiss


class HybridSearchRerank:
    """混合检索 + Rerank系统"""

    def __init__(self, embedding_dim=384):
        """
        初始化混合检索系统

        Args:
            embedding_dim: 向量维度
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
        self.model = None
        self._initialized = False
        self.bm25 = None
        self.tokenized_corpus = []

    def initialize(self):
        """初始化嵌入模型"""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            print("加载嵌入模型...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self._initialized = True
            print(f"[V] 嵌入模型加载成功，维度: {self.embedding_dim}")
        except Exception as e:
            print(f"[X] 嵌入模型加载失败: {e}")
            raise

    def load_index(self, index_path: str = 'data/medical_books.index',
                   metadata_path: str = 'data/medical_books_metadata.json') -> bool:
        """加载Faiss索引"""
        try:
            print("加载 Faiss 索引...")

            index_file = Path(index_path)
            metadata_file = Path(metadata_path)

            if not index_file.exists():
                print(f"[X] 索引文件不存在: {index_path}")
                return False

            if not metadata_file.exists():
                print(f"[X] 元数据文件不存在: {metadata_path}")
                return False

            self.index = faiss.read_index(str(index_file))
            print(f"[V] 索引加载成功，包含 {self.index.ntotal} 个向量")

            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"[V] 元数据加载成功，包含 {len(self.metadata)} 条")

            self.initialize()

            print("构建BM25索引...")
            self._build_bm25_index()

            return True

        except Exception as e:
            print(f"[X] 加载索引失败: {e}")
            return False

    def _build_bm25_index(self):
        """构建BM25索引"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[!] rank_bm25未安装，使用简单关键词匹配作为备选")
            self._build_simple_keywords_index()
            return

        self.tokenized_corpus = []
        for meta in self.metadata:
            text = f"{meta.get('book', '')} {meta.get('h1', '')} {meta.get('h2', '')} {meta.get('h3', '')} {meta.get('h4', '')}"
            tokens = self._tokenize_chinese(text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[V] BM25索引构建完成，包含 {len(self.tokenized_corpus)} 条文档")

    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词（简单二元组分词）"""
        chinese_words = re.findall(r'[\u4e00-\u9fa5]+', text.lower())
        tokens = []
        for word in chinese_words:
            if len(word) >= 2:
                tokens.append(word)
                for i in range(len(word) - 1):
                    tokens.append(word[i:i+2])
        return tokens

    def _build_simple_keywords_index(self):
        """备选：简单关键词倒排索引"""
        self.keywords_index = {}
        for idx, meta in enumerate(self.metadata):
            book = meta.get('book', '')
            h1 = meta.get('h1', '')
            h2 = meta.get('h2', '')
            h3 = meta.get('h3', '')
            h4 = meta.get('h4', '')

            keywords = set()

            for text in [book, h1, h2, h3, h4]:
                chinese_words = re.findall(r'[\u4e00-\u9fa5]+', text.lower())
                for word in chinese_words:
                    if len(word) >= 2:
                        keywords.add(word)
                    for i in range(len(word) - 1):
                        bigram = word[i:i+2]
                        keywords.add(bigram)

            for keyword in keywords:
                if keyword not in self.keywords_index:
                    self.keywords_index[keyword] = []
                self.keywords_index[keyword].append(idx)

        print(f"[V] 简单关键词索引构建完成，包含 {len(self.keywords_index)} 个关键词")

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25关键词搜索"""
        if self.bm25 is not None:
            tokens = self._tokenize_chinese(query.lower())
            scores = self.bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        else:
            return self._keyword_search_simple(query, top_k)

    def _keyword_search_simple(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """备选：简单关键词搜索"""
        query_keywords = set()
        chinese_words = re.findall(r'[\u4e00-\u9fa5]+', query.lower())

        for word in chinese_words:
            if len(word) >= 2:
                query_keywords.add(word)
            for i in range(len(word) - 1):
                bigram = word[i:i+2]
                query_keywords.add(bigram)

        doc_scores = {}
        for keyword in query_keywords:
            if keyword in self.keywords_index:
                for doc_idx in self.keywords_index[keyword]:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_docs[:top_k]]

    def _semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """语义搜索（Faiss）"""
        if not self._initialized:
            return []

        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector, dtype='float32')

        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / norms

        distances, indices = self.index.search(query_vector, top_k)

        return [(idx, float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]

    def hybrid_search(self, query: str, top_k: int = 20,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> List[Tuple[int, float, str]]:
        """
        混合搜索

        Args:
            query: 查询文本
            top_k: 返回数量
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重

        Returns:
            [(doc_idx, score, source), ...]
        """
        semantic_results = self._semantic_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        combined_scores = {}

        for idx, score in semantic_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * semantic_weight

        for idx, score in bm25_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * keyword_weight

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return [(idx, score, "hybrid") for idx, score in sorted_results[:top_k]]

    def rerank(self, query: str, candidates: List[Tuple[int, float, str]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank精排

        Args:
            query: 查询文本
            candidates: 候选文档 [(doc_idx, score, source), ...]
            top_k: 返回数量

        Returns:
            精排后的结果
        """
        if not candidates:
            return []

        rerank_prompt = f"""你是一位医学专家，请根据问题对相关文档进行相关性排序。

问题：{query}

请阅读以下文档，判断它们与问题的相关性：

"""

        for i, (doc_idx, score, source) in enumerate(candidates[:10], 1):
            meta = self.metadata[doc_idx]
            doc_text = f"""
文档{i}（相关性得分：{score:.2f}）：
- 书籍：{meta.get('book', '')}
- 章节：{meta.get('h1', '')} > {meta.get('h2', '')}
- 内容长度：{meta.get('char_len', 0)}字
"""
            rerank_prompt += doc_text

        rerank_prompt += """
请按相关性从高到低排序，输出文档编号（如：1, 3, 2, 5, 4）。
只输出编号，不要其他内容。"""

        from .llm_client import MedicalLLMClient
        llm_client = MedicalLLMClient()
        ranking_str = llm_client.generate(rerank_prompt)

        try:
            ranking = [int(x.strip()) - 1 for x in ranking_str.split(',') if x.strip().isdigit()]
        except:
            ranking = list(range(min(10, len(candidates))))

        results = []
        for rank, doc_idx in enumerate(ranking[:top_k], 1):
            if doc_idx < len(candidates):
                idx, score, source = candidates[doc_idx]
                meta = self.metadata[idx]
                results.append({
                    'rank': rank,
                    'doc_idx': idx,
                    'score': score,
                    'source': source,
                    **meta
                })

        return results

    def search(self, query: str, top_k: int = 5,
               use_rerank: bool = True) -> List[Dict[str, Any]]:
        """
        完整搜索流程：混合检索 + Rerank

        Args:
            query: 查询文本
            top_k: 返回数量
            use_rerank: 是否使用Rerank

        Returns:
            搜索结果
        """
        print(f"搜索: {query}")

        print("  [1/3] 混合检索...")
        candidates = self.hybrid_search(query, top_k=top_k * 3,
                                      semantic_weight=0.7,
                                      keyword_weight=0.3)

        if not candidates:
            return []

        print(f"  [2/3] 获取候选文档 {len(candidates)} 条...")

        if use_rerank:
            print("  [3/3] Rerank精排...")
            results = self.rerank(query, candidates, top_k=top_k)
        else:
            results = []
            for i, (idx, score, source) in enumerate(candidates[:top_k], 1):
                meta = self.metadata[idx]
                results.append({
                    'rank': i,
                    'doc_idx': idx,
                    'score': score,
                    'source': source,
                    **meta
                })

        return results

    def get_context(self, query: str, top_k: int = 5, use_rerank: bool = True) -> str:
        """获取检索上下文"""
        results = self.search(query, top_k=top_k, use_rerank=use_rerank)

        if not results:
            return "没有找到相关的医学知识信息。"

        context_parts = []
        for i, result in enumerate(results, 1):
            book = result.get('book', '未知书籍')
            h1 = result.get('h1', '')
            h2 = result.get('h2', '')
            section = f"{h1}"
            if h2:
                section += f" > {h2}"

            context_parts.append(
                f"[{i}] {book} - {section} (相似度: {result['score']:.2f})"
            )

        return "\n\n".join(context_parts)


def create_hybrid_search() -> HybridSearchRerank:
    """创建混合检索系统"""
    searcher = HybridSearchRerank()

    base_path = Path(__file__).parent.parent
    index_path = base_path / 'data' / 'medical_books.index'
    metadata_path = base_path / 'data' / 'medical_books_metadata.json'

    if searcher.load_index(str(index_path), str(metadata_path)):
        return searcher
    else:
        raise RuntimeError("索引加载失败")


if __name__ == "__main__":
    print("=" * 70)
    print("混合检索 + Rerank 测试")
    print("=" * 70)

    try:
        searcher = create_hybrid_search()

        test_queries = ["高血压", "胸痛", "糖尿病"]

        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 70)

            results = searcher.search(query, top_k=5, use_rerank=True)

            for result in results:
                print(f"\n[{result['rank']}] {result['book']}")
                print(f"    章节: {result['h1']} > {result['h2']}")
                print(f"    得分: {result['score']:.3f}")

        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)

    except Exception as e:
        print(f"[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()
