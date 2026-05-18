#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faiss索引加载和搜索模块
用于加载和使用已创建的Faiss索引
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss


class FaissMedicalIndexer:
    """Faiss医学索引管理器"""

    def __init__(self):
        self.index = None
        self.metadata = []
        self.model = None
        self.embedding_dim = 384
        self._loaded = False

    def load_medical_books_index(self, index_path: str = 'data/medical_books.index',
                                 metadata_path: str = 'data/medical_books_metadata.json') -> bool:
        """加载医学书籍索引"""
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

            self._load_embedding_model()
            self._loaded = True

            return True

        except Exception as e:
            print(f"[X] 加载索引失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_embedding_model(self):
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            print("加载嵌入模型...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"[V] 嵌入模型加载成功，维度: {self.embedding_dim}")
        except Exception as e:
            print(f"[X] 加载嵌入模型失败: {e}")
            raise

    def search(self, query: str, k: int = 5, min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """语义搜索

        Args:
            query: 查询文本
            k: 返回数量
            min_similarity: 最小相似度阈值（0-1）

        Returns:
            搜索结果列表
        """
        if not self._loaded:
            print("[!] 索引未加载")
            return []

        try:
            query_vector = self.model.encode([query])
            query_vector = np.array(query_vector, dtype='float32')

            norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
            query_vector = query_vector / norms

            distances, indices = self.index.search(query_vector, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    if dist >= min_similarity:
                        meta = self.metadata[idx]
                        result = {
                            'score': float(dist),
                            'similarity': f"{dist * 100:.1f}%",
                            'rank': len(results) + 1,
                            **meta
                        }
                        results.append(result)

            return results

        except Exception as e:
            print(f"[X] 搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_context(self, query: str, k: int = 5, max_chars: int = 2000) -> str:
        """获取检索上下文（用于RAG）

        Args:
            query: 查询文本
            k: 检索数量
            max_chars: 最大字符数

        Returns:
            格式化后的上下文字符串
        """
        results = self.search(query, k=k)

        if not results:
            return "没有找到相关的医学知识信息。"

        context_parts = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            book = result.get('book', '未知书籍')
            h1 = result.get('h1', '')
            h2 = result.get('h2', '')
            similarity = result.get('similarity', '0%')

            section = f"{h1}"
            if h2:
                section += f" > {h2}"

            context_parts.append(
                f"[{i}] {book} - {section} (相似度: {similarity})"
            )

        return "\n\n".join(context_parts)

    def get_detailed_results(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """获取详细搜索结果"""
        results = self.search(query, k=k)

        detailed = []
        for result in results:
            detailed.append({
                'rank': result['rank'],
                'book': result.get('book', ''),
                'section': f"{result.get('h1', '')} > {result.get('h2', '')}".strip(' > '),
                'similarity': result['similarity'],
                'score': result['score'],
                'char_length': result.get('char_len', 0)
            })

        return detailed


def load_medical_faiss_index() -> Optional[FaissMedicalIndexer]:
    """加载医学Faiss索引的便捷函数"""
    indexer = FaissMedicalIndexer()

    base_path = Path(__file__).parent.parent
    index_path = base_path / 'data' / 'medical_books.index'
    metadata_path = base_path / 'data' / 'medical_books_metadata.json'

    if indexer.load_medical_books_index(str(index_path), str(metadata_path)):
        return indexer
    else:
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("Faiss 索引加载测试")
    print("=" * 70)

    indexer = FaissMedicalIndexer()

    if indexer.load_medical_books_index():
        print("\n测试搜索...")
        test_queries = ["胸痛", "高血压", "糖尿病"]

        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 70)

            results = indexer.search(query, k=3)

            for result in results:
                print(f"\n[{result['rank']}] {result['book']}")
                print(f"    章节: {result['h1']} > {result['h2']}")
                print(f"    相似度: {result['similarity']}")

        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)
    else:
        print("[X] 索引加载失败")
