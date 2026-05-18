#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faiss索引测试脚本
快速验证医学书籍索引是否正常工作
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path

def test_medical_faiss_index():
    """测试医学书籍Faiss索引"""

    print("=" * 70)
    print("Faiss 索引测试")
    print("=" * 70)

    # 加载索引
    print("\n加载 Faiss 索引...")
    import faiss

    index_path = Path(__file__).parent / 'data' / 'medical_books.index'
    metadata_path = Path(__file__).parent / 'data' / 'medical_books_metadata.json'

    if not index_path.exists():
        print(f"[X] 索引文件不存在: {index_path}")
        return

    if not metadata_path.exists():
        print(f"[X] 元数据文件不存在: {metadata_path}")
        return

    # 加载索引
    index = faiss.read_index(str(index_path))
    print(f"[V] 索引加载成功，包含 {index.ntotal} 个向量")

    # 加载元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"[V] 元数据加载成功，包含 {len(metadata)} 条")

    # 测试查询
    print("\n" + "=" * 70)
    print("语义搜索测试")
    print("=" * 70)

    test_queries = [
        "胸痛",
        "高血压",
        "糖尿病",
        "肺炎",
        "心电图"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 70)

        results = semantic_search(index, metadata, query, k=3)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['book']}")
            print(f"    章节: {result['h1']} > {result['h2']}")
            print(f"    相似度: {result['similarity']}")
            print(f"    内容预览: {result['content'][:100]}...")

def semantic_search(index, metadata, query, k=5):
    """语义搜索"""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    query_vector = model.encode([query])
    query_vector = np.array(query_vector, dtype='float32')

    norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_vector = query_vector / norms

    distances, indices = index.search(query_vector, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0 and idx < len(metadata):
            meta = metadata[idx]
            result = {
                'score': float(dist),
                'similarity': f"{dist * 100:.1f}%",
                **meta
            }
            results.append(result)

    return results

if __name__ == "__main__":
    test_medical_faiss_index()
