#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速搜索测试
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index('data/medical_books.index')
with open('data/medical_books_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

query = input("请输入搜索关键词: ")

query_vector = model.encode([query])
query_vector = np.array(query_vector, dtype='float32')
norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
query_vector = query_vector / norms

distances, indices = index.search(query_vector, 3)

print(f"\n查询: {query}")
print("=" * 70)

for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    if idx >= 0 and idx < len(metadata):
        meta = metadata[idx]
        print(f"\n[{i}] {meta['book']}")
        print(f"    章节: {meta['h1']} > {meta['h2']}")
        print(f"    相似度: {dist * 100:.1f}%")
        print(f"    字符长度: {meta.get('char_len', 0)} 字")
