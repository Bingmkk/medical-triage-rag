#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的医学知识Faiss索引器
处理 medical-books-embedding 项目中的 chunks.json
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

class MedicalBooksFaissIndexer:
    """完整的医学书籍Faiss索引器"""

    def __init__(self, embedding_dim=384):
        """初始化"""
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
        self.embeddings_model = None
        self._initialized = False

    def initialize(self):
        """初始化嵌入模型"""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            print("正在加载文本嵌入模型...")
            self.embeddings_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.embedding_dim = self.embeddings_model.get_sentence_embedding_dimension()
            self._initialized = True
            print(f"模型加载完成，向量维度: {self.embedding_dim}")

        except ImportError:
            print("[X] 请先安装依赖: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"[X] 模型加载失败: {e}")
            raise

    def load_chunks_data(self, data_path: str) -> Tuple[List[str], List[Dict]]:
        """加载完整的 chunks 数据"""

        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        texts = []
        metadata = []

        for chunk in chunks:
            content = chunk.get('content', '')
            metadata_info = chunk.get('metadata', {})

            if content:
                texts.append(content)
                metadata.append({
                    'book': metadata_info.get('book', '未知书籍'),
                    'h1': metadata_info.get('h1', ''),
                    'h2': metadata_info.get('h2', ''),
                    'h3': metadata_info.get('h3', ''),
                    'h4': metadata_info.get('h4', ''),
                    'char_len': chunk.get('char_len', 0)
                })

        books_count = len(set(m['book'] for m in metadata))
        print(f"[V] 加载了 {len(texts)} 条医学知识")
        print(f"    - 书籍数量: {books_count}")
        print(f"    - 平均长度: {sum(m['char_len'] for m in metadata) / len(metadata):.0f} 字")

        return texts, metadata

    def create_index(self, texts: List[str], metadata: List[Dict]):
        """创建Faiss索引"""

        self.initialize()

        print("\n正在生成文本向量...")
        print(f"总文本数: {len(texts)}")

        start_time = time.time()

        embeddings = self.embeddings_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        elapsed = time.time() - start_time
        print(f"向量生成完成，耗时: {elapsed:.1f}秒")
        print(f"向量形状: {embeddings.shape}")

        embeddings = np.array(embeddings, dtype='float32')

        from sentence_transformers import util
        util.normalize_embeddings(embeddings)

        import faiss
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        self.metadata = metadata

        print(f"[V] 索引创建完成，包含 {self.index.ntotal} 个向量")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似医学知识"""

        if self._initialized and self.index is not None:
            query_vector = self.embeddings_model.encode([query])
            query_vector = np.array(query_vector, dtype='float32')

            from sentence_transformers import util
            util.normalize_embeddings(query_vector)

            distances, indices = self.index.search(query_vector, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    meta = self.metadata[idx]
                    result = {
                        'score': float(dist),
                        'similarity': f"{dist * 100:.1f}%",
                        'rank': len(results) + 1,
                        **meta
                    }
                    results.append(result)

            return results
        else:
            print("[!] Faiss索引未初始化")
            return []

    def save_index(self, index_path: str = 'data/medical_books.index'):
        """保存索引和元数据"""

        if self.index is not None:
            import faiss
            faiss.write_index(self.index, index_path)
            print(f"[V] 索引已保存到: {index_path}")

        metadata_path = Path(index_path).parent / f"{Path(index_path).stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"[V] 元数据已保存到: {metadata_path}")

    def load_index(self, index_path: str = 'data/medical_books.index'):
        """加载索引和元数据"""

        self.initialize()

        index_path = Path(index_path)

        if not index_path.exists():
            print(f"[X] 索引文件不存在: {index_path}")
            return False

        import faiss
        self.index = faiss.read_index(str(index_path))
        print(f"[V] 索引已加载，包含 {self.index.ntotal} 个向量")

        metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"[V] 元数据已加载，包含 {len(self.metadata)} 条")

        return True


def create_full_medical_index():
    """创建完整的医学书籍索引（主函数）"""

    print("=" * 70)
    print("完整的医学书籍Faiss索引创建")
    print("=" * 70)

    indexer = MedicalBooksFaissIndexer()

    base_path = Path(__file__).parent
    chunks_path = base_path / 'data' / 'chunks.json'

    try:
        texts, metadata = indexer.load_chunks_data(chunks_path)

        indexer.create_index(texts, metadata)

        indexer.save_index()

        print("\n" + "=" * 70)
        print("测试搜索")
        print("=" * 70)

        test_queries = [
            '胸痛',
            '发热',
            '高血压',
            '糖尿病',
            '肺炎'
        ]

        for query in test_queries:
            print(f"\n搜索: {query}")
            print("-" * 70)

            results = indexer.search(query, k=3)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n[{i}] {result['book']}")
                    print(f"    章节: {result['h1']} > {result['h2']}")
                    print(f"    相似度: {result['similarity']}")
            else:
                print("    未找到相关结果")

        print("\n" + "=" * 70)
        print("索引创建成功！")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n[X] 错误: {e}")
        print("\n请先下载完整的 chunks.json 文件")
    except Exception as e:
        print(f"\n[X] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_full_medical_index()
