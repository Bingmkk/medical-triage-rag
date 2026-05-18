#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合RAG系统测试
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_engine import create_rag_engine


def test_rag_engine():
    """测试整合的RAG引擎"""

    print("=" * 70)
    print("整合RAG系统测试")
    print("=" * 70)

    try:
        engine = create_rag_engine()

        print("\n" + "=" * 70)
        print("测试1: 医学知识查询")
        print("=" * 70)

        test_queries = [
            "高血压有哪些症状？",
            "糖尿病的诊断标准是什么？",
            "胸痛应该挂什么科？"
        ]

        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 70)

            result = engine.query(query, use_knowledge_graph=True, k=3)

            if result.get("success"):
                print(f"\n回答:\n{result['answer']}")

                print(f"\n检索到的相关知识 ({len(result['retrieved_docs'])} 条):")
                for doc in result['retrieved_docs'][:3]:
                    print(f"  - {doc['book']}: {doc['section']} ({doc['similarity']})")
            else:
                print(f"[X] 查询失败: {result.get('error')}")

        print("\n" + "=" * 70)
        print("测试2: 症状分诊")
        print("=" * 70)

        triage_queries = [
            "我最近经常胸痛，伴随出冷汗和呼吸困难",
            "我胃疼了好几天了，吃什么都没胃口"
        ]

        for symptom in triage_queries:
            print(f"\n症状: {symptom}")
            print("-" * 70)

            result = engine.triage(symptom, k=3)

            if result.get("success"):
                print(f"\n紧急程度: {result['urgency_level']['name']}")
                print(f"\n分析:\n{result['analysis']}")
            else:
                print(f"[X] 分诊失败: {result.get('error')}")

        print("\n" + "=" * 70)
        print("所有测试完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rag_engine()
