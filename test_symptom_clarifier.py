#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
症状追问系统测试
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.symptom_clarifier import create_clarifier


def test_symptom_clarifier():
    """测试症状追问系统"""

    print("=" * 70)
    print("症状梳理对话")
    print("=" * 70)

    clarifier = create_clarifier(max_rounds=5)

    print("\n👋 您好！我来帮您理清一下症状。")
    print("请描述您哪里不舒服：")

    initial = input("\n📝 ")

    print(f"\n好的，我听到了。让我帮您理清一下...\n")

    result = clarifier.start_clarification(initial)

    print(f"\n💬 {result['question']}")
    print(f"   [{result['progress']}]")

    for i in range(2, 7):
        answer = input(f"\n📝 您的回答：")
        result = clarifier.continue_clarification(answer)

        if not result["continue"]:
            print(f"\n✅ 追问结束，共问了 {result['total_rounds']} 个问题")
            print(f"已覆盖：{', '.join(result['asked_topics'])}")
            break

        print(f"\n💬 {result['question']}")
        print(f"   [{result['progress']}]")

    print("\n" + "=" * 70)
    print("【症状总结】")
    print("=" * 70)
    print(result["summary"])


if __name__ == "__main__":
    test_symptom_clarifier()
