#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学分诊系统 - 完整版
整合症状追问、Faiss搜索、分诊建议、科室医生推荐
"""
import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.symptom_clarifier import create_clarifier
from src.rag_engine import create_optimized_rag_engine


class MedicalTriageSystem:
    """完整的医学分诊系统"""

    def __init__(self):
        self.clarifier = None
        self.rag_engine = None
        self._initialized = False

    def initialize(self) -> bool:
        """初始化系统"""
        print("=" * 70)
        print("初始化医学分诊系统")
        print("=" * 70)

        try:
            print("\n[1/2] 初始化症状追问系统...")
            self.clarifier = create_clarifier(max_rounds=5)
            print("[V] 症状追问系统就绪")

            print("\n[2/2] 初始化RAG引擎（混合检索 + Rerank）...")
            self.rag_engine = create_optimized_rag_engine()
            print("[V] RAG引擎就绪（已优化）")

            self._initialized = True
            print("\n" + "=" * 70)
            print("✅ 医学分诊系统初始化完成！")
            print("=" * 70)

            return True

        except Exception as e:
            print(f"\n[X] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_triage(self):
        """运行分诊对话"""
        if not self._initialized:
            print("[X] 系统未初始化")
            return

        print("\n" + "=" * 70)
        print("🏥 医学分诊系统")
        print("=" * 70)

        print("""
👋 您好！欢迎使用医学分诊系统。

我可以帮您：
• 梳理您的症状
• 评估紧急程度
• 推荐合适的科室和医生

让我先了解您的情况...
        """)

        initial_symptom = input("请描述您哪里不舒服：\n").strip()

        if not initial_symptom:
            print("\n请告诉我您的症状，我才能帮您分诊。")
            return

        print(f"\n好的，我听到了。让我帮您理清一下...\n")

        result = self.clarifier.start_clarification(initial_symptom)

        print(f"\n💬 {result['question']}")

        for i in range(2, 7):
            answer = input(f"\n📝 您的回答：")
            result = self.clarifier.continue_clarification(answer)

            if not result["continue"]:
                break

            print(f"\n💬 {result['question']}")

        print("\n" + "=" * 70)
        print("✅ 症状梳理完成！")
        print("=" * 70)

        print("\n【您的症状总结】")
        print(result["summary"])

        print("\n" + "=" * 70)
        print("🔍 正在分析病情...")
        print("=" * 70)

        clarified_symptom = self.clarifier.get_clarified_symptom()

        triage_result = self.rag_engine.triage(clarified_symptom, k=5)

        print("\n" + "=" * 70)
        print("📋 分诊建议")
        print("=" * 70)

        if triage_result.get("success"):
            urgency = triage_result.get("urgency_level", {})
            urgency_text = urgency.get("name", "未知")
            urgency_color = urgency.get("color", "white")

            color_emoji = {
                "red": "🔴",
                "orange": "🟠",
                "yellow": "🟡",
                "green": "🟢"
            }

            emoji = color_emoji.get(urgency_color, "⚪")

            print(f"\n{emoji} 紧急程度：{urgency_text}\n")
            print("-" * 70)
            print("\n【分析结果】\n")
            print(triage_result["analysis"])

            hospital_context = triage_result.get("hospital_context", "")
            if hospital_context:
                print("\n" + "-" * 70)
                print("\n【本院就诊建议】\n")
                print(hospital_context)

            print("\n" + "=" * 70)
            print("💡 温馨提示")
            print("=" * 70)
            print("""
以上建议仅供参考，不能替代专业医生的诊断。

如有以下情况，请立即就医或拨打急救电话：
• 剧烈胸痛、呼吸困难
• 意识模糊、昏迷
• 严重出血
• 高热不退

祝您早日康复！
            """)

        else:
            print(f"\n[X] 分诊失败: {triage_result.get('error')}")

        print("\n" + "=" * 70)
        print("感谢使用医学分诊系统！")
        print("=" * 70)


def main():
    """主函数"""
    system = MedicalTriageSystem()

    if system.initialize():
        try:
            system.run_triage()
        except KeyboardInterrupt:
            print("\n\n已取消分诊。再见！")
        except Exception as e:
            print(f"\n\n[X] 系统错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n系统初始化失败，请检查配置。")


if __name__ == "__main__":
    main()
