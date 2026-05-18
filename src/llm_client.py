"""
LLM客户端模块 - 支持多种国内大语言模型
"""
from typing import Optional, Dict, Any, List, Generator
import os
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from config import LLM_CONFIG


class LLMClient:
    """支持多种LLM的客户端"""

    def __init__(self, model_type: str = "qwen", streaming: bool = True):
        if model_type not in LLM_CONFIG:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {list(LLM_CONFIG.keys())}")

        self.config = LLM_CONFIG[model_type]
        self.model_type = model_type
        self.streaming = streaming
        self._client = None

    def _get_client(self) -> ChatOpenAI:
        if self._client is None:
            if self.config["provider"] == "dashscope":
                os.environ["DASHSCOPE_API_KEY"] = self.config["api_key"]

            callbacks = []
            if self.streaming:
                callbacks.append(StreamingStdOutCallbackHandler())

            self._client = ChatOpenAI(
                model=self.config["model_name"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                api_key=self.config["api_key"],
                base_url=self.config["api_base"],
                streaming=self.streaming,
                callback_manager=CallbackManager(callbacks) if callbacks else None
            )
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        client = self._get_client()
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        response = client(messages)
        return response.content

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        client = self._get_client()
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        for chunk in client.stream(messages):
            if chunk.content:
                yield chunk.content


class MedicalLLMClient(LLMClient):
    """专门用于医学领域的LLM客户端"""

    def __init__(self, model_type: str = "qwen", streaming: bool = True):
        super().__init__(model_type, streaming)
        self.medical_system_prompt = """你是一个专业的医学分诊助手，具有以下能力：
1. 根据患者的症状描述，进行初步的医学分诊
2. 评估病情的紧急程度（危急、紧急、次紧急、非紧急）
3. 提供初步的医疗建议和注意事项
4. 根据知识图谱和检索结果，给出专业的医学建议

请注意：
- 你的建议仅供参考，不能替代专业医生的诊断
- 对于危急情况，应立即建议患者拨打急救电话或前往急诊
- 请使用清晰易懂的语言，避免过度专业的术语
- 结合知识图谱中的医学知识进行推理

请开始帮助患者进行分诊。"""

    def generate_medical_response(self, user_input: str, context: Optional[str] = None) -> str:
        if context:
            prompt = f"""基于以下医学知识信息：
{context}

患者描述：{user_input}

请根据以上信息进行分诊。"""
        else:
            prompt = f"""患者描述：{user_input}

请根据患者描述进行分诊。"""

        return self.generate(prompt, self.medical_system_prompt)
