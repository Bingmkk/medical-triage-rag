"""
医学分诊RAG系统配置文件
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
KG_STORE_DIR = BASE_DIR / "knowledge_graph"

DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
KG_STORE_DIR.mkdir(exist_ok=True)

LLM_CONFIG = {
    "qwen": {
        "provider": "dashscope",
        "model_name": "qwen-turbo",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "temperature": 0.3,
        "max_tokens": 1000
    },
    "deepseek": {
        "provider": "openai",
        "model_name": "deepseek-chat",
        "api_base": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.3,
        "max_tokens": 1000
    },
    "zhipu": {
        "provider": "zhipu",
        "model_name": "glm-4",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "temperature": 0.3,
        "max_tokens": 1000
    }
}

DEFAULT_LLM = "qwen"

EMBEDDING_CONFIG = {
    "provider": "dashscope",
    "model_name": "text-embedding-v3",
    "api_base": "https://dashscope.aliyuncs.com/api/v1",
    "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
    "dimension": 1536
}

FAISS_CONFIG = {
    "index_type": "IP",
    "nlist": 100,
    "nprobe": 10
}

TRIAGE_CONFIG = {
    "urgency_levels": [
        {"level": 1, "name": "危急", "description": "立即危及生命，需立即处理", "color": "red"},
        {"level": 2, "name": "紧急", "description": "严重威胁健康，需尽快处理", "color": "orange"},
        {"level": 3, "name": "次紧急", "description": "需要处理但不紧急", "color": "yellow"},
        {"level": 4, "name": "非紧急", "description": "可以等待或门诊处理", "color": "green"}
    ],
    "max_retrieval_docs": 5,
    "min_confidence_threshold": 0.6
}

RAG_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retrieval_top_k": 5,
    "rerank_top_n": 3
}

WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}
