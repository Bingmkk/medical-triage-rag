# 医学分诊RAG系统

基于LangChain + Faiss + 知识图谱的智能医学分诊系统，提供完整的症状追问、智能分诊和科室推荐功能。

## 🌟 主要特性

- 🔬 **混合检索**: 结合Faiss向量检索（70%）+ 关键词匹配（30%），支持Rerank精排
- 📊 **知识图谱**: 基于NetworkX构建医学知识图谱，包含疾病、症状、药物、科室等实体关系
- 🤖 **智能分诊**: AI驱动的紧急程度评估（4级分级）和精准科室推荐
- 💬 **症状追问**: 多轮智能追问系统，自动避免重复提问，支持紧急情况检测
- 🏥 **医院RAG**: 科室信息向量化，通过向量检索匹配最适合的科室和医生
- ⚡ **流式输出**: 支持实时流式输出，提升用户体验
- 🌐 **Web界面**: 三阶段完整分诊流程（症状输入 → 症状追问 → 分诊结果）

## 🛠️ 技术栈

| 分类 | 技术 | 说明 |
|------|------|------|
| **核心框架** | LangChain | LLM应用开发框架 |
| **向量检索** | Faiss | Facebook AI相似性搜索（IP索引，19,225条医学知识） |
| **知识图谱** | NetworkX | 医学实体关系网络（GraphML格式） |
| **嵌入模型** | SentenceTransformer | paraphrase-multilingual-MiniLM-L12-v2（384维） |
| **Web服务** | Flask | 轻量级Web框架 |
| **LLM支持** | 通义千问 / DeepSeek / 智谱AI | 多平台统一接入 |

## 📁 项目结构

```
medical_triage_rag/
├── app.py                    # Flask Web应用入口
├── config.py                 # 配置文件（LLM配置、路径配置）
├── requirements.txt          # Python依赖列表
├── .env                      # 环境变量（API密钥）
├── .env.example              # 环境变量示例
├── PROJECT_SUMMARY.md        # 项目总结文档
├── api/                      # RESTful API模块
│   ├── __init__.py
│   └── routes.py             # API路由定义
├── src/                      # 核心源代码
│   ├── __init__.py
│   ├── rag_engine.py         # 统一RAG引擎（混合检索+知识图谱+医院RAG）
│   ├── llm_client.py         # LLM客户端（支持多平台）
│   ├── knowledge_graph.py    # 医学知识图谱模块
│   ├── hybrid_search_rerank.py # 混合检索+Rerank实现
│   ├── symptom_clarifier.py  # 智能症状追问系统
│   ├── hospital_rag.py       # 医院信息RAG检索
│   ├── hospital_loader.py    # 医院科室数据加载
│   ├── faiss_loader.py       # Faiss向量索引加载
│   ├── prompts.py            # 提示词模板（分诊专用）
│   ├── triage_agent.py       # 分诊Agent
│   └── smart_referral.py     # 智能推荐工具
├── data/                     # 数据文件
│   ├── medical_books.index   # Faiss向量索引（19,225条）
│   ├── chunks.json           # 医学知识分块
│   ├── medical_kg.graphml    # 知识图谱（GraphML格式）
│   ├── hospital_departments.json # 医院科室数据
│   └── medical_books_metadata.json # 医学书籍元数据
└── templates/                # HTML模板
    └── index.html            # 主页面（三阶段分诊流程）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

复制`.env.example`为`.env`并填入API密钥：

```bash
cp .env.example .env
```

编辑`.env`文件：

```env
# 至少配置一个API密钥
DASHSCOPE_API_KEY=your_dashscope_api_key_here
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
# ZHIPU_API_KEY=your_zhipu_api_key_here
```

获取API密钥：
- **通义千问**: https://dashscope.console.aliyun.com/
- **DeepSeek**: https://platform.deepseek.com/
- **智谱AI**: https://open.bigmodel.cn/

### 3. 启动服务

```bash
python app.py
```

### 4. 访问服务

启动后访问：http://localhost:5000

## 🔌 API接口

### 1. 开始症状追问

```bash
POST /api/clarify/start

{
    "symptom": "头痛、头晕、伴有恶心"
}

# 返回
{
    "continue": true,
    "question": "您说头痛头晕，这种情况大概持续多久了？",
    "round": 1,
    "max_rounds": 5,
    "progress": "1/5"
}
```

### 2. 继续追问

```bash
POST /api/clarify/answer

{
    "session_id": "xxx",
    "answer": "持续了大概2小时"
}

# 返回
{
    "continue": true,
    "question": "头痛是一直疼还是一阵阵的？",
    "round": 2,
    "progress": "2/5"
}
```

### 3. 确认追问并分诊

```bash
POST /api/clarify/confirm

{
    "session_id": "xxx"
}

# 返回
{
    "continue": false,
    "summary": "【症状总结】...",
    "can_proceed_to_triage": true
}
```

### 4. 直接分诊

```bash
POST /api/triage

{
    "symptom_description": "头痛、头晕、伴有恶心，持续2小时"
}

# 返回
{
    "success": true,
    "symptom_description": "...",
    "analysis": "...",
    "urgency_level": {
        "level": 3,
        "name": "次紧急",
        "color": "yellow"
    },
    "hospital_context": "..."
}
```

### 5. 流式分诊

```bash
POST /api/triage/stream

{
    "symptom_description": "头痛、头晕"
}

# 返回（SSE流式）
data: {"status": "retrieving", "message": "正在检索相关知识..."}
data: {"status": "generating", "content": "【紧急程度】"}
data: {"status": "generating", "content": "非紧急"}
data: {"status": "done", "result": {...}}
```

### 6. 获取推荐科室

```bash
POST /api/department/recommend

{
    "symptom": "头痛"
}

# 返回
{
    "success": true,
    "departments": [
        {"name": "神经内科", "doctors": [...], "location": "..."}
    ]
}
```

## 📋 分诊流程

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: 症状输入                                             │
│ 用户描述症状 → 系统初步分析                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段2: 症状追问（最多5轮）                                   │
│ 智能追问 → 避免重复 → 紧急检测 → 收集关键信息               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段3: 分诊结果展示                                         │
│ 紧急程度评估 → 症状分析 → 科室推荐 → 医生信息               │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 紧急程度分级

| 等级 | 名称 | 颜色 | 说明 |
|------|------|------|------|
| 1 | 危急 | 🔴 红色 | 危及生命，需立即拨打120 |
| 2 | 紧急 | 🟠 橙色 | 严重威胁健康，需尽快就医 |
| 3 | 次紧急 | 🟡 黄色 | 需要处理但不紧急 |
| 4 | 非紧急 | 🟢 绿色 | 可以门诊预约 |

## 🧠 RAG引擎工作原理

```
用户症状 → 向量化 → Faiss检索（语义匹配）
                  → 知识图谱检索（关系推理）
                  → 医院RAG检索（科室匹配）
                        ↓
                  混合上下文 → LLM生成 → 分诊结果
```

### 检索策略

1. **Faiss向量检索**: 基于SentenceTransformer生成384维向量，IP内积相似度匹配
2. **关键词匹配**: 补充基于TF-IDF的关键词检索（30%权重）
3. **Rerank精排**: 对候选结果进行二次排序，提升Top-3准确率
4. **知识图谱增强**: 从疾病-症状-药物-科室关系中获取结构化知识

## 🔧 配置说明

### LLM配置 (`config.py`)

```python
LLM_CONFIG = {
    "qwen-turbo": {
        "provider": "dashscope",
        "model_name": "qwen-turbo",
        "temperature": 0.3,  # 降低随机性，提升一致性
        "max_tokens": 1000   # 精简输出，加快响应
    }
}
```

### 检索配置 (`src/hybrid_search_rerank.py`)

```python
# 混合检索参数
VECTOR_WEIGHT = 0.7    # 向量检索权重
KEYWORD_WEIGHT = 0.3   # 关键词检索权重
TOP_K = 5              # 检索数量
USE_RERANK = True      # 是否启用精排
```

## ⚠️ 免责声明

**本系统仅供初步参考，不能替代专业医生的诊断和治疗。**

- 对于危急症状，请立即拨打120急救电话或前往最近的急诊科
- 请在专业医生的指导下进行任何医疗决策
- 系统可能存在误判风险，使用前请知悉并自行承担风险

## 📝 许可证

MIT License

## 📧 联系方式

如有问题或建议，请联系开发团队。

---

**版本**: v2.0.0  
**更新日期**: 2026  
**数据规模**: 19,225条医学知识向量 | 知识图谱实体关系 | 医院科室数据