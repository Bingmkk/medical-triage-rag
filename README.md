# 基于ReAct推理与RAG的智能医学分诊Agent

基于ReAct推理范式与RAG技术栈的智能医学分诊Agent系统，整合19,225条医学知识向量与NetworkX知识图谱，实现症状输入→智能追问→分诊结果流式输出的完整流程。底层构建混合检索引擎（Faiss语义检索70%+BM25关键词匹配30%），Agent联合知识图谱进行关联推理，结合智能分流工具实现时间感知的科室/医生精准推荐。

## 🌟 核心亮点

- 🤖 **ReAct Agent驱动**：基于ReAct推理范式，实现智能症状追问与分诊决策
- 🔬 **混合检索增强**：Faiss向量检索(70%) + BM25关键词匹配(30%) + LLM Rerank
- 📊 **知识图谱推理**：基于NetworkX构建疾病-症状-药物-科室关系网络
- 💬 **多轮追问系统**：模拟医生问诊流程，智能动态追问（最多5轮）
- 🏥 **时间感知分流**：根据当前时间智能推荐在班医生，按出诊状态+专业匹配+职称排序
- ⚡ **流式输出**：支持实时SSE流式输出，提升用户体验

## 🛠️ 技术栈

| 分类 | 技术 | 说明 |
|------|------|------|
| **Agent框架** | ReAct范式 | Thought-Action-Observation推理循环 |
| **混合检索** | Faiss + BM25 | 语义检索(70%) + 关键词匹配(30%) |
| **Rerank精排** | LLM | 大模型二次排序，提升准确率 |
| **知识图谱** | NetworkX | 19,225条医学知识向量，GraphML格式 |
| **智能分流** | SmartReferral | 时间感知 + 专业匹配 + 职称评分（出诊100分+专业50分+职称40分） |
| **嵌入模型** | SentenceTransformer | paraphrase-multilingual-MiniLM-L12-v2（384维） |
| **LLM支持** | 通义千问/DeepSeek/智谱AI | 多平台统一接入 |

## 📁 项目结构

```
medical_triage_rag/
├── app.py                         # Flask Web应用入口
├── config.py                      # 配置文件（LLM配置、路径配置）
├── requirements.txt               # Python依赖列表
├── .env                           # 环境变量（API密钥）
├── .env.example                   # 环境变量示例
├── api/                           # RESTful API模块
│   ├── __init__.py
│   └── routes.py                  # API路由定义（分诊、追问、分流接口）
├── src/                           # 核心源代码
│   ├── __init__.py
│   ├── rag_engine.py              # 统一RAG引擎（混合检索+知识图谱+医院RAG）
│   ├── llm_client.py              # LLM客户端（支持多平台流式输出）
│   ├── knowledge_graph.py         # 医学知识图谱模块
│   ├── hybrid_search_rerank.py    # 混合检索（Faiss+BM25）+ Rerank实现
│   ├── symptom_clarifier.py       # 智能症状追问系统（多轮对话）
│   ├── hospital_rag.py            # 医院信息RAG检索
│   ├── hospital_loader.py         # 医院科室数据加载+中文症状匹配
│   ├── smart_referral.py          # 智能分流工具（时间感知+医生排序）
│   ├── faiss_loader.py            # Faiss向量索引加载
│   ├── prompts.py                 # 提示词模板（分诊专用）
│   ├── triage_agent.py            # 分诊Agent（ReAct推理）
│   └── department_recommender.py  # 科室推荐器
├── data/                          # 数据文件
│   ├── medical_books.index        # Faiss向量索引（19,225条）
│   ├── chunks.json                # 医学知识分块
│   ├── medical_kg.graphml         # 知识图谱（GraphML格式）
│   ├── hospital_departments.json   # 医院科室数据（含排班、职称、诊室）
│   └── medical_books_metadata.json # 医学书籍元数据
├── templates/                     # HTML模板
│   └── index.html                 # 主页面（三阶段分诊流程+智能分流UI）
└── run.ps1 / run.bat             # 启动脚本
```

## 🚀 运行方式

### 前置条件

使用已有 Conda 环境 `pytorch2.2.2`：

```powershell
conda activate pytorch2.2.2
cd E:\TraePJ\AGENT\medical_triage_rag
```

### 安装依赖

```powershell
pip install -r requirements.txt
```

### 配置API密钥

复制 `.env.example` 为 `.env` 并填入 API 密钥（至少配置一个）：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
# ZHIPU_API_KEY=your_zhipu_api_key_here
```

### 启动服务

```powershell
python app.py
```

或使用启动脚本：`.\run.ps1`（PowerShell）或 `run.bat`（CMD）

访问 http://localhost:5000

## 🔌 API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/triage` | POST | 普通分诊（含智能分流） |
| `/api/triage/stream` | POST | 流式分诊（SSE，含智能分流） |
| `/api/clarify/start` | POST | 开始症状追问 |
| `/api/clarify/answer` | POST | 继续追问 |
| `/api/clarify/confirm` | POST | 确认追问并分诊 |
| `/api/department/recommend` | POST | 科室推荐 |
| `/api/chat` | POST | 自由对话 |
| `/api/doctor/info` | GET | 医生信息查询 |
| `/api/hospital` | GET | 医院基本信息 |

## 🧠 Agent Tools

| 工具 | 功能 | 调用方式 |
|------|------|---------|
| **Faiss语义检索** | 384维向量相似度搜索 | 本地计算(0 API) |
| **BM25关键词检索** | 词频-逆文档频率精确匹配 | 本地计算(0 API) |
| **LLM Rerank** | 大模型二次排序 | 1次API调用(可选) |
| **知识图谱查询** | 疾病-症状-药物-科室关系推理 | 本地计算(0 API) |
| **时间感知分流** | 按当前时间推荐在班医生 | 本地计算(0 API) |
| **症状追问** | 多轮对话收集关键信息 | 1次API/轮 |

## 🔬 检索引擎架构

```
用户症状
    ↓
┌──────────────────────────────────────────────┐
│            混合检索引擎                        │
├──────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Faiss语义检索    │  │ BM25关键词匹配    │   │
│  │ (70%权重)        │  │ (30%权重)        │   │
│  │ 384维向量相似度  │  │ rank_bm25算法    │   │
│  └─────────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────┘
    ↓ 候选结果
┌──────────────────────────────────────────────┐
│    知识图谱增强 + 中文症状科室匹配           │
└──────────────────────────────────────────────┘
    ↓ 增强上下文
┌──────────────────────────────────────────────┐
│  Agent推理 + 智能分流（时间感知+医生排序）   │
└──────────────────────────────────────────────┘
```

## 🎯 紧急程度分级

| 等级 | 名称 | 颜色 | 说明 |
|------|------|------|------|
| 1 | 危急 | 🔴 红色 | 危及生命，需立即拨打120 |
| 2 | 紧急 | 🟠 橙色 | 严重威胁健康，需尽快就医 |
| 3 | 次紧急 | 🟡 黄色 | 需要处理但不紧急 |
| 4 | 非紧急 | 🟢 绿色 | 可以门诊预约 |

## ⚠️ 免责声明

**本系统仅供初步参考，不能替代专业医生的诊断和治疗。**

- 对于危急症状，请立即拨打120急救电话或前往最近的急诊科
- 请在专业医生的指导下进行任何医疗决策

---

**版本**: v2.0.0
**更新日期**: 2026-05
**数据规模**: 19,225条医学知识向量 | 知识图谱实体关系 | 280+医生 | 7个科室
