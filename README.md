# 智招Agents —— 招商引资智能辅助系统

## 场景简述

面向地方政府招商部门，解决两个核心痛点：
1. **信息检索效率低**：从海量企业年报、行业报告、政策文件中快速定位目标企业关键信息和匹配政策
2. **谈判准备不充分**：缺乏模拟谈判演练手段，难以预先准备应对企业关心的土地、税收、人才等问题

本系统提供 **RAG智能信息查询** 和 **AI模拟谈判对话** 两大核心功能。

## 技术架构

```
┌─────────────────────────────────────────────┐
│              Streamlit Web UI               │
│          (8502端口，浏览器交互界面)           │
├──────────────┬──────────────────────────────┤
│  RAG 检索模块 │     模拟谈判 Agent            │
│  LangChain   │     OpenAI SDK               │
│  Chroma向量库 │     + 企业画像工具(Tool)       │
│  BGE Embedding│     + 多轮对话管理            │
├──────────────┴──────────────────────────────┤
│           DeepSeek API (LLM)                 │
│          本地 BGE-small-zh (Embedding)        │
├─────────────────────────────────────────────┤
│  本地知识库: 企业数据JSON + 政策文件JSON        │
│             + 招商政策知识库MD                  │
└─────────────────────────────────────────────┘
```

## 关键技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 核心框架 | LangChain | RAG管道和Agent工作流 |
| 向量数据库 | Chroma | 轻量级本地向量存储 |
| 大语言模型 | DeepSeek API | 生成回答、驱动谈判对话 |
| Embedding | BGE-small-zh (本地) | 中文语义向量，数据不出本地 |
| Web框架 | Streamlit | 快速构建交互式界面 |
| 模拟数据 | JSON + Python | 模拟企业画像查询 |

## 数据隐私设计

- **文件不上云**：文档解析、嵌入、存储全流程本地完成
- **最小化信息披露**：LLM交互仅发送脱敏后的查询文本和检索片段
- **模拟数据替代**：企业画像使用本地JSON预设数据，不接触真实敏感信息
- **配置隔离**：API Key通过.env管理，己加入.gitignore

## 运行说明

### 1. 环境要求
- Python 3.10+
- 已下载 BGE-small-zh 模型（或网络可访问HuggingFace）

### 2. 安装

```bash
cd smart-invest-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 4. 启动系统

```bash
streamlit run ui/app.py --server.port 8502 --server.address 0.0.0.0
```

浏览器打开 http://localhost:8502

## 使用示例

### 智能信息查询
1. 选择查询类型（企业查询/政策检索/自由提问）
2. 企业查询：选择目标企业 → 获得企业概况、投资动态、核心关注点
3. 政策检索：选择政策方向 → 获得匹配的优惠政策清单
4. 自由提问：输入任意招商相关问题

### 模拟谈判演练
1. 选择谈判对象企业 → 查看企业画像
2. 点击「开始谈判」→ AI以企业负责人身份开场
3. 输入谈判话术 → AI给出专业回应
4. 多轮对话后点击「结束本轮谈判」

## 项目结构

```
smart-invest-agent/
├── ui/app.py                  # Streamlit 前端
├── src/
│   ├── rag_engine.py          # RAG检索引擎
│   ├── negotiation_agent.py   # 模拟谈判Agent
│   └── enterprise_tool.py     # 企业画像工具(模拟数据)
├── data/
│   ├── enterprises/           # 企业模拟数据
│   ├── policies/              # 政策模拟数据
│   └── knowledge/             # 招商知识库
├── chroma_db/                 # Chroma向量存储（自动生成）
├── config.py                  # 配置管理
├── requirements.txt           # Python依赖
└── .env.example              # 环境变量模板
```

## 挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| HuggingFace不可访问 | 使用预先下载的本地BGE模型，开启HF_HUB_OFFLINE |
| DeepSeek API兼容性 | 使用OpenAI SDK + 自定义base_url |
| 谈判Agent角色一致性 | 系统提示词约束 + 企业画像上下文注入 |

## 后续改进方向

- 增加更多珠三角城市的企业和政策数据
- 接入真实企业信息API（企查查等）替代模拟数据
- 谈判过程记录与分析报告生成
- 支持语音交互
- Docker一键部署
