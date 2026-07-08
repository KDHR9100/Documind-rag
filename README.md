# DocuMind

> 基于 Rerank 与大模型的生产级本地化文档问答系统

DocuMind 是一个高性能的 RAG（检索增强生成）系统，能够将 PDF、Word、TXT 等文档转换为智能问答接口，支持通过 FastAPI 服务或交互式命令行进行文档查询。

---

## 🎯 项目目标

### 最终目标
打造一个**开箱即用的企业级文档问答系统**，实现：
- 📚 **多格式文档支持**：PDF、DOCX、TXT 等主流文档格式
- 🔍 **精准检索**：基于向量检索 + Cross-Encoder 重排序的双重检索机制
- ⚡ **高性能**：支持离线模式运行，无需网络即可完成文档处理
- 🌐 **多端访问**：提供 RESTful API 和交互式命令行两种访问方式
- 🐳 **容器化部署**：一键 Docker 部署，环境零配置

### 核心价值
- **知识沉淀**：将企业文档转化为可查询的知识库
- **效率提升**：员工快速获取文档信息，减少搜索时间
- **成本控制**：支持免费额度模型（qwen3.6-flash），降低运营成本

---

## 🛠️ 技术方案

### 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层                                  │
│  ┌──────────────────┐    ┌──────────────────────────────────┐  │
│  │  交互式命令行      │    │         FastAPI REST API         │  │
│  │  (pdf_ragpromax) │    │  /api/v1/query  /api/v1/index    │  │
│  └────────┬─────────┘    └──────────────────┬───────────────┘  │
└───────────┼─────────────────────────────────┼──────────────────┘
            │                                 │
┌───────────▼─────────────────────────────────▼──────────────────┐
│                        RAG 引擎层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  文档加载器   │→│  文本分割器   │→│     向量索引构建      │  │
│  │ (PyPDFLoader)│  │(TextSplitter)│  │   (Chroma + BGE)     │  │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘  │
│                                                  │              │
│  ┌───────────────────────────────────────────────▼───────────┐  │
│  │                    检索与重排序                            │  │
│  │  ┌─────────────┐    ┌─────────────────────────────────┐   │  │
│  │  │ 向量检索     │→→→│  Cross-Encoder 重排序            │   │  │
│  │  │ (top_k=5)   │    │ (BAAI/bge-reranker-v2-m3)      │   │  │
│  │  └─────────────┘    └─────────────────┬───────────────┘   │  │
│  └───────────────────────────────────────│───────────────────┘  │
│                                          │                      │
│  ┌───────────────────────────────────────▼───────────────────┐  │
│  │                    LLM 生成                               │  │
│  │            ChatTongyi / qwen3.6-flash                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                        数据层                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   docs/      │    │ vector_store/│    │   config/        │  │
│  │  原始文档     │    │  向量数据库   │    │  配置文件         │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 技术栈

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **框架** | LangChain | 0.3.x | RAG 核心框架 |
| **向量存储** | ChromaDB | 0.6.x | 轻量级向量数据库 |
| **嵌入模型** | SentenceTransformers | 3.4.x | BGE-large-zh 嵌入 |
| **重排序** | CrossEncoder | - | BGE-reranker-v2-m3 |
| **LLM** | DashScope | 1.26.x | 通义千问 API |
| **API服务** | FastAPI | 0.139.x | RESTful API |
| **服务器** | Uvicorn | 0.50.x | ASGI 服务器 |
| **包管理** | uv | 0.11.x | 极速 Python 包管理 |
| **部署** | Docker / Docker Compose | - | 容器化部署 |

### 关键技术亮点

1. **混合检索 + 重排序**
   - 首先通过向量检索获取 Top-K 候选文档
   - 然后使用 Cross-Encoder 进行精排，提升相关性

2. **多模态接口支持**
   - 自动检测模型类型，qwen3.6-flash 使用多模态接口
   - 其他模型使用标准文本接口

3. **离线模式运行**
   - 嵌入模型和重排序模型本地化部署
   - 支持 `TRANSFORMERS_OFFLINE` 模式，无需网络即可运行

4. **生产级特性**
   - API 自动重试机制（tenacity）
   - Token 消耗和费用统计
   - 健康检查接口

---

## 📁 项目结构

```
DocuMind/
├── app/                      # FastAPI 应用
│   ├── api/
│   │   └── routes.py         # API 路由定义
│   ├── core/
│   │   └── rag_engine.py     # RAG 引擎核心逻辑
│   ├── models/
│   │   └── schemas.py        # Pydantic 数据模型
│   ├── utils/
│   │   └── config_loader.py  # 配置加载工具
│   └── main.py               # 应用入口
├── config/                   # 配置文件
│   └── config.json           # 主配置文件
├── docs/                     # 文档目录（放置PDF等文件）
├── vector_store/             # 向量索引目录（自动生成）
├── pdf_ragpromax.py          # 交互式命令行入口
├── config.json               # 根目录配置文件
├── pyproject.toml            # 项目依赖声明
├── uv.lock                   # 依赖锁定文件
├── requirements.txt          # pip 格式依赖列表
├── Dockerfile                # Docker 镜像构建文件
├── docker-compose.yml        # Docker Compose 配置
└── .env.example              # 环境变量示例
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Conda 或 uv（推荐）
- 通义千问 API Key（[申请地址](https://dashscope.console.aliyun.com/)）

### 方式一：使用 Conda（推荐）

```bash
# 1. 创建并激活环境
conda create -n documind python=3.11 -y
conda activate documind

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置环境变量
cp .env.example .env
# 编辑 .env 文件，填入您的 DASHSCOPE_API_KEY

# 4. 运行交互式问答
python pdf_ragpromax.py

# 5. 或运行 FastAPI 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 方式二：使用 uv

```bash
# 1. 创建虚拟环境并安装依赖
uv sync

# 2. 设置环境变量
cp .env.example .env
# 编辑 .env 文件

# 3. 运行
uv run python pdf_ragpromax.py
# 或
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 方式三：使用 Docker（推荐用于生产环境）

```bash
# 1. 设置环境变量
cp .env.example .env
# 编辑 .env 文件

# 2. 构建并启动
docker-compose up --build -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

---

## 📖 使用指南

### 1. 准备文档

将 PDF、DOCX、TXT 文件放入 `docs/` 目录：

```
docs/
├── 用户手册.pdf
├── 产品说明书.docx
└── FAQ.txt
```

### 2. 运行交互式问答

```bash
python pdf_ragpromax.py
```

```
💬 交互式查询模式 (输入 'exit' 退出)

您的问题: 产品的主要功能有哪些？
```

### 3. 使用 API

**健康检查**
```bash
curl http://localhost:8000/health
```

**问答接口**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "产品的主要功能有哪些？"}'
```

**响应示例**
```json
{
    "question": "产品的主要功能有哪些？",
    "answer": "根据文档内容，本产品主要包含以下功能：...",
    "stats": {
        "api_calls": 1,
        "total_cost": 0.002,
        "input_tokens": 10,
        "output_tokens": 150
    }
}
```

**重建索引**
```bash
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

### 4. API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## ⚙️ 配置说明

### config.json 主要配置项

```json
{
    "model_config": {
        "llm": {
            "model": "qwen3.6-flash",
            "temperature": 0.5,
            "max_tokens": 2500
        },
        "embedding": {
            "model": "BAAI/bge-large-zh",
            "device": "cpu",
            "normalize_embeddings": true
        }
    },
    "rag_config": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "rerank_top_n": 3,
        "rerank_model": "BAAI/bge-reranker-v2-m3"
    },
    "paths": {
        "docs": "./docs",
        "vector_store": "./vector_store"
    }
}
```

### 支持的 LLM 模型

| 模型名称 | 特点 | 免费额度 |
|----------|------|----------|
| `qwen3.6-flash` | 高性能，1M上下文 | ✅ 有 |
| `qwen-plus` | 标准版，平衡性能 | ❌ 无 |
| `qwen-turbo` | 极速版，响应快 | ✅ 有 |
| `qwen-max` | 高级版，最强能力 | ❌ 无 |

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 文档加载速度 | 取决于文档大小 |
| 向量构建速度 | ~500 docs/s (CPU) |
| 检索延迟 | ~100ms (含重排序) |
| API 响应延迟 | ~1-3s (含 LLM 生成) |
| 支持文档格式 | PDF, DOCX, TXT |
| 最大文档数量 | 无限制 |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发流程

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/xxx`
3. 提交代码：`git commit -m "feat: xxx"`
4. 推送到远程：`git push origin feature/xxx`
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 添加必要的注释
- 编写单元测试

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 🙋‍♂️ 常见问题

### Q: 为什么 API 返回 403 错误？
A: 免费额度已用完，请登录阿里云控制台关闭"仅使用免费额度"模式或完成付费信息填写。

### Q: 如何更换模型？
A: 修改 `config.json` 中的 `model_config.llm.model` 字段即可。

### Q: 向量索引在哪里？
A: 向量索引存储在 `vector_store/` 目录，删除该目录可强制重建索引。

### Q: 支持 GPU 加速吗？
A: 支持，修改 `config.json` 中 `embedding.device` 为 `"cuda"` 即可。

---

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 Issue
- 发送邮件

---

**DocuMind** - 让文档问答更智能 ✨