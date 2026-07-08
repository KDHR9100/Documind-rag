"""
DocuMind: 基于Rerank与大模型的本地化文档问答系统
重构版 - 生产级RAG流水线
环境要求: Python 3.9+, langchain-huggingface, langchain-community, dashscope, chromadb
支持通过 config.json 配置所有参数，并支持命令行覆盖
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain 核心组件
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 向量存储与嵌入
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 文档加载器
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# LLM & Rerank
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import dashscope
from dashscope import Generation, MultiModalConversation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 JSON 配置文件，若文件不存在则返回空字典"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.warning(f"配置文件 {config_path} 不存在，使用默认参数")
        return {}


class DocuMindRAG:
    """DocuMind: 生产级RAG系统，支持配置化"""

    # 文档加载器映射（支持扩展）
    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 RAG 系统
        :param config: 配置字典，包含 model_config, rag_config, paths 等
        """
        # ---------- 读取路径配置 ----------
        paths = config.get("paths", {})
        self.docs_dir = Path(paths.get("docs", "./docs"))
        self.vector_store_dir = Path(paths.get("vector_store", "./vector_store"))

        # ---------- 检查 API Key ----------
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise EnvironmentError("❌ 未设置 DASHSCOPE_API_KEY 环境变量，请在 .env 或系统中设置")

        # ---------- 读取 RAG 参数 ----------
        rag_cfg = config.get("rag_config", {})
        chunk_size = rag_cfg.get("chunk_size", 500)
        chunk_overlap = rag_cfg.get("chunk_overlap", 50)
        separators = rag_cfg.get("separators", ["\n\n", "\n", "。", "！", "？", "；", " ", ""])
        self.top_k = rag_cfg.get("top_k", 5)           # 向量检索返回数量
        self.rerank_top_n = rag_cfg.get("rerank_top_n", 3)  # 重排序后保留数量

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        # ---------- 加载嵌入模型 ----------
        embedding_cfg = config.get("model_config", {}).get("embedding", {})
        embed_model_name = embedding_cfg.get("model", "BAAI/bge-large-zh")
        device = embedding_cfg.get("device", "cpu")
        normalize = embedding_cfg.get("normalize_embeddings", True)

        logger.info(f"📦 加载嵌入模型 {embed_model_name} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize},
        )

        # ---------- 加载大模型 ----------
        llm_cfg = config.get("model_config", {}).get("llm", {})
        llm_model = llm_cfg.get("model", "qwen-turbo")
        temperature = llm_cfg.get("temperature", 0.1)
        max_tokens = llm_cfg.get("max_tokens", 2000)

        logger.info(f"📦 加载大模型 {llm_model} ...")
        
        if llm_model.startswith("qwen3.") or llm_model.startswith("qwen36"):
            logger.info(f"  使用多模态接口调用 {llm_model}")
            from langchain_core.language_models import BaseChatModel
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
            
            class DashScopeMultiModalLLM(BaseChatModel):
                model_name: str
                temperature: float = 0.5
                max_tokens: int = 2000
                
                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    dashscope_messages = []
                    for msg in messages:
                        if isinstance(msg, HumanMessage):
                            dashscope_messages.append({"role": "user", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            dashscope_messages.append({"role": "assistant", "content": msg.content})
                        elif isinstance(msg, SystemMessage):
                            dashscope_messages.append({"role": "system", "content": msg.content})
                    
                    response = MultiModalConversation.call(
                        model=self.model_name,
                        messages=dashscope_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    if response.status_code != 200 or response.output is None:
                        error_msg = f"API Error: {response.code} - {response.message}" if hasattr(response, 'code') else "Unknown API error"
                        raise RuntimeError(error_msg)
                    
                    content = response.output.choices[0].message.content
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    else:
                        content = str(content)
                    
                    return ChatResult(
                        generations=[ChatGeneration(message=AIMessage(content=content))]
                    )
                
                @property
                def _llm_type(self):
                    return "dashscope_multimodal"
                
                @property
                def _identifying_params(self):
                    return {"model_name": self.model_name}
            
            self.llm = DashScopeMultiModalLLM(
                model_name=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            self.llm = ChatTongyi(
                model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # ---------- 加载重排序模型 ----------
        rerank_model_name = rag_cfg.get("rerank_model", "BAAI/bge-reranker-v2-m3")
        logger.info(f"📦 加载重排序模型 {rerank_model_name} ...")
        rerank_model = HuggingFaceCrossEncoder(model_name=rerank_model_name)
        self.compressor = CrossEncoderReranker(model=rerank_model, top_n=self.rerank_top_n)

        # ---------- 运行时统计 ----------
        self.stats = {"api_calls": 0, "total_cost": 0.0, "input_tokens": 0, "output_tokens": 0}

        self.vector_store: Optional[Chroma] = None
        self.chain = None

    def load_documents(self) -> List[Document]:
        """加载文档目录下的所有支持文件"""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"❌ 文档目录不存在: {self.docs_dir}")

        documents = []
        files = [f for f in self.docs_dir.rglob("*") if f.suffix.lower() in self.LOADER_MAP]

        if not files:
            logger.warning(f"⚠️ 未在 {self.docs_dir} 中找到支持的文档文件")
            return documents

        logger.info(f"📁 发现 {len(files)} 个文档文件")
        for i, file_path in enumerate(files, 1):
            loader_cls = self.LOADER_MAP.get(file_path.suffix.lower())
            if loader_cls is None:
                continue
            try:
                loader = loader_cls(str(file_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({"source": str(file_path), "filename": file_path.name})
                documents.extend(docs)
                logger.info(f"  [{i}/{len(files)}] ✅ {file_path.name} ({len(docs)} 页)")
            except Exception as e:
                logger.error(f"  [{i}/{len(files)}] ❌ {file_path.name}: {e}")

        logger.info(f"📄 共加载 {len(documents)} 个文档块")
        return documents

    def build_index(self, force_rebuild: bool = False):
        """构建或加载向量索引"""
        logger.info("=" * 60)
        logger.info("🔧 初始化向量索引")
        logger.info("=" * 60)

        if not force_rebuild and self.vector_store_dir.exists():
            try:
                logger.info(f"♻️ 检测到已有索引，直接加载: {self.vector_store_dir}")
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_store_dir),
                    embedding_function=self.embeddings,
                )
                count = self.vector_store._collection.count()
                logger.info(f"✅ 索引加载成功，包含 {count} 个向量")
                self._setup_chain()
                return
            except Exception as e:
                logger.warning(f"⚠️ 索引加载失败，将重新构建: {e}")

        documents = self.load_documents()
        if not documents:
            logger.error("❌ 没有可处理的文档，索引构建终止")
            return

        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"✂️ 分割为 {len(split_docs)} 个文档块")

        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=str(self.vector_store_dir),
        )
        logger.info(f"✅ 向量索引构建完成 → {self.vector_store_dir}")
        self._setup_chain()

    def _setup_chain(self):
        """构建 RAG 链（包含检索、重排序、提示、LLM）"""
        if self.vector_store is None:
            raise RuntimeError("向量索引未初始化")

        # 基础检索：返回 top_k 个文档（注意这里使用 self.top_k）
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        # 压缩检索器：内部使用 CrossEncoderReranker 重排序
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever,
        )

        prompt = ChatPromptTemplate.from_template(
            "基于以下文档内容回答问题。如果文档中没有相关信息，请明确说明。\n\n"
            "文档内容：\n{context}\n\n"
            "问题：{question}\n\n"
            "请提供详细、准确的回答："
        )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            RunnableParallel({
                "context": compression_retriever | format_docs,
                "question": RunnablePassthrough(),
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("✅ RAG查询链构建完成 (含Cross-Encoder Rerank)")

    def query(self, question: str) -> Dict[str, Any]:
        """执行单次问答"""
        if self.chain is None:
            return {"question": question, "answer": "❌ 系统未初始化，请先调用 build_index()"}

        logger.info(f"\n{'='*60}\n❓ {question}\n{'='*60}")

        # 执行链调用
        result = self.chain.invoke(question, config={"metadata": {"query": question}})

        # 统计（估算 token 和费用）
        self.stats["api_calls"] += 1
        input_chars, output_chars = len(question), len(result)
        est_input_tokens = max(1, input_chars // 2)
        est_output_tokens = max(1, output_chars // 2)
        # 定价可配置（此处取 qwen-plus 价格）
        cost = (est_input_tokens * 0.004 + est_output_tokens * 0.012) / 1000

        self.stats["total_cost"] += cost
        self.stats["input_tokens"] += est_input_tokens
        self.stats["output_tokens"] += est_output_tokens

        logger.info(f"💬 回答生成完毕 | Token≈{est_input_tokens}+{est_output_tokens} | 费用≈¥{cost:.4f}")
        print(f"\n{result}")

        return {"question": question, "answer": result}

    def print_stats(self):
        """打印统计信息"""
        s = self.stats
        logger.info(f"\n📊 统计 | API调用:{s['api_calls']} | "
                    f"Token:{s['input_tokens']}+{s['output_tokens']} | "
                    f"费用:¥{s['total_cost']:.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DocuMind RAG 系统")
    parser.add_argument("--config", type=str, default="config.json",
                        help="配置文件路径 (默认: config.json)")
    parser.add_argument("--docs", type=str, default=None,
                        help="文档目录路径 (覆盖配置文件中的设置)")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="强制重建向量索引")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载配置文件
    config = load_config(args.config)

    # 2. 命令行参数覆盖
    if args.docs:
        # 确保 paths 段存在
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["docs"] = args.docs

    # 3. 初始化 RAG 系统
    rag = DocuMindRAG(config)

    # 4. 构建索引
    rag.build_index(force_rebuild=args.force_rebuild)

    # 5. 交互式问答
    print("\n💬 交互式查询模式 (输入 'exit' 退出)")
    while True:
        try:
            q = input("\n您的问题: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if q:
                rag.query(q)
        except KeyboardInterrupt:
            break

    rag.print_stats()


if __name__ == "__main__":
    main()