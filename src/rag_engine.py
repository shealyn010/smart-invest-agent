"""RAG信息检索引擎 - 基于Chroma + 本地Embedding的政策/知识检索"""
import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from config import POLICY_DIR, KNOWLEDGE_DIR, CHROMA_DIR, EMBEDDING_MODEL, EMBEDDING_DEVICE


class RAGEngine:
    """本地RAG检索引擎：加载政策JSON和知识库Markdown，构建向量索引"""

    def __init__(self):
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE, "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = None
        self._build_index()

    def _load_documents(self) -> list:
        """加载知识库文档"""
        docs = []

        # 1. 加载政策JSON
        policy_path = POLICY_DIR / "policies.json"
        if policy_path.exists():
            with open(policy_path) as f:
                policies = json.load(f)["policies"]
            for p in policies:
                text = f"【{p['level']}政策】{p['title']}\n发文单位：{p['department']}\n年份：{p['year']}\n要点：\n"
                for kp in p["key_points"]:
                    text += f"- {kp}\n"
                text += f"标签：{'、'.join(p['tags'])}"
                from langchain_core.documents import Document
                docs.append(Document(
                    page_content=text,
                    metadata={"source": p["title"], "type": "policy", "level": p["level"]}
                ))

        # 2. 加载企业JSON
        enterprise_path = Path(__file__).resolve().parent.parent / "data/enterprises/enterprises.json"
        if enterprise_path.exists():
            with open(enterprise_path) as f:
                enterprises = json.load(f)["enterprises"]
            from langchain_core.documents import Document
            for e in enterprises:
                text = f"企业：{e['name']}\n行业：{e['industry']}\n城市：{e['city']}\n"
                text += f"注册资本：{e['registered_capital']}\n员工：{e['employees']}\n"
                text += f"营收：{e['revenue_2024']}\n主营：{e['main_business']}\n"
                text += f"投资动态：{e['investment_history']}\n关注点：{'、'.join(e['key_concerns'])}\n"
                text += f"扩张计划：{e['expansion_plan']}"
                docs.append(Document(
                    page_content=text,
                    metadata={"source": e["name"], "type": "enterprise"}
                ))

        # 3. 加载知识库Markdown
        if KNOWLEDGE_DIR.exists():
            for md_file in KNOWLEDGE_DIR.glob("*.md"):
                loader = TextLoader(str(md_file), encoding="utf-8")
                docs.extend(loader.load())

        return docs

    def _build_index(self):
        """构建/重建向量索引"""
        docs = self._load_documents()
        if not docs:
            print("[RAG] 没有文档需要索引")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        print(f"[RAG] 文档分块：{len(docs)} 篇 → {len(chunks)} 个chunk")

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        print(f"[RAG] 向量索引构建完成，持久化到 {CHROMA_DIR}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """检索与查询最相关的文档片段"""
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search(query, k=top_k)
        return [{"content": r.page_content, "metadata": r.metadata} for r in results]

    def query_with_context(self, question: str, top_k: int = 5) -> str:
        """检索并组装上下文，供LLM生成回答"""
        docs = self.search(question, top_k)
        if not docs:
            return "未找到相关信息。"

        context = "\n\n---\n\n".join([d["content"] for d in docs])
        sources = list(set([d["metadata"].get("source", "") for d in docs if d["metadata"].get("source")]))
        return context, sources
