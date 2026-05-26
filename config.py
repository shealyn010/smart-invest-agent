"""智招Agents - 配置"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
ENTERPRISE_DIR = DATA_DIR / "enterprises"
POLICY_DIR = DATA_DIR / "policies"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# LLM - 使用 DeepSeek API（兼容 OpenAI 格式）
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Embedding - 使用本地 BGE 模型（无需网络，数据不出本地）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", str(PROJECT_ROOT.parent / "hidden-vector-test/models/bge-small-zh"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
