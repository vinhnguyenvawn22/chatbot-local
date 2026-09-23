"""Cấu hình dùng chung cho project chatbot."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def resolve_path(value: str | None, default: Path) -> Path:
    """Đổi một cấu hình đường dẫn thành đường dẫn tuyệt đối."""
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


# Mặc định dùng thư mục document đang có bên cạnh project.
DOCUMENT_DIR = resolve_path(
    os.getenv("DOCUMENT_DIR"),
    PROJECT_DIR.parent / "document",
)

# Chroma sẽ lưu vector tại đây để không phải embedding lại mỗi lần chạy app.
CHROMA_DIR = resolve_path(
    os.getenv("CHROMA_DIR"),
    PROJECT_DIR / "storage" / "chroma",
)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "school_documents")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "2"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
# Mỗi phương pháp (vector và BM25) lấy TOP_K ứng viên trước khi gộp hạng.
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_SUBQUESTIONS = int(os.getenv("MAX_SUBQUESTIONS", "4"))
# Sau khi gộp, giữ tối đa số chunk này cho mỗi câu hỏi con.
TOP_K_PER_SUBQUESTION = int(os.getenv("TOP_K_PER_SUBQUESTION", "4"))
MAX_TOTAL_DOCUMENTS = int(os.getenv("MAX_TOTAL_DOCUMENTS", "6"))


def get_embedding_device() -> str:
    """Ưu tiên NVIDIA GPU; tự quay về CPU nếu CUDA không dùng được."""
    configured_device = EMBEDDING_DEVICE.lower()
    if configured_device != "auto":
        return configured_device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embedding_model_kwargs(device: str) -> dict:
    """Dùng float16 trên GPU để phù hợp card 4 GB; CPU dùng float32."""
    kwargs: dict = {"device": device}

    if device == "cuda":
        import torch

        kwargs["model_kwargs"] = {"dtype": torch.float16}

    return kwargs
