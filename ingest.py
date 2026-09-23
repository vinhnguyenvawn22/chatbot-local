"""Đọc tài liệu trong thư mục document và đưa vào Chroma.

Chạy file này khi thêm, xóa hoặc sửa tài liệu:
    python ingest.py
"""

import argparse
import hashlib
import sys
from pathlib import Path

import chromadb
import openpyxl
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCUMENT_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    get_embedding_device,
    get_embedding_model_kwargs,
)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt"}

# PowerShell/Terminal trên một số máy Windows mặc định dùng cp1252.
# Chuyển output sang UTF-8 để in tên file và thông báo tiếng Việt đúng.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def add_file_metadata(documents: list[Document], file_path: Path) -> list[Document]:
    """Gắn thông tin nguồn vào tài liệu để chatbot có thể trích dẫn."""
    relative_path = file_path.relative_to(DOCUMENT_DIR)
    department = relative_path.parts[0] if len(relative_path.parts) > 1 else "Khác"

    for document in documents:
        document.metadata.update(
            {
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "relative_path": str(relative_path),
                "department": department,
            }
        )

    return documents


def load_excel(file_path: Path) -> list[Document]:
    """Mỗi sheet Excel được chuyển thành một Document dạng văn bản."""
    workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    documents: list[Document] = []

    try:
        for sheet in workbook.worksheets:
            lines: list[str] = []

            for row in sheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                line = " | ".join(values).strip(" |")
                if line:
                    lines.append(line)

            if lines:
                documents.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={"sheet_name": sheet.title},
                    )
                )
    finally:
        workbook.close()

    return documents


def load_one_file(file_path: Path) -> list[Document]:
    """Chọn loader dựa trên phần mở rộng của file."""
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        documents = PyPDFLoader(str(file_path)).load()
    elif extension == ".docx":
        documents = Docx2txtLoader(str(file_path)).load()
    elif extension == ".xlsx":
        documents = load_excel(file_path)
    elif extension == ".txt":
        documents = TextLoader(str(file_path), encoding="utf-8").load()
    else:
        return []

    # Bỏ các trang/file không lấy được chữ, thường gặp ở PDF scan ảnh.
    documents = [doc for doc in documents if doc.page_content.strip()]
    return add_file_metadata(documents, file_path)


def load_all_documents() -> tuple[list[Document], list[str], int]:
    """Đọc toàn bộ file được hỗ trợ và trả về cả danh sách cảnh báo."""
    documents: list[Document] = []
    warnings: list[str] = []
    loaded_file_count = 0

    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục tài liệu: {DOCUMENT_DIR}")

    for file_path in sorted(DOCUMENT_DIR.rglob("*")):
        if not file_path.is_file() or file_path.name.startswith("~$"):
            continue

        extension = file_path.suffix.lower()

        if extension == ".doc":
            warnings.append(f"Bỏ qua file Word cũ, hãy đổi sang .docx: {file_path}")
            continue

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        try:
            loaded = load_one_file(file_path)
            if loaded:
                documents.extend(loaded)
                loaded_file_count += 1
            else:
                warnings.append(f"Không lấy được văn bản, có thể cần OCR: {file_path}")
        except Exception as error:
            warnings.append(f"Lỗi khi đọc {file_path}: {error}")

    return documents, warnings, loaded_file_count


def split_documents(documents: list[Document]) -> list[Document]:
    """Chia tài liệu thành các đoạn vừa đủ để tìm kiếm."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def make_chunk_ids(chunks: list[Document]) -> list[str]:
    """Tạo ID ổn định từ nguồn, trang, sheet và nội dung của mỗi đoạn."""
    ids: list[str] = []

    for chunk_index, chunk in enumerate(chunks):
        raw_id = "|".join(
            [
                str(chunk_index),
                str(chunk.metadata.get("relative_path", "")),
                str(chunk.metadata.get("page", "")),
                str(chunk.metadata.get("sheet_name", "")),
                chunk.page_content,
            ]
        )
        ids.append(hashlib.sha256(raw_id.encode("utf-8")).hexdigest())

    return ids


def rebuild_vector_store(chunks: list[Document]) -> None:
    """Xóa collection cũ và tạo lại index từ tài liệu hiện tại."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection_names = {
        getattr(collection, "name", collection)
        for collection in client.list_collections()
    }
    if COLLECTION_NAME in collection_names:
        client.delete_collection(name=COLLECTION_NAME)

    device = get_embedding_device()
    print(f"Thiết bị embedding: {device}")

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=get_embedding_model_kwargs(device),
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
    )

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Thêm theo lô để không dùng quá nhiều RAM khi có nhiều tài liệu.
    batch_size = 100
    chunk_ids = make_chunk_ids(chunks)

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        vector_store.add_documents(
            documents=chunks[start:end],
            ids=chunk_ids[start:end],
        )
        print(f"Đã embedding {min(end, len(chunks))}/{len(chunks)} đoạn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nạp tài liệu trường vào Chroma")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ đọc và chia tài liệu, không tạo embedding",
    )
    args = parser.parse_args()

    print(f"Thư mục tài liệu: {DOCUMENT_DIR}")
    documents, warnings, file_count = load_all_documents()
    print(f"Đã đọc {file_count} file thành {len(documents)} Document")

    chunks = split_documents(documents)
    print(f"Đã chia thành {len(chunks)} đoạn")

    if warnings:
        print(f"\nCó {len(warnings)} cảnh báo:")
        for warning in warnings:
            print(f"- {warning}")

    if not chunks:
        raise RuntimeError("Không có nội dung nào để đưa vào vector database")

    if args.dry_run:
        print("\nDry run hoàn tất. Chưa thay đổi vector database.")
        return

    print("\nBắt đầu tạo embedding local bằng BGE-M3...")
    rebuild_vector_store(chunks)
    print(f"\nHoàn tất. Vector database được lưu tại: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
