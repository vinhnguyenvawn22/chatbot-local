"""Luồng RAG cơ bản: tìm tài liệu rồi đưa context cho Qwen trả lời."""

import gc
import re

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langsmith import traceable
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    MAX_SUBQUESTIONS,
    MAX_TOTAL_DOCUMENTS,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    TOP_K,
    TOP_K_PER_SUBQUESTION,
    get_embedding_device,
    get_embedding_model_kwargs,
)


QUESTION_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """Bạn có nhiệm vụ dựa vào lịch sử hội thoại để hiểu câu hỏi mới, sau đó kiểm tra câu hỏi có chứa nhiều ý độc lập hay không.

Quy tắc:
- Nếu câu hỏi mới dùng các từ như "nó", "đó", "còn", "thế thì" hoặc thiếu chủ ngữ, hãy dựa vào lịch sử để viết lại thành câu hỏi đầy đủ nghĩa.
- Chỉ tách khi người dùng thực sự hỏi từ hai nội dung khác nhau trở lên.
- Mỗi câu hỏi con phải đầy đủ nghĩa và giữ nguyên ý định của người dùng.
- Không tự thêm thông tin mới.
- Nếu chỉ có một ý, vẫn trả về câu hỏi đã được viết đầy đủ nghĩa trong danh sách.
- Tối đa {max_subquestions} câu hỏi con.

Chỉ trả về JSON theo đúng cấu trúc:
{{"is_multi_intent": true, "questions": ["Câu hỏi 1", "Câu hỏi 2"]}}

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÂU HỎI MỚI CỦA NGƯỜI DÙNG:
{question}"""
)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """Bạn là trợ lý hỏi đáp về các quy định và thông tin của trường.

Chỉ sử dụng nội dung trong phần TÀI LIỆU để trả lời.
LỊCH SỬ HỘI THOẠI chỉ dùng để hiểu câu hỏi nối tiếp; mọi thông tin dùng để trả lời vẫn phải có trong TÀI LIỆU.
Nếu tài liệu không đủ thông tin, hãy nói rõ rằng bạn chưa tìm thấy thông tin.
Không được tự đoán hoặc tạo thêm quy định.
Khi trả lời phải trả lời thông tin thật đầy đủ không được bỏ sót nếu có trong tài liệu.
Khi dùng thông tin từ một đoạn, hãy trích dẫn số nguồn như [1], [2].
Nếu câu hỏi có nhiều ý, phải trả lời lần lượt và không được bỏ sót ý nào.
Với câu hỏi nhiều ý, hãy dùng tiêu đề hoặc danh sách đánh số để câu trả lời dễ đọc.
Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu.

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÁC Ý ĐÃ NHẬN DIỆN:
{subquestions}

TÀI LIỆU:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:"""
)


def format_chat_history(history: list[dict], max_messages: int = 6) -> str:
    """Đổi các tin nhắn gần nhất thành văn bản để Qwen hiểu câu hỏi nối tiếp."""
    lines: list[str] = []

    for message in history[-max_messages:]:
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        role = message.get("role")
        speaker = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{speaker}: {content}")

    return "\n".join(lines) if lines else "Chưa có lịch sử hội thoại."


def tokenize_for_bm25(text: str) -> list[str]:
    """Đổi văn bản thành các từ viết thường để BM25 so khớp từ khóa."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def calculate_generation_metrics(response: object) -> dict:
    """Tính tốc độ sinh token từ metadata native mà Ollama trả về.

    Ollama trả các trường duration theo nanosecond. Hàm này không tự đếm từ
    trong câu trả lời, vì token của mô hình không luôn tương đương với từ.
    """
    metadata = getattr(response, "response_metadata", {}) or {}

    def metadata_int(name: str) -> int:
        try:
            return int(metadata.get(name) or 0)
        except (TypeError, ValueError):
            return 0

    eval_count = metadata_int("eval_count")
    eval_duration_ns = metadata_int("eval_duration")
    prompt_eval_count = metadata_int("prompt_eval_count")
    prompt_eval_duration_ns = metadata_int("prompt_eval_duration")

    eval_seconds = eval_duration_ns / 1_000_000_000
    prompt_eval_seconds = prompt_eval_duration_ns / 1_000_000_000

    return {
        "generated_tokens": eval_count,
        "generation_seconds": round(eval_seconds, 3),
        "generation_tokens_per_second": (
            round(eval_count / eval_seconds, 2) if eval_seconds > 0 else None
        ),
        "prompt_tokens": prompt_eval_count,
        "prompt_seconds": round(prompt_eval_seconds, 3),
        "prompt_tokens_per_second": (
            round(prompt_eval_count / prompt_eval_seconds, 2)
            if prompt_eval_seconds > 0
            else None
        ),
        "model_load_seconds": round(metadata_int("load_duration") / 1_000_000_000, 3),
        "total_model_seconds": round(metadata_int("total_duration") / 1_000_000_000, 3),
    }


def document_key(document: Document) -> tuple:
    """Tạo khóa để nhận biết hai Document có phải cùng một chunk hay không."""
    return (
        document.metadata.get("relative_path"),
        document.metadata.get("page"),
        document.metadata.get("sheet_name"),
        document.page_content,
    )


@traceable(name="Gộp kết quả vector và BM25", run_type="chain")
def fuse_search_results(
    vector_documents: list[Document],
    bm25_documents: list[Document],
    rrf_k: int = 60,
) -> list[Document]:
    """Gộp thứ hạng vector và BM25 bằng Reciprocal Rank Fusion (RRF)."""
    scores: dict[tuple, float] = {}
    documents_by_key: dict[tuple, Document] = {}

    for result_list in (vector_documents, bm25_documents):
        for rank, document in enumerate(result_list, start=1):
            key = document_key(document)
            documents_by_key[key] = document
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [documents_by_key[key] for key in ranked_keys]


def source_name(document: Document) -> str:
    """Tạo tên nguồn dễ đọc từ metadata."""
    file_name = str(document.metadata.get("file_name", "Không rõ tên file"))
    page = document.metadata.get("page")
    sheet = document.metadata.get("sheet_name")

    details: list[str] = []
    if page is not None:
        # PyPDF đếm trang từ 0, còn người đọc đếm từ 1.
        details.append(f"trang {int(page) + 1}")
    if sheet:
        details.append(f"sheet {sheet}")

    return f"{file_name} ({', '.join(details)})" if details else file_name


def format_context(groups: list[dict]) -> str:
    """Ghép tài liệu theo từng ý và đánh số nguồn liên tục."""
    sections: list[str] = []
    source_index = 1

    for group_index, group in enumerate(groups, start=1):
        sections.append(f"Ý {group_index}: {group['question']}")

        for document in group["documents"]:
            sections.append(
                f"[{source_index}] Nguồn: {source_name(document)}\n"
                f"{document.page_content}"
            )
            source_index += 1

    return "\n\n---\n\n".join(sections)


def flatten_documents(groups: list[dict]) -> list[Document]:
    documents: list[Document] = []

    for group in groups:
        for document in group["documents"]:
            documents.append(document)

    return documents


def build_sources(documents: list[Document]) -> list[dict]:
    """Tạo danh sách nguồn đúng thứ tự đánh số trong context."""
    sources: list[dict] = []

    for citation, document in enumerate(documents, start=1):
        metadata = document.metadata
        sources.append(
            {
                "citation": citation,
                "file_name": metadata.get("file_name", "Không rõ tên file"),
                "relative_path": metadata.get("relative_path", ""),
                "department": metadata.get("department", ""),
                "page": (
                    int(metadata["page"]) + 1
                    if metadata.get("page") is not None
                    else None
                ),
                "sheet_name": metadata.get("sheet_name"),
            }
        )

    return sources


class SchoolRAG:
    """Một class nhỏ gom retriever và Qwen để giao diện dễ sử dụng."""

    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection_names = {
            getattr(collection, "name", collection)
            for collection in client.list_collections()
        }

        if COLLECTION_NAME not in collection_names:
            raise RuntimeError(
                "Chưa có vector database. Hãy chạy `python ingest.py` trước."
            )

        # Vector store không giữ BGE-M3 trong RAM. Model embedding chỉ được
        # nạp tạm thời lúc truy xuất rồi giải phóng trước khi gọi Qwen.
        self.vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=None,
        )

        # BM25 cần toàn bộ nội dung chunk để tạo chỉ mục từ khóa trong RAM.
        collection = client.get_collection(name=COLLECTION_NAME)
        stored_data = collection.get(include=["documents", "metadatas"])
        stored_contents = stored_data.get("documents") or []
        stored_metadatas = stored_data.get("metadatas") or []

        self.bm25_documents = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(stored_contents, stored_metadatas)
            if content
        ]
        tokenized_corpus = [
            tokenize_for_bm25(document.page_content)
            for document in self.bm25_documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        self.answer_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        self.question_analysis_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        self.question_analysis_chain = (
            QUESTION_ANALYSIS_PROMPT
            | self.question_analysis_llm
            | JsonOutputParser()
        )
        # Giữ lại AIMessage thay vì chuyển ngay thành str để lấy được metadata
        # Ollama, ví dụ eval_count và eval_duration.
        self.answer_chain = RAG_PROMPT | self.answer_llm

    @traceable(name="Phân tích câu hỏi", run_type="chain")
    def analyze_question(self, question: str, chat_history: str) -> list[str]:
        """Dùng lịch sử để hiểu câu hỏi nối tiếp rồi tách các ý nếu cần."""
        try:
            result = self.question_analysis_chain.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "max_subquestions": MAX_SUBQUESTIONS,
                }
            )
            questions = result.get("questions", [])

            if not isinstance(questions, list):
                return [question]

            cleaned_questions = [
                str(item).strip()
                for item in questions
                if str(item).strip()
            ]
            return cleaned_questions[:MAX_SUBQUESTIONS] or [question]
        except Exception:
            # Nếu Qwen trả JSON sai, chatbot vẫn hoạt động như câu hỏi một ý.
            return [question]

    @traceable(name="Tìm kiếm BM25", run_type="retriever")
    def search_with_bm25(self, question: str, k: int) -> list[Document]:
        """Tìm tối đa k chunk có nhiều từ khóa liên quan nhất."""
        if self.bm25 is None:
            return []

        query_tokens = tokenize_for_bm25(question)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            self.bm25_documents[index]
            for index in ranked_indices[:k]
            if scores[index] > 0
        ]

    @traceable(name="Tìm kiếm hybrid", run_type="retriever")
    def retrieve_documents(self, questions: list[str]) -> list[dict]:
        """Tìm bằng cả vector và BM25 rồi gộp kết quả cho từng ý."""
        device = get_embedding_device()
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=get_embedding_model_kwargs(device),
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": EMBEDDING_BATCH_SIZE,
            },
        )

        question_vectors = embedding.embed_documents(questions)
        candidate_groups: list[dict] = []

        for question, question_vector in zip(questions, question_vectors):
            vector_documents = self.vector_store.similarity_search_by_vector(
                embedding=question_vector,
                k=TOP_K,
            )
            bm25_documents = self.search_with_bm25(
                question=question,
                k=TOP_K,
            )
            documents = fuse_search_results(
                vector_documents=vector_documents,
                bm25_documents=bm25_documents,
            )
            candidate_groups.append(
                {"question": question, "documents": documents}
            )

        # Đây là bước quan trọng trên máy 16 GB RAM / RTX 3050 4 GB.
        # Không để BGE-M3 và Qwen cùng nằm trong bộ nhớ.
        del embedding
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

        # Chọn tài liệu luân phiên giữa các ý để một ý không chiếm hết context.
        selected_groups = [
            {"question": group["question"], "documents": []}
            for group in candidate_groups
        ]
        seen_documents: set[tuple] = set()
        total_documents = 0

        for rank in range(TOP_K_PER_SUBQUESTION):
            for group_index, group in enumerate(candidate_groups):
                if total_documents >= MAX_TOTAL_DOCUMENTS:
                    break
                if rank >= len(group["documents"]):
                    continue

                document = group["documents"][rank]
                key = document_key(document)
                if key in seen_documents:
                    continue

                seen_documents.add(key)
                selected_groups[group_index]["documents"].append(document)
                total_documents += 1

        return selected_groups

    @traceable(name="School RAG - một lượt hỏi", run_type="chain")
    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        """Tìm tài liệu rồi hỏi Qwen và trả câu trả lời kèm nguồn."""
        chat_history = format_chat_history(history or [])
        subquestions = self.analyze_question(question, chat_history)
        groups = self.retrieve_documents(subquestions)
        documents = flatten_documents(groups)

        if not documents:
            return {
                "answer": "Tôi chưa tìm thấy tài liệu phù hợp để trả lời câu hỏi này.",
                "sources": [],
                "subquestions": subquestions,
                "generation_metrics": None,
            }

        subquestions_text = "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(subquestions, start=1)
        )
        answer_response = self.answer_chain.invoke(
            {
                "context": format_context(groups),
                "question": question,
                "chat_history": chat_history,
                "subquestions": subquestions_text,
            }
        )
        answer = str(answer_response.content)

        return {
            "answer": answer,
            "sources": build_sources(documents),
            "subquestions": subquestions,
            "generation_metrics": calculate_generation_metrics(answer_response),
        }
