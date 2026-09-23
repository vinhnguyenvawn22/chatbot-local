"""Giao diện chatbot đơn giản bằng Streamlit."""

import streamlit as st

from rag import SchoolRAG


st.set_page_config(
    page_title="Chatbot tài liệu trường",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Chatbot tài liệu trường")
st.caption("Qwen và BGE-M3 chạy local • Câu trả lời có kèm nguồn tài liệu")


@st.cache_resource(show_spinner="Đang kết nối Chroma...")
def load_rag() -> SchoolRAG:
    """Chỉ tải model và vector database một lần."""
    return SchoolRAG()


with st.sidebar:
    st.header("Thông tin")
    st.write(
        "Chatbot chỉ trả lời dựa trên tài liệu đã được nạp trong thư mục `document`."
    )
    st.code("python ingest.py", language="powershell")
    st.caption("Chạy lại lệnh trên khi tài liệu thay đổi.")

    if st.button("Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()


try:
    rag = load_rag()
except Exception as error:
    st.error(str(error))
    st.info("Mở Terminal, vào thư mục school_chatbot và chạy `python ingest.py`.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    """Hiển thị danh sách tài liệu đã được dùng để tạo câu trả lời."""
    if not sources:
        return

    with st.expander("Nguồn tài liệu"):
        for index, source in enumerate(sources, start=1):
            citation = source.get("citation", index)
            details = []
            if source.get("page"):
                details.append(f"trang {source['page']}")
            if source.get("sheet_name"):
                details.append(f"sheet {source['sheet_name']}")

            suffix = f" — {', '.join(details)}" if details else ""
            st.markdown(
                f"**[{citation}] {source['file_name']}**{suffix}  \n"
                f"Phòng/nhóm: `{source.get('department', '')}`"
            )


def show_subquestions(subquestions: list[str]) -> None:
    """Cho người dùng xem chatbot đã tách câu hỏi thành những ý nào."""
    if len(subquestions) <= 1:
        return

    with st.expander(f"Chatbot đã nhận diện {len(subquestions)} ý"):
        for index, subquestion in enumerate(subquestions, start=1):
            st.markdown(f"{index}. {subquestion}")


def show_generation_metrics(metrics: dict | None) -> None:
    """Hiển thị tốc độ sinh token nếu Ollama đã trả metric."""
    if not metrics:
        st.caption("Không có số đo generation vì hệ thống không gọi model để tạo câu trả lời.")
        return

    generation_rate = metrics.get("generation_tokens_per_second")
    if generation_rate is None:
        st.caption("Ollama không trả về số liệu tốc độ sinh token cho lượt này.")
        return

    st.caption(
        f"⚡ Tốc độ sinh: {generation_rate} token/giây "
        f"• {metrics.get('generated_tokens', 0)} token trong "
        f"{metrics.get('generation_seconds', 0)} giây"
    )

    with st.expander("Hiệu năng tạo câu trả lời"):
        st.caption(
            "Token là đơn vị văn bản của mô hình, không hoàn toàn bằng một từ."
        )
        st.caption(
            f"Nạp model: {metrics.get('model_load_seconds', 0)} giây • "
            f"Tổng thời gian gọi model: {metrics.get('total_model_seconds', 0)} giây"
        )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_subquestions(message.get("subquestions", []))
            show_sources(message.get("sources", []))
            show_generation_metrics(message.get("generation_metrics"))


question = st.chat_input("Nhập câu hỏi về tài liệu của trường...")

if question:
    # Lưu lại hội thoại cũ trước khi thêm câu hỏi hiện tại. Lịch sử này giúp
    # RAG hiểu các câu nối tiếp như "còn lớp 11 thì sao?" hoặc "nó áp dụng khi nào?".
    conversation_history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang phân tích câu hỏi, tìm tài liệu và tạo câu trả lời..."):
                result = rag.ask(question, history=conversation_history)

            st.markdown(result["answer"])
            show_subquestions(result["subquestions"])
            show_sources(result["sources"])
            show_generation_metrics(result.get("generation_metrics"))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "subquestions": result["subquestions"],
                    "generation_metrics": result.get("generation_metrics"),
                }
            )
        except Exception as error:
            error_message = f"Không thể tạo câu trả lời: {error}"
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "subquestions": [],
                    "generation_metrics": None,
                }
            )
