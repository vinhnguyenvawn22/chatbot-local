# Chatbot tài liệu trường — phiên bản dễ học

Project này minh họa luồng RAG cơ bản:

1. `ingest.py` đọc PDF, DOCX, XLSX và TXT.
2. Tài liệu được chia thành các đoạn nhỏ.
3. BGE-M3 tạo vector và lưu vào Chroma.
4. `rag.py` tìm các đoạn liên quan bằng cả BGE-M3 và BM25, sau đó gộp thứ hạng.
5. Qwen đọc các đoạn đó và tạo câu trả lời.
6. `app.py` hiển thị giao diện chat và nguồn tài liệu.

Nếu câu hỏi có nhiều ý, Qwen sẽ tách thành các câu hỏi con. BGE-M3 tìm theo ngữ nghĩa, BM25 tìm theo từ khóa, rồi hệ thống gộp hai kết quả cho từng ý. Sau đó Qwen tổng hợp một câu trả lời đầy đủ. Giao diện hiển thị các ý đã nhận diện để người dùng kiểm tra.

Project không dùng Tavily và mặc định không bật LangSmith tracing. Qwen, embedding và vector database đều chạy local.

Trên máy có ít RAM/VRAM, BGE-M3 chỉ được nạp trong lúc tìm tài liệu rồi được giải phóng trước khi Qwen chạy. Qwen cũng được gỡ khỏi bộ nhớ sau mỗi câu trả lời. Cách này chậm hơn một chút nhưng tránh để hai model lớn chiếm bộ nhớ cùng lúc.

## 1. Chuẩn bị môi trường

Mở PowerShell tại `C:\nam4\rag_new`:

```powershell
.\.venv\Scripts\Activate.ps1
cd .\school_chatbot
python -m pip install -r requirements.txt
```

Đảm bảo Ollama đang chạy và đã có model:

```powershell
ollama list
ollama run qwen3:4b-instruct
```

Nhấn `Ctrl+C` để thoát phần chat thử của Ollama.

## 2. Nạp tài liệu

Thử đọc tài liệu mà chưa tạo embedding:

```powershell
python ingest.py --dry-run
```

Nếu kết quả ổn, tạo vector database:

```powershell
python ingest.py
```

Lần đầu có thể mất thời gian vì BGE-M3 phải xử lý toàn bộ tài liệu. Khi thêm, xóa hoặc sửa tài liệu, chạy lại `python ingest.py`.

Các file `.doc` kiểu cũ sẽ bị bỏ qua. Hãy mở chúng bằng Microsoft Word và lưu lại thành `.docx`. PDF scan chỉ chứa ảnh cần được OCR trước.

## 3. Chạy chatbot

```powershell
streamlit run app.py
```

Trình duyệt sẽ mở giao diện tại `http://localhost:8501`.

## 4. Xem luồng chạy trên LangSmith (tùy chọn)

Qwen, BGE-M3 và Chroma vẫn chạy local. Khi bật tracing, câu hỏi, lịch sử, các chunk truy xuất và câu trả lời có thể được gửi lên LangSmith để hiển thị luồng chạy.

1. Đăng nhập tại `https://smith.langchain.com` và tạo một API key.
2. Tạo file `.env` từ `.env.example` nếu chưa có.
3. Sửa ba dòng sau trong `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...api-key-cua-ban...
LANGSMITH_PROJECT=school-chatbot-local
```

4. Khởi động lại Streamlit và gửi một câu hỏi.
5. Mở LangSmith, vào `Tracing` rồi chọn project `school-chatbot-local`.

Mỗi lượt hỏi sẽ có trace cha `School RAG - một lượt hỏi`. Bên trong có các bước phân tích câu hỏi, tìm kiếm BM25, gộp kết quả vector với BM25 và gọi Qwen. Muốn tắt việc gửi trace, đặt lại:

```env
LANGSMITH_TRACING=false
```

## Các file quan trọng

- `config.py`: tên model, đường dẫn và các con số cấu hình.
- `ingest.py`: đọc, chia và embedding tài liệu.
- `rag.py`: tìm tài liệu và hỏi Qwen.
- `app.py`: giao diện Streamlit.

## Thay đổi cấu hình

Sao chép `.env.example` thành `.env`, sau đó sửa giá trị mong muốn. Mặc định `auto` sẽ ưu tiên NVIDIA GPU khi PyTorch CUDA hoạt động, nếu không sẽ tự dùng CPU:

```env
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=2
```

## Giới hạn của phiên bản đầu

- Chưa có đăng nhập và phân quyền.
- Chưa có web search.
- Chưa có lịch sử hội thoại dài hạn.
- Chưa OCR PDF scan.
- Chưa có bộ đánh giá tự động.

Những phần này nên được thêm sau khi đã hiểu và kiểm thử ổn định luồng RAG cơ bản.
# chatbot-local
