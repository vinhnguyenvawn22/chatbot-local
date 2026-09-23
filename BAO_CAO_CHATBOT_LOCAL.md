# BÁO CÁO XÂY DỰNG VÀ ĐÁNH GIÁ CHATBOT TRA CỨU TÀI LIỆU TRƯỜNG

## 1. Thông tin chung

- **Tên đề tài:** Xây dựng chatbot hỏi đáp tài liệu trường sử dụng mô hình ngôn ngữ chạy cục bộ
- **Sinh viên thực hiện:** ........................................................
- **Người hướng dẫn:** ........................................................
- **Thời gian thực hiện:** ........................................................

## 2. Tóm tắt đề tài

Đề tài xây dựng một chatbot có khả năng trả lời câu hỏi dựa trên các tài liệu nội bộ của trường như PDF, Word, Excel và tệp văn bản. Hệ thống sử dụng kỹ thuật **RAG** (Retrieval-Augmented Generation), nghĩa là trước khi trả lời, chatbot phải tìm những đoạn tài liệu liên quan rồi mới đưa chúng cho mô hình ngôn ngữ.

Điểm chính của đề tài là các thành phần quan trọng đều có thể chạy trên máy cá nhân:

- **Qwen3 4B Instruct** đọc tài liệu tìm được và tạo câu trả lời.
- **BGE-M3** tìm kiếm theo ý nghĩa của câu hỏi.
- **BM25** tìm kiếm theo từ khóa xuất hiện trong câu hỏi.
- **Chroma** lưu trữ các vector của tài liệu.
- **Streamlit** cung cấp giao diện trò chuyện.

Mục tiêu tiếp theo là so sánh chatbot local với chatbot sử dụng Gemini trên cùng một tập tài liệu và cùng một bộ câu hỏi. Việc so sánh cần dựa trên kết quả đo thực tế, không chỉ dựa trên cảm nhận.

## 3. Lý do chọn đề tài

Các tài liệu của trường thường nằm ở nhiều tệp khác nhau, có nội dung dài và khó tra cứu thủ công. Người dùng có thể phải mở nhiều văn bản mới tìm được câu trả lời cần thiết. Chatbot giúp người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên và nhận câu trả lời kèm nguồn.

Việc thử nghiệm mô hình local có các ý nghĩa sau:

- Chủ động kiểm soát mô hình và dữ liệu.
- Có khả năng hoạt động trong mạng nội bộ hoặc khi không có Internet.
- Không phụ thuộc hoàn toàn vào một dịch vụ AI bên ngoài.
- Có thể giảm chi phí API khi số lượt hỏi lớn.
- Kiểm tra xem một mô hình nhỏ chạy local có đáp ứng được bài toán hẹp hay không.

Tuy nhiên, mô hình local không mặc nhiên tốt hơn Gemini. Chất lượng cuối cùng còn phụ thuộc vào tài liệu, cách chia đoạn, khả năng tìm kiếm, prompt, cấu hình máy và tiêu chí đánh giá.

## 4. Mục tiêu của hệ thống

Hệ thống được xây dựng với các mục tiêu:

1. Đọc được tài liệu định dạng PDF, DOCX, XLSX và TXT.
2. Tìm đúng các đoạn tài liệu liên quan đến câu hỏi.
3. Chỉ trả lời dựa trên tài liệu đã nạp, hạn chế bịa thông tin.
4. Hiển thị tên tệp, số trang hoặc tên sheet để người dùng kiểm tra.
5. Hiểu câu hỏi có nhiều ý và câu hỏi nối tiếp trong hội thoại.
6. Chạy được trên máy có 16 GB RAM và GPU RTX 3050 4 GB.
7. Có thể theo dõi từng bước xử lý bằng LangSmith khi cần kiểm tra.

## 5. Công nghệ và mô hình sử dụng

### 5.1. Qwen3 4B Instruct

Hệ thống đang dùng model `qwen3:4b-instruct`, chạy thông qua Ollama.

Qwen được dùng cho hai nhiệm vụ:

1. **Phân tích câu hỏi:** dựa vào lịch sử hội thoại, viết lại câu hỏi nối tiếp cho đầy đủ nghĩa và tách câu hỏi nhiều ý thành các câu hỏi con.
2. **Tạo câu trả lời:** đọc các đoạn tài liệu được tìm thấy, tổng hợp câu trả lời bằng tiếng Việt và chèn ký hiệu nguồn như `[1]`, `[2]`.

Trong mã nguồn, `temperature=0` được sử dụng để câu trả lời ổn định hơn và giảm việc sáng tạo ngoài nội dung tài liệu.

### 5.2. BGE-M3

BGE-M3 là model embedding. Model này không trực tiếp viết câu trả lời mà chuyển văn bản thành một dãy số gọi là **vector**.

Các câu có ý nghĩa gần nhau thường có vector gần nhau. Vì vậy, hệ thống có thể tìm thấy đoạn liên quan ngay cả khi câu hỏi và tài liệu không dùng chính xác cùng một từ.

Ví dụ:

- Câu hỏi: “Sinh viên được nhận hỗ trợ tài chính nào?”
- Tài liệu: “Các loại học bổng dành cho người học...”

Hai câu không hoàn toàn giống từ khóa nhưng có thể gần nhau về ý nghĩa. Đây là phần mà tìm kiếm vector hỗ trợ tốt.

### 5.3. BM25

BM25 là phương pháp tìm kiếm dựa trên từ khóa. Nó ưu tiên những đoạn chứa các từ quan trọng xuất hiện trong câu hỏi.

Ví dụ, nếu người dùng hỏi về `Điều 3`, `học bổng khuyến khích học tập` hoặc một mã quyết định cụ thể, BM25 có thể tìm rất tốt vì những cụm từ này xuất hiện trực tiếp trong tài liệu.

### 5.4. Tìm kiếm hybrid và RRF

Hệ thống kết hợp BGE-M3 và BM25, được gọi là **hybrid search**:

- BGE-M3 mạnh về tìm kiếm theo ngữ nghĩa.
- BM25 mạnh về tìm kiếm từ khóa chính xác.

Sau khi hai phương pháp trả về hai danh sách kết quả, hệ thống dùng **RRF** (Reciprocal Rank Fusion) để gộp thứ hạng. Một đoạn xuất hiện ở vị trí cao trong một hoặc cả hai danh sách sẽ được cộng điểm và có cơ hội được chọn cao hơn.

Ý tưởng tính điểm được biểu diễn đơn giản như sau:

```text
Điểm RRF của một đoạn = tổng của 1 / (60 + thứ hạng của đoạn)
```

RRF không cần so sánh trực tiếp điểm vector với điểm BM25 vì hai loại điểm có thang đo khác nhau.

### 5.5. Chroma

Chroma là vector database của hệ thống. Nó lưu:

- Nội dung từng chunk.
- Vector do BGE-M3 tạo ra.
- Metadata như tên tệp, đường dẫn tương đối, phòng/nhóm, trang PDF và sheet Excel.

Tên collection mặc định là `school_documents`. Đây là tên một nhóm dữ liệu bên trong Chroma, không phải tên một tệp trong thư mục.

### 5.6. Các công nghệ hỗ trợ

- **LangChain:** kết nối prompt, model, tài liệu và bộ tìm kiếm.
- **Ollama:** chạy Qwen trên máy local.
- **Streamlit:** tạo giao diện web cho chatbot.
- **LangSmith:** xem trace của luồng xử lý khi bật tracing.
- **PyPDF, Docx2txt, OpenPyXL:** đọc PDF, Word và Excel.
- **Tiktoken/RecursiveCharacterTextSplitter:** hỗ trợ chia tài liệu thành các đoạn nhỏ.

Lưu ý: khi bật LangSmith tracing, câu hỏi, chunk và kết quả có thể được gửi lên dịch vụ LangSmith để quan sát. Vì vậy, chế độ này cần được tắt nếu yêu cầu dữ liệu phải hoàn toàn nằm trong máy.

## 6. Kiến trúc tổng thể

Hệ thống có hai luồng chính: **nạp tài liệu** và **trả lời câu hỏi**.

### 6.1. Luồng nạp tài liệu

```text
PDF / DOCX / XLSX / TXT
          ↓
      Đọc nội dung
          ↓
 Gắn metadata nguồn tài liệu
          ↓
 Chia thành các chunk 700 token
   (chồng lấn 100 token)
          ↓
 BGE-M3 tạo vector embedding
          ↓
 Lưu chunk + vector + metadata vào Chroma
```

Giải thích từng bước:

1. `ingest.py` duyệt toàn bộ thư mục tài liệu.
2. Mỗi loại tệp được đọc bằng loader phù hợp.
3. Hệ thống bỏ qua nội dung rỗng và cảnh báo đối với tệp không đọc được.
4. Metadata được gắn vào tài liệu để dùng khi trích dẫn nguồn.
5. Nội dung được chia thành các chunk có kích thước mặc định 700 token, chồng lấn 100 token.
6. BGE-M3 chuyển mỗi chunk thành vector.
7. Chroma lưu dữ liệu để những lần chạy chatbot sau không phải embedding lại toàn bộ tài liệu.

Phần chồng lấn giúp giảm nguy cơ một ý quan trọng bị cắt đôi đúng tại ranh giới giữa hai chunk.

### 6.2. Luồng trả lời câu hỏi

```text
Người dùng nhập câu hỏi
          ↓
Lấy tối đa 6 tin nhắn gần nhất
          ↓
Qwen hiểu câu nối tiếp và tách câu hỏi nhiều ý
          ↓
Với mỗi câu hỏi con:
   ├── BGE-M3 tìm TOP_K=5 chunk theo ngữ nghĩa
   └── BM25 tìm TOP_K=5 chunk theo từ khóa
          ↓
RRF gộp và sắp xếp kết quả
          ↓
Loại chunk trùng và chọn luân phiên giữa các ý
          ↓
Giữ tối đa 4 chunk cho mỗi ý,
tổng cộng tối đa 6 chunk
          ↓
Ghép chunk thành context và đánh số nguồn
          ↓
Qwen tạo câu trả lời chỉ dựa trên context
          ↓
Streamlit hiển thị câu trả lời, ý đã tách và nguồn
```

> Ghi chú: dòng “tổng cộng tối đa 6 chunk” là giới hạn cho toàn bộ một lượt hỏi, không phải cho từng câu hỏi con.

## 7. Cách hệ thống xử lý một ví dụ cụ thể

Giả sử người dùng hỏi:

> “Có mấy loại học bổng và điều kiện của từng loại là gì?”

Hệ thống xử lý như sau:

1. Qwen nhận thấy câu hỏi có thể gồm hai ý:
   - Có những loại học bổng nào?
   - Điều kiện của từng loại là gì?
2. BGE-M3 tạo vector cho hai câu hỏi con.
3. Với mỗi câu hỏi con, Chroma tìm các chunk gần nhất về ý nghĩa.
4. BM25 đồng thời tìm các chunk có từ “loại”, “học bổng”, “điều kiện”.
5. RRF gộp hai danh sách kết quả.
6. Hệ thống chọn tài liệu luân phiên giữa hai ý để một ý không chiếm toàn bộ context.
7. Các chunk được đánh số `[1]`, `[2]`, ... và gửi cho Qwen.
8. Qwen tổng hợp câu trả lời, đồng thời chỉ rõ thông tin lấy từ nguồn nào.

Nếu bước tìm kiếm chỉ lấy được tài liệu về “học bổng khuyến khích học tập”, Qwen sẽ không thể biết đầy đủ các loại học bổng khác. Trường hợp này là lỗi hoặc giới hạn của bước truy xuất tài liệu, không nhất thiết là lỗi của model tạo câu trả lời.

## 8. Xử lý lịch sử hội thoại

Hệ thống lưu lịch sử trong `st.session_state.messages`. Khi có câu hỏi mới, tối đa 6 tin nhắn gần nhất được chuyển thành văn bản cho Qwen.

Ví dụ:

```text
Người dùng: Học bổng khuyến khích học tập có điều kiện gì?
Trợ lý: ...
Người dùng: Còn thời gian xét thì sao?
```

Câu cuối thiếu chủ ngữ. Qwen dựa vào lịch sử để hiểu và viết lại thành câu đầy đủ, chẳng hạn:

```text
Thời gian xét học bổng khuyến khích học tập là khi nào?
```

Lịch sử chỉ dùng để hiểu câu hỏi nối tiếp. Prompt yêu cầu mọi dữ kiện trong câu trả lời vẫn phải có trong tài liệu tìm được.

## 9. Tối ưu cho máy cá nhân

Máy thử nghiệm có 16 GB RAM và RTX 3050 4 GB VRAM, nên không phù hợp để giữ đồng thời nhiều model lớn trong bộ nhớ.

Hệ thống xử lý bằng cách:

1. Chỉ nạp BGE-M3 khi cần embedding câu hỏi.
2. Sau khi tìm tài liệu xong, xóa đối tượng embedding.
3. Thu gom bộ nhớ bằng `gc.collect()`.
4. Nếu có CUDA, giải phóng cache GPU.
5. Sau đó mới yêu cầu Qwen tạo câu trả lời.
6. `OLLAMA_KEEP_ALIVE=0` cho phép Ollama gỡ model khỏi bộ nhớ sau khi xử lý.

Cách làm này tiết kiệm bộ nhớ nhưng có thể làm phản hồi chậm hơn vì model phải được nạp lại ở các lượt hỏi sau.

## 10. Điểm mạnh và giới hạn

### 10.1. Điểm mạnh

- Model sinh câu trả lời, embedding và vector database có thể chạy local.
- Câu trả lời có trích dẫn để người dùng kiểm tra.
- Kết hợp tìm kiếm ngữ nghĩa và từ khóa.
- Có khả năng xử lý câu hỏi nhiều ý.
- Hiểu được một số câu hỏi nối tiếp dựa trên lịch sử gần nhất.
- Hỗ trợ nhiều định dạng tài liệu.
- Cấu hình đơn giản và phù hợp mục đích học tập.

### 10.2. Giới hạn

- PDF scan chỉ có hình ảnh chưa được OCR nên có thể không đọc được chữ.
- Chưa hỗ trợ tệp Word `.doc` kiểu cũ.
- Chất lượng phụ thuộc nhiều vào cách chia chunk và chất lượng tài liệu đầu vào.
- BM25 hiện tách từ bằng biểu thức chính quy, chưa dùng bộ tách từ chuyên biệt cho tiếng Việt.
- Chưa có reranker để đánh giá sâu lại các ứng viên sau RRF.
- Context hiện chỉ nhận tối đa 6 chunk nên có thể thiếu thông tin đối với câu hỏi quá rộng.
- Lịch sử chỉ lấy tối đa 6 tin nhắn gần nhất, chưa có cơ chế tóm tắt hội thoại dài.
- Chưa có OCR, đăng nhập, phân quyền và đánh giá tự động.
- Model 4B nhẹ hơn nhưng khả năng suy luận và diễn đạt có thể kém model cloud lớn trong một số câu hỏi khó.

## 11. So sánh chatbot local với Gemini

### 11.1. Câu hỏi nghiên cứu

> Với cùng tài liệu của trường và cùng bộ câu hỏi, chatbot local có cho kết quả ổn định và phù hợp hơn chatbot sử dụng Gemini hay không?

Không nên chỉ hỏi hai chatbot vài câu rồi kết luận. Để công bằng, hai hệ thống cần được thử trong điều kiện gần giống nhau.

### 11.2. Điều kiện so sánh công bằng

- Sử dụng cùng một tập tài liệu.
- Sử dụng cùng một bộ câu hỏi.
- Không cho một hệ thống xem đáp án trước.
- Prompt của hai hệ thống phải có cùng yêu cầu: chỉ dựa trên tài liệu và phải trích dẫn nguồn.
- Mỗi câu hỏi nên chạy nhiều lần nếu muốn đo độ ổn định.
- Đo thời gian từ lúc gửi câu hỏi đến lúc có câu trả lời hoàn chỉnh.
- Người chấm nên có đáp án chuẩn hoặc tài liệu tham chiếu.
- Không thay cấu hình giữa các lượt chỉ để ưu tiên một hệ thống.

### 11.3. Bộ câu hỏi kiểm thử đề xuất

Nên chuẩn bị khoảng 30–50 câu, chia thành các nhóm:

| Nhóm câu hỏi | Ví dụ | Mục đích |
|---|---|---|
| Một thông tin trực tiếp | “Mức học bổng loại A là bao nhiêu?” | Kiểm tra tra cứu cơ bản |
| Tổng hợp nhiều đoạn | “Có mấy loại học bổng và điều kiện từng loại?” | Kiểm tra độ đầy đủ |
| Nhiều ý | “Điều kiện là gì và thời gian xét khi nào?” | Kiểm tra tách ý |
| Câu hỏi nối tiếp | “Còn loại thứ hai thì sao?” | Kiểm tra lịch sử |
| Từ khóa chính xác | “Quyết định 863 quy định nội dung gì?” | Kiểm tra BM25 |
| Diễn đạt khác tài liệu | “Sinh viên khó khăn được hỗ trợ thế nào?” | Kiểm tra semantic search |
| Không có đáp án | “Trường có học bổng du học Sao Hỏa không?” | Kiểm tra chống bịa |
| Nguồn dễ nhầm | Hai văn bản có nội dung gần giống nhau | Kiểm tra trích dẫn |

### 11.4. Tiêu chí chấm điểm

Có thể chấm mỗi tiêu chí từ 0 đến 2:

| Tiêu chí | 0 điểm | 1 điểm | 2 điểm |
|---|---|---|---|
| Chính xác | Sai | Đúng một phần | Đúng hoàn toàn |
| Đầy đủ | Thiếu phần lớn | Thiếu một phần | Đủ các ý chính |
| Bám sát tài liệu | Bịa hoặc mâu thuẫn | Có chi tiết chưa chắc chắn | Mọi ý đều có căn cứ |
| Nguồn trích dẫn | Không có/sai | Có nhưng chưa đầy đủ | Đúng và dễ kiểm tra |
| Hiểu câu hỏi | Hiểu sai | Hiểu một phần | Hiểu đúng ý định |
| Dễ đọc | Khó hiểu | Tạm rõ | Rõ ràng, có cấu trúc |

Điểm nội dung của một câu hỏi:

```text
Tổng điểm = Chính xác + Đầy đủ + Bám sát tài liệu
           + Nguồn trích dẫn + Hiểu câu hỏi + Dễ đọc
```

Điểm tối đa là 12. Ngoài ra cần ghi riêng:

- Thời gian phản hồi tính bằng giây.
- RAM và VRAM sử dụng.
- Tỷ lệ câu trả lời có thông tin không nằm trong tài liệu.
- Chi phí API, nếu Gemini được gọi qua API trả phí.
- Khả năng hoạt động khi mất Internet.

### 11.5. Bảng ghi kết quả

| STT | Câu hỏi | Local /12 | Gemini /12 | Local (giây) | Gemini (giây) | Nhận xét |
|---:|---|---:|---:|---:|---:|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| ... |  |  |  |  |  |  |

Sau khi có kết quả, tính điểm trung bình:

```text
Điểm trung bình = Tổng điểm của tất cả câu / Số câu kiểm thử
```

Nên tính riêng theo từng nhóm câu hỏi. Ví dụ, local có thể tốt ở câu hỏi chứa mã văn bản nhờ BM25, còn Gemini có thể tốt hơn ở câu hỏi cần diễn đạt hoặc suy luận phức tạp.

### 11.6. Khung so sánh dự kiến

| Khía cạnh | Chatbot local | Chatbot dùng Gemini |
|---|---|---|
| Quyền kiểm soát | Chủ động model, dữ liệu và cấu hình | Phụ thuộc dịch vụ và API |
| Riêng tư | Có thể giữ dữ liệu trong máy khi tắt tracing | Phải xem chính sách và cách gửi dữ liệu của dịch vụ |
| Internet | Có thể chạy không Internet sau khi đã tải model | Thường cần kết nối tới dịch vụ |
| Chi phí theo lượt | Không có phí API nhưng tốn phần cứng và điện | Có thể có hạn mức hoặc phí API |
| Tốc độ | Phụ thuộc cấu hình máy; có thể chậm khi nạp lại model | Phụ thuộc mạng và tốc độ dịch vụ |
| Chất lượng suy luận | Bị giới hạn bởi model 4B | Model cloud lớn có thể mạnh hơn |
| Khả năng tùy chỉnh | Dễ kiểm soát pipeline RAG | Tùy thuộc tính năng của nền tảng/API |
| Bảo trì | Người phát triển tự vận hành | Nhà cung cấp vận hành model |

Bảng trên là phân tích kiến trúc, không phải kết quả chất lượng. Kết luận chatbot nào trả lời tốt hơn phải dựa vào bảng kiểm thử thực tế.

## 12. Kết quả thực nghiệm

Phần này được hoàn thiện sau khi chạy bộ câu hỏi trên cả hai hệ thống.

### 12.1. Môi trường thử nghiệm

- CPU: ........................................................
- RAM: 16 GB
- GPU: NVIDIA RTX 3050 4 GB
- Model local: Qwen3 4B Instruct
- Embedding: BAAI/bge-m3
- Vector database: Chroma
- Model Gemini dùng để so sánh: ........................................................
- Số tài liệu: ........................................................
- Tổng số chunk: ........................................................
- Số câu hỏi kiểm thử: ........................................................

### 12.2. Kết quả tổng hợp

| Chỉ số | Chatbot local | Gemini |
|---|---:|---:|
| Điểm chính xác trung bình |  |  |
| Điểm đầy đủ trung bình |  |  |
| Điểm bám sát tài liệu trung bình |  |  |
| Điểm nguồn trung bình |  |  |
| Tổng điểm trung bình /12 |  |  |
| Thời gian phản hồi trung bình |  |  |
| Tỷ lệ câu có thông tin ngoài tài liệu |  |  |

### 12.3. Nhận xét kết quả

Cần trình bày rõ:

- Hệ thống nào tìm đúng tài liệu nhiều hơn?
- Hệ thống nào trả lời đầy đủ hơn?
- Hệ thống nào ít đưa ra thông tin không có căn cứ hơn?
- Hệ thống nào trích dẫn nguồn dễ kiểm tra hơn?
- Chênh lệch thời gian phản hồi là bao nhiêu?
- Local có đáp ứng yêu cầu thực tế dù không đạt điểm cao nhất hay không?

## 13. Kết luận mẫu

Sau khi có số liệu, có thể viết kết luận theo cấu trúc sau:

> Đề tài đã xây dựng thành công chatbot RAG chạy local, sử dụng Qwen3 4B Instruct để phân tích và trả lời, BGE-M3 kết hợp BM25 để truy xuất tài liệu, RRF để gộp kết quả và Chroma để lưu vector. Hệ thống đọc được nhiều định dạng tài liệu, hỗ trợ câu hỏi nối tiếp, câu hỏi nhiều ý và hiển thị nguồn tham khảo.
>
> Trên bộ kiểm thử gồm ... câu hỏi, chatbot local đạt .../12 điểm trung bình, trong khi Gemini đạt .../12. Chatbot local có ưu thế ở ..., còn Gemini có ưu thế ở .... Vì vậy, kết quả cho thấy ....
>
> Chatbot local phù hợp khi ưu tiên quyền kiểm soát dữ liệu, khả năng tùy chỉnh và hoạt động nội bộ. Tuy nhiên, hệ thống vẫn cần cải thiện bước truy xuất, xử lý PDF scan và đánh giá tự động trước khi triển khai rộng rãi.

Không nên viết “chatbot local tốt hơn Gemini” nếu số liệu chỉ cho thấy local tốt hơn ở một vài tiêu chí. Có thể kết luận chính xác hơn, ví dụ: “local phù hợp hơn với yêu cầu riêng tư và chi phí”, hoặc “Gemini đạt độ chính xác cao hơn nhưng local vẫn đáp ứng được ngưỡng sử dụng”.

## 14. Hướng phát triển

1. Thêm OCR cho PDF scan.
2. Dùng bộ tách từ tiếng Việt tốt hơn cho BM25.
3. Xây dựng bộ câu hỏi và đáp án chuẩn để đánh giá tự động.
4. Đo Recall@K để biết retriever có tìm được chunk chứa đáp án hay không.
5. Thử nghiệm lại chunk size và chunk overlap.
6. Cân nhắc reranker nếu kết quả RRF còn nhiều chunk nhiễu.
7. Tóm tắt lịch sử khi hội thoại dài.
8. Thêm đăng nhập và phân quyền theo phòng ban.
9. Ghi nhận thời gian xử lý của từng bước.
10. Tách riêng đánh giá retriever và đánh giá model sinh câu trả lời.

## 15. Kịch bản thuyết trình ngắn

### Slide 1 – Bài toán

“Tài liệu của trường nằm ở nhiều tệp và người dùng mất thời gian tra cứu. Em xây dựng chatbot để hỏi bằng ngôn ngữ tự nhiên và nhận câu trả lời kèm nguồn.”

### Slide 2 – Mục tiêu nghiên cứu

“Mục tiêu không chỉ là làm chatbot chạy được. Em muốn kiểm tra một model nhỏ chạy local có đáp ứng bài toán tài liệu trường và có ưu điểm gì so với Gemini.”

### Slide 3 – Các model

“Qwen3 4B chịu trách nhiệm hiểu câu hỏi và viết câu trả lời. BGE-M3 chuyển câu hỏi và tài liệu thành vector để tìm theo ý nghĩa. BM25 bổ sung khả năng tìm theo từ khóa.”

### Slide 4 – Luồng nạp dữ liệu

“Tài liệu được đọc, gắn thông tin nguồn, chia thành chunk, embedding bằng BGE-M3 và lưu vào Chroma. Bước này chỉ cần chạy lại khi tài liệu thay đổi.”

### Slide 5 – Luồng hỏi đáp

“Khi có câu hỏi, Qwen hiểu ngữ cảnh và tách ý. BGE-M3 cùng BM25 tìm tài liệu. RRF gộp hai danh sách. Các chunk tốt nhất được đưa cho Qwen trả lời và trích nguồn.”

### Slide 6 – Vì sao dùng hybrid search?

“Vector search hiểu ý nghĩa nhưng có thể bỏ qua mã hoặc cụm từ chính xác. BM25 tìm từ khóa tốt nhưng khó hiểu cách diễn đạt khác. Kết hợp hai phương pháp giúp bù trừ hạn chế cho nhau.”

### Slide 7 – Tối ưu local

“Do GPU chỉ có 4 GB, hệ thống không giữ BGE-M3 và Qwen trong bộ nhớ cùng lúc. BGE-M3 tìm xong sẽ được giải phóng rồi Qwen mới chạy. Đổi lại, tốc độ có thể chậm hơn.”

### Slide 8 – Cách so sánh Gemini

“Hai chatbot được dùng cùng tài liệu, cùng câu hỏi và cùng yêu cầu. Em chấm độ chính xác, đầy đủ, bám tài liệu, nguồn, khả năng hiểu câu và thời gian phản hồi.”

### Slide 9 – Kết quả

Trình bày số liệu thực tế bằng bảng hoặc biểu đồ, không chỉ đưa một vài ví dụ thuận lợi.

### Slide 10 – Kết luận

“Local có/không đạt chất lượng tương đương Gemini trong bộ thử nghiệm. Điểm mạnh chính là ..., điểm yếu là ..., và hướng cải thiện tiếp theo là ...”

## 16. Những câu người hướng dẫn có thể hỏi

### Vì sao không chỉ dùng Qwen để đọc toàn bộ tài liệu?

Vì tài liệu có thể dài hơn giới hạn context, tốn bộ nhớ và chứa nhiều phần không liên quan. RAG chọn trước các đoạn phù hợp nên đầu vào cho Qwen ngắn và tập trung hơn.

### Vì sao phải chia chunk?

Nếu đưa cả tệp dài vào tìm kiếm, vector sẽ đại diện cho quá nhiều nội dung khác nhau. Chunk nhỏ giúp hệ thống xác định chính xác đoạn có đáp án.

### Vì sao cần cả BGE-M3 và BM25?

BGE-M3 tìm theo nghĩa, còn BM25 tìm từ khóa chính xác. Hai phương pháp giải quyết hai loại câu hỏi khác nhau và được kết hợp bằng RRF.

### RRF có tác dụng gì?

RRF gộp thứ hạng của hai bộ tìm kiếm mà không cần đưa điểm BM25 và điểm vector về cùng một thang đo.

### Nếu chatbot trả lời sai thì sai ở đâu?

Cần kiểm tra theo hai tầng:

1. Retriever có tìm đúng chunk chứa đáp án không?
2. Nếu đã có đúng chunk, Qwen có đọc và tổng hợp đúng không?

Nếu chunk đúng không được tìm thấy, tăng model sinh câu trả lời chưa chắc giải quyết được vấn đề.

### Local có chắc chắn bảo mật hơn không?

Chỉ chắc chắn khi toàn bộ dữ liệu và log đều ở trong máy. Nếu bật LangSmith tracing hoặc dùng dịch vụ ngoài, một phần nội dung có thể được gửi ra ngoài.

### Vì sao chưa thể khẳng định local tốt hơn Gemini?

Vì cần một bộ câu hỏi, đáp án chuẩn và tiêu chí đo thống nhất. Kết luận phải dựa vào số liệu thực nghiệm trên cùng điều kiện.
