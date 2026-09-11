# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Quy luật em nhận thấy là dù temperature tăng từ 0.0 lên 1.5, mô hình vẫn giữ nguyên chủ đề cốt lõi (Hang Sơn Đoòng) và cấu trúc câu trả lời tương đối ổn định. Sự khác biệt chủ yếu nằm ở cách hành văn và các chi tiết phụ được lấy ra ngẫu nhiên: ở 0.0 có thêm thông tin về tỷ lệ họ Nguyễn, ở 0.5 và 1.0 xuất hiện chi tiết "ngọc trai hang động", trong khi ở 1.5 lại bổ sung thông tin về "giới hạn 1.000 du khách".

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Với chatbot hỗ trợ khách hàng, em sẽ đặt temperature ở mức 0.0 đến 0.2. Vì trong nghiệp vụ chăm sóc khách hàng, sự chính xác, tính nhất quán và việc bám sát thông tin/chính sách nội bộ là ưu tiên cao nhất. Mức nhiệt độ thấp giúp mô hình đưa ra các câu trả lời an toàn, có thể dự đoán được và giảm thiểu tối đa rủi ro ảo giác như bịa ra chính sách bảo hành hay giá cả sai lệch.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.000 người dùng, mỗi người gọi 3 lần, mỗi lần 350 token đầu ra, hệ thống sẽ sinh ra tổng cộng **10,5 triệu token đầu ra** mỗi ngày. Dựa trên mức giá API hiện tại, **GPT-4o đắt hơn khoảng 25 lần** so với GPT-4o-mini cho workload này.
**Trường hợp GPT-4o xứng đáng với chi phí:** Các tác vụ yêu cầu suy luận logic phức tạp, phân tích dữ liệu chuyên sâu, xử lý tình huống khiếu nại nhiều bước của khách hàng VIP, hoặc các tác vụ đa phương thức cần trích xuất thông tin từ tài liệu/hình ảnh phức tạp.
**Trường hợp nên dùng GPT-4o-mini:** Các tác vụ lặp đi lặp lại khối lượng lớn như phân loại luồng ý định (intent routing), tóm tắt lịch sử chat, hoặc trả lời các câu hỏi FAQ đơn giản đã có sẵn kịch bản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> **Sự khác biệt:** Với persona "giáo viên tiểu học", phản hồi thường ngắn gọn, sử dụng từ vựng đời thường và ví dụ trực quan (như ví von blockchain là "cuốn sổ chung của cả lớp ai cũng xem được nhưng không ai tẩy xóa được"). Ngược lại, persona "chuyên gia tài chính" tạo ra câu trả lời dài hơn, cấu trúc phức tạp và sử dụng đậm đặc thuật ngữ chuyên ngành (sổ cái phân tán, cơ chế đồng thuận, tính phi tập trung). 
> **Ảnh hưởng của System Prompt:** System prompt đóng vai trò định hình bối cảnh, giới hạn không gian từ vựng và thiết lập tông giọng (tone of voice) ngay từ đầu. Nó hoạt động như một "bộ lọc" buộc mô hình phải nhập vai, từ đó căn chỉnh toàn bộ cách diễn đạt và mức độ chi tiết của câu trả lời cho phù hợp với khán giả mục tiêu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> **So sánh số liệu:** Theo công thức ước lượng gốc (số từ / 0.75), 100 từ được kỳ vọng sẽ quy đổi thành khoảng **133 tokens**. Tuy nhiên, khi sử dụng `tiktoken` để đếm thực tế, 100 từ tiếng Việt thường tiêu tốn khoảng **180 đến 220 tokens** (cao hơn mức ước lượng từ **35% đến 65%**).
**Nguyên nhân tiếng Việt tốn nhiều token:** Các bộ tokenizer (như thuật toán BPE của OpenAI) được huấn luyện và tối ưu chủ yếu trên kho dữ liệu tiếng Anh, nơi một từ thường tương ứng gọn gàng với 1 token. Trong khi đó, tiếng Việt chứa nhiều ký tự Unicode có dấu (ă, ơ, ư, thanh điệu) và sử dụng từ ghép. Do không có sẵn trong từ điển token chuẩn, một từ tiếng Việt đơn giản có thể bị "băm" thành 2–4 token dạng byte lẻ, dẫn đến việc tiêu tốn token nhiều hơn đáng kể so với tiếng Anh có cùng độ dài.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming đặc biệt quan trọng trong các ứng dụng tương tác trực tiếp với người dùng (như chatbot, trợ lý ảo), nơi các câu trả lời dài có thể mất nhiều giây để tạo ra; việc hiển thị từng từ (token) ngay lập tức giúp giảm thiểu "thời gian chờ đến token đầu tiên" (Time To First Token), giữ chân người dùng không bị rời đi và tạo cảm giác hệ thống phản hồi tự nhiên. Ngược lại, non-streaming (đợi toàn bộ kết quả) lại phù hợp hơn cho các tác vụ xử lý ngầm (background jobs) như trích xuất dữ liệu có cấu trúc (JSON), dịch thuật tài liệu hàng loạt, hoặc các quy trình tự động hóa – những nơi mà chương trình máy tính cần toàn bộ đầu ra hoàn chỉnh để bắt đầu thực thi bước logic tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với việc dùng delay cố định, chiến lược **exponential backoff** (thử lại với thời gian chờ tăng theo cấp số nhân, ví dụ: 1s, 2s, 4s, 8s) giúp giãn cách thời gian giữa các lần gọi API, từ đó giảm dần áp lực lên máy chủ đang bị quá tải và cho phép hệ thống có không gian để phục hồi. Nếu hàng nghìn client cùng bị từ chối và đồng loạt retry với một khoảng delay cố định giống hệt nhau, chúng sẽ vô tình tạo ra một cuộc tấn công lưu lượng (hiện tượng "Thundering Herd" hay "hiệu ứng bầy đàn"). Toàn bộ lượng request khổng lồ đó sẽ lại ập đến máy chủ cùng một phần nghìn giây ở lần thử tiếp theo, khiến hệ thống không thể phục hồi và tiếp tục sập hoặc chặn API hàng loạt.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> **Persona đã chọn:** Trợ lý hỗ trợ lập trình (Senior Developer Assistant).
**System prompt:** "Bạn là một kỹ sư phần mềm Senior. Hãy trả lời các câu hỏi kỹ thuật một cách ngắn gọn, đi thẳng vào trọng tâm. Luôn cung cấp đoạn code minh họa nếu phù hợp. Trả lời bằng tiếng Việt, nhưng giữ nguyên các thuật ngữ tiếng Anh chuyên ngành (như API, deploy, bug, framework)."
**Giải thích lựa chọn từ ngữ:**
*"Ngắn gọn, đi thẳng vào trọng tâm"*: Lập trình viên thường cần giải pháp ngay lập tức. Yêu cầu này giúp tránh việc mô hình dài dòng giải thích lại các khái niệm cơ bản không cần thiết, đồng thời giúp tiết kiệm token đầu ra (giảm chi phí) và giảm độ trễ (latency).
*"Giữ nguyên thuật ngữ tiếng Anh chuyên ngành"*: Chỉ định rõ ràng này giúp tránh tình trạng LLM dịch máy móc các từ kỹ thuật (ví dụ dịch "load balancing" thành "cân bằng tải" đôi khi mất tự nhiên, hoặc "bug" thành "bọ"), giữ cho văn phong chuyên nghiệp và sát với thực tế công việc IT.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất:** Trợ lý hiện tại chỉ dựa vào bộ nhớ ngắn hạn (ngữ cảnh hội thoại - context window bị giới hạn bởi số lượt chat). Khi cuộc hội thoại kéo dài, các tin nhắn cũ nhất sẽ bị đẩy ra khỏi mảng `messages` để tránh vượt quá giới hạn token (hoặc làm tăng chi phí API), khiến trợ lý "quên" mất những yêu cầu hoặc định dạng mà người dùng đã thiết lập ở đầu buổi chat.
**Đề xuất cải thiện:** Tích hợp bộ nhớ dài hạn bằng kiến trúc **RAG (Retrieval-Augmented Generation)** kết hợp Vector Database.
**Cách triển khai:** 
> 1. Lưu trữ các đoạn hội thoại trước đó (hoặc tài liệu cá nhân của người dùng) dưới dạng vector embedding vào một cơ sở dữ liệu vector (như ChromaDB, Qdrant).
> 2. Mỗi khi người dùng đặt câu hỏi mới, hệ thống sẽ không gửi toàn bộ lịch sử chat khổng lồ đi, mà sẽ truy vấn (query) Vector DB để rút trích ra top 3-5 đoạn hội thoại/tài liệu có nội dung liên quan nhất.
> 3. Tiêm (inject) các đoạn thông tin liên quan này vào system prompt dưới dạng "Ngữ cảnh tham khảo" rồi mới gọi API. Cách này giúp trợ lý nhớ được thông tin quan trọng trong quá khứ mà vẫn tối ưu chi phí token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
