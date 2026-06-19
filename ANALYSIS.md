# Phân tích kết quả benchmark — Day 17 / Track 03

Chạy lại bất cứ lúc nào:

```bash
python src/benchmark.py        # in 2 bảng (Standard + Long-Context Stress)
python -m pytest src/test_agents.py -q
```

## Kết quả (offline, tất định)

### Standard Benchmark (`data/conversations.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 1465              | 14972                   | 0.0                  | 0.5              | 0                     | 0           |
| Advanced | 1612              | 21350                   | 1.0                  | 1.0              | 284                   | 0           |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 265               | 22391                   | 0.0                  | 0.5              | 0                     | 0           |
| Advanced | 320               | 11091                   | 1.0                  | 1.0              | 222                   | 24          |

## 1. Vì sao Advanced recall tốt hơn?

Câu hỏi recall luôn được hỏi ở **thread mới** (cross-session). Baseline chỉ có short-term
memory trong một thread → sang thread mới là rỗng → recall = 0 ở cả hai suite. Advanced ghi
các fact ổn định (tên, nơi ở, nghề nghiệp, đồ uống/món ăn, thú cưng, style trả lời, mối quan
tâm) vào `User.md` cho từng user. File này **bền vững xuyên thread**, nên khi mở thread mới
nó vẫn đọc lại được → recall = 1.0.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Ở Standard benchmark, mỗi hội thoại chỉ ~10 lượt ngắn. Mỗi lượt Advanced phải mang theo ngữ
cảnh = `User.md` + summary + recent messages. `User.md` của `dungct` lớn dần qua các phiên,
và bị **đọc lại mỗi lượt** → `Prompt tokens processed` của Advanced (21350) **cao hơn**
Baseline (14972). Khi thread còn ngắn, lịch sử của Baseline chưa kịp phình, nên overhead của
persistent memory chưa được "hoàn vốn". Đây là điểm mấu chốt: thêm memory **không miễn phí**.

## 3. Vì sao compact thắng ở hội thoại dài?

Stress benchmark là **một thread duy nhất, các lượt rất dài**. Baseline tái xử lý **toàn bộ**
lịch sử mỗi lượt → `Prompt tokens processed` tăng gần như bậc hai theo độ dài (22391). Advanced
kích hoạt compact **24 lần**: đẩy các lượt cũ vào một summary ngắn và chỉ giữ
`compact_keep_messages` lượt gần nhất ở dạng nguyên văn. Nhờ đó ngữ cảnh mỗi lượt bị **chặn
trên**, tổng `Prompt tokens processed` chỉ còn **11091 — bằng một nửa Baseline**, trong khi
recall vẫn 1.0 (vì các fact ổn định nằm ở `User.md`, không bị summary làm mất).

Lưu ý quan trọng: compact tối ưu **`Prompt tokens processed`** (chi phí ngữ cảnh), không phải
`Agent tokens only`. Nó không phải lúc nào cũng thắng — ở hội thoại ngắn (Standard) compact
không kích hoạt (0 lần) và Advanced vẫn tốn hơn.

## 4. File memory tăng trưởng & rủi ro

`Memory growth` đo kích thước `User.md`: 284 bytes (standard) / 222 bytes (stress) — nhỏ vì ta
chỉ lưu fact đã chuẩn hoá thay vì nguyên văn hội thoại. Rủi ro thực tế:

- **File phình to**: nếu lưu mọi câu nói, `User.md` lớn dần → mỗi lượt đọc lại càng đắt (đúng
  lỗi của Baseline nhưng dịch sang persistent layer). Cần giữ schema fact gọn.
- **Lưu sai fact**: nếu biến câu hỏi thành fact (vd "đang ở **đâu**?" → nơi ở = "đâu") thì hồ
  sơ bị hỏng. Đã chặn bằng guardrail bỏ qua lượt câu hỏi.
- **Summary làm mất thông tin**: compact là lossy; phải đảm bảo fact ổn định nằm ở `User.md`
  chứ không phụ thuộc summary.

## Guardrails / bonus đã cài (hướng 90–100)

- **Bỏ qua lượt câu hỏi (confidence guard)**: `extract_profile_updates` trả `{}` cho turn
  nghi vấn → không bao giờ ghi câu hỏi thành fact.
- **Conflict handling**: `upsert_fact` ghi đè theo key → khi user đính chính
  (Đà Nẵng→Huế, backend→MLOps) chỉ còn **một** dòng đúng, không giữ song song fact cũ sai.
- **Negation / "current" cue handling**: trong một message có cả thông tin cũ lẫn mới
  (vd "không còn là backend engineer ... vẫn là MLOps engineer"), bộ trích chọn ưu tiên
  giá trị được đánh dấu "hiện tại/vẫn" và loại giá trị bị phủ định.
- **Entity extraction có cấu trúc**: facts lưu theo field (`name`, `location`, `profession`,
  `drink`, `food`, `pet`, `response_style`, `interests`) trong `User.md` dạng markdown
  parse được hai chiều.

### Hướng mở rộng tiếp theo
- **Memory decay**: gắn timestamp cho fact, hạ độ tin cậy theo thời gian để fact cũ tự mờ.
- **Confidence score theo nguồn**: chỉ ghi `User.md` khi độ chắc vượt ngưỡng (vd cần ≥2 lần
  nhắc cho preference mơ hồ).
