# predictTAC
This repository is created for miniproject in the end of the internship.
## Giới thiệu
Trong các hệ thống mạng di động (4G/5G), việc xác định vị trí của thiết bị người dùng (User Equipment - UE) là rất quan trọng để thực hiện các tác vụ như paging, truyền dữ liệu và duy trì kết nối. Do UE có tính di động cao, mạng không thể luôn biết chính xác vị trí hiện tại của thiết bị. Vì vậy, việc dự đoán TAC tiếp theo dựa trên dữ liệu lịch sử là một hướng tiếp cận hiệu quả nhằm giảm chi phí signaling, tối ưu tài nguyên mạng và tăng hiệu quả paging

## Bài toán
Cho dữ liệu lịch sử di chuyển của UE: (TAC_{t-2}, TAC_{t-1}, TAC_t, time_t) → TAC_{t+1}
Mục tiêu là dự đoán TAC tiếp theo của UE
Trong bài toán sử dụng top-1 prediction, top-3 prediction (quan trọng trong paging)

## Phương pháp sử dụng
1. Markov Chain : dựa trên xác suất chuyển tiếp giữa các TAC, đơn giản, dễ triển khai và dùng làm baseline
2. Random Forest : mô hình học máy (ensemble learning), khai thác lịch sử di chuyển và thông tin thời gian để học được các quy luật phi tuyến
3. Hybrid Model : kết hợp Markov + Random Forest, sử dụng xác suất Markov (markov_prob) làm feature và tận dụng cả thông tin thống kê và ngữ cảnh

## Dữ liệu
Do không có dữ liệu thực tế, dữ liệu được mô phỏng với các đặc điểm:
- ~1000 UE
- 7 ngày liên tục
- Theo từng giờ
- TAC từ T1 → T6
### Hành vi mô phỏng:
- Ngày thường: nhà → Công việc → Nhà
- Cuối tuần: di chuyển linh hoạt hơn
- TAC graph: chỉ cho phép di chuyển giữa các TAC lân cận

### Feature sử dụng
- Nhóm vị trí: current_TAC, prev_TAC, prev_2_TAC
- Nhóm thời gian: hour, day_of_week, is_weekend
- Hybrid: markov_prob

## Cấu trúc project
├── config.py
├── pipeline.py
├── Model/
│   ├── markov_model.py
│   ├── rf_model.py
│   ├── hybrid_model.py
├── Utils/
│   ├── evaluation.py
│   ├── logger.py
├── visualize.py
├── data/
├── logs/
└── models/

## Pipeline xử lý
Load dữ liệu
Encode TAC
Train Markov
Tạo feature markov_prob
Train Random Forest
Train Hybrid Model
Evaluate
Vẽ biểu đồ

## Đánh giá
- Accuracy: top-1 Accuracy, Top-3 Accuracy
- Paging Cost: nếu đúng vị trí k → cost = k, nếu không nằm trong Top-3 → cost = 3
- Cost càng thấp càng tốt

## Cài đặt môi trường
- Các thư viện cần thiết:
pip install pandas numpy scikit-learn matplotlib joblib

## Hướng phát triển
- Sử dụng mô hình chuỗi (LSTM, Transformer)
- Cải thiện Hybrid model
- Áp dụng trên dữ liệu thực tế
- Tối ưu chiến lược paging
