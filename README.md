# Hybrid Recommendation System – Suggest Product

## 1. Giới thiệu

**Suggest Product** là một **Hybrid Recommendation System** được xây dựng nhằm mô phỏng một hệ thống gợi ý sản phẩm hoàn chỉnh trong thực tế. Dự án triển khai **end-to-end pipeline** cho bài toán Recommendation System, bao gồm từ xử lý dữ liệu thô, xây dựng đặc trưng, phân cụm người dùng, khai phá luật kết hợp có ngữ cảnh, cho đến sinh gợi ý, đánh giá offline và triển khai demo web.

Hệ thống không tập trung vào một thuật toán đơn lẻ, mà nhấn mạnh vào **kiến trúc kết hợp (hybrid architecture)**, trong đó mỗi thành phần đảm nhiệm một vai trò riêng trong quá trình ra quyết định gợi ý.

### Các kỹ thuật chính

- User Clustering (Behavior, Preference, Lifecycle)
- Context-Aware Association Rules (FP-Growth)
- Hybrid Recommendation Engine
- Offline Evaluation
- Web API & Frontend Demo

Mục tiêu của dự án là xây dựng một hệ thống có **cấu trúc rõ ràng, dễ mở rộng**, phù hợp cho **đồ án học phần / đồ án tốt nghiệp / nghiên cứu học thuật** trong lĩnh vực *Machine Learning & Data Science*.

---

## 2. Kiến trúc tổng thể hệ thống

```
Raw Data
  → Preprocessing
    → Feature Engineering
      → User Clustering
        → Context-Aware Association Rules
          → Hybrid Recommendation Engine
            → Offline Evaluation & Web API
```

---

## 3. Cấu trúc thư mục

```
Suggest_Product/
├── dataset/
├── src/
├── checkpoints/
├── results/
├── main/
├── notebooks/
├── web/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## 4. Dataset (dataset/processed)

| File | Mô tả |
|------|------|
| behavior_features.csv | Đặc trưng hành vi mua sắm |
| preference_features.csv | Đặc trưng sở thích theo department |
| lifecycle_features.csv | Đặc trưng vòng đời người dùng |
| transactions_context.csv | Giao dịch có gắn ngữ cảnh |
| transactions_context_extended.parquet | Giao dịch mở rộng |
| user_features.csv | Tổng hợp đặc trưng người dùng |

---

## 5. Preprocessing & Feature Engineering

Thực hiện làm sạch dữ liệu, xây dựng đặc trưng hành vi, sở thích, vòng đời và transaction có ngữ cảnh.

---

## 6. User Clustering

- **Behavior Clustering**: dựa trên hành vi mua sắm
- **Preference Clustering**: dựa trên phân bố sở thích sản phẩm
- **Lifecycle Assignment**: phân loại New / Active / Loyal / Churn

---

## 7. Association Rules – Context Aware

Khai phá luật kết hợp có xét ngữ cảnh bằng **FP-Growth**, lưu luật và index phục vụ truy vấn nhanh.

---

## 8. Hybrid Recommendation Engine

Luồng sinh gợi ý:

```
User + Context
 → Candidate Generation
   → Behavior Adjustment
     → Preference Filtering
       → Lifecycle Adjustment
         → Ranking
```

Hỗ trợ fallback cho cold-start user.

---

## 9. Evaluation

Đánh giá offline bằng Precision@K, Recall@K, Hit Rate@K.  
Kết quả lưu trong `results/evaluate/`.

---

## 10. Web Demo

- Backend: FastAPI
- Frontend: HTML / CSS / JavaScript  
Mục đích minh họa khả năng hoạt động của hệ thống.

---

## 11. Cách chạy hệ thống

### Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### Chạy pipeline

```bash
python run_pipeline.py
```

Hoặc chạy từng bước trong thư mục `main/`.

---

## 12. Định hướng mở rộng

- Collaborative Filtering
- Online evaluation (A/B testing)
- Real-time personalization

---

## 13. Kết luận

Dự án minh họa một hệ thống gợi ý **Hybrid & Context-Aware**, kết hợp nhiều tín hiệu để tạo gợi ý cá nhân hóa, phù hợp cho học tập và nghiên cứu.

---

**Tác giả**: Long Pham  
**Lĩnh vực**: Machine Learning – Recommendation Systems
