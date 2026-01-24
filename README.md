# Hybrid Recommendation System – Suggest Product

## 1. Giới thiệu

Dự án **Suggest Product** là một hệ thống gợi ý sản phẩm thông minh (Hybrid Recommendation System), được thiết kế nhằm mô phỏng và triển khai một pipeline hoàn chỉnh trong bài toán Recommendation System, từ tiền xử lý dữ liệu, trích xuất đặc trưng, phân cụm người dùng, khai phá luật kết hợp cho đến gợi ý và đánh giá chất lượng hệ thống.

Hệ thống kết hợp nhiều kỹ thuật:

* **User Clustering** (Behavior, Preference, Lifecycle)
* **Context-Aware Association Rules (Apriori)**
* **Hybrid Recommendation Engine**
* **Offline Evaluation & Web Demo**

Mục tiêu chính là tạo ra một kiến trúc rõ ràng, dễ mở rộng, phù hợp cho **đồ án học phần / đồ án tốt nghiệp** trong lĩnh vực Machine Learning & Data Science.

---

## 2. Kiến trúc tổng thể hệ thống

Pipeline tổng quát của hệ thống:

```
Raw Data
  → Preprocessing
    → Feature Engineering
      → User Clustering
        → Association Rules (Context-aware)
          → Hybrid Recommendation Engine
            → Evaluation & Web API
```

---

## 3. Cấu trúc thư mục

```
project-root/
├── dataset/                 # Dữ liệu sau tiền xử lý
├── src/                     # Mã nguồn chính
├── checkpoints/             # Model & object đã train
├── results/                 # Kết quả đánh giá & phân cụm
├── main/                    # Script chạy từng giai đoạn
├── web/                     # Backend & Frontend demo
├── notebooks/               # Notebook phân tích & thử nghiệm
├── run_pipeline.py          # Chạy toàn bộ pipeline
└── README.md
```

---

## 4. Dataset (dataset/processed)

Thư mục này chứa dữ liệu **đã được tiền xử lý**, dùng trực tiếp cho các bước ML:

| File                     | Mô tả                                             |
| ------------------------ | ------------------------------------------------- |
| behavior_features.csv    | Đặc trưng hành vi người dùng                      |
| preference_features.csv  | Đặc trưng sở thích theo category                  |
| lifecycle_features.csv   | Trạng thái vòng đời người dùng                    |
| transactions_context.csv | Giao dịch có gắn ngữ cảnh (time, weekend, bucket) |
| user_features.csv        | Tổng hợp đặc trưng người dùng                     |

---

## 5. Preprocessing & Feature Engineering (src/preprocessing)

Thực hiện các bước:

* Làm sạch dữ liệu
* Chuẩn hóa định dạng
* Trích xuất đặc trưng hành vi, sở thích, vòng đời
* Xây dựng transaction có ngữ cảnh

Các file chính:

* `clean_data.py`
* `build_behavior_features.py`
* `build_preference_features.py`
* `build_lifecycle_features.py`
* `build_transactions_context.py`

---

## 6. User Clustering (src/clustering)

Hệ thống phân cụm người dùng theo 3 khía cạnh:

### 6.1 Behavior Clustering

Phân cụm dựa trên hành vi mua sắm.

* Vector hóa đặc trưng
* Chuẩn hóa dữ liệu
* Huấn luyện KMeans

### 6.2 Preference Clustering

Phân cụm dựa trên sở thích sản phẩm (department/category).

### 6.3 Lifecycle Assignment

Gán người dùng vào các nhóm vòng đời (New, Active, Churn, Loyal) theo luật.

Kết quả được lưu trong thư mục `results/` và model được lưu trong `checkpoints/`.

---

## 7. Association Rules – Context Aware (src/association_rules)

Khai phá luật kết hợp dựa trên Apriori mở rộng, có xét **ngữ cảnh giao dịch**.

Các bước chính:

* Xây dựng itemset có context
* Sinh candidate itemsets
* Tính support, confidence
* Sinh luật A → B
* Lưu luật vào file `.pkl`

File quan trọng:

* `train_context_aware.py`
* `rule_builder.py`
* `association_rules.pkl`

---

## 8. Hybrid Recommendation Engine (src/recommendation)

Kết hợp nhiều nguồn để sinh gợi ý:

* Association Rules
* User Preference Cluster
* Lifecycle Adjustment
* Ranking & Filtering
* Fallback cho cold-start user

File trung tâm:

* `hybrid_recommender.py`

---

## 9. Evaluation (src/evaluation)

Đánh giá offline chất lượng hệ thống bằng các metric phổ biến:

* Precision@K
* Recall@K
* Hit Rate

Kết quả đánh giá được lưu trong thư mục `results/`.

---

## 10. Web Demo (web/)

Hệ thống có demo triển khai đơn giản:

### Backend

* FastAPI
* Endpoint gợi ý sản phẩm

### Frontend

* HTML/CSS/JS
* Streamlit app cho demo nhanh

---

## 11. Cách chạy hệ thống

### 11.1 Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 11.2 Chạy toàn bộ pipeline

```bash
python run_pipeline.py
```

Hoặc chạy từng bước:

```bash
python main/run_preprocessing.py
python main/run_clustering.py
python main/run_associations.py
python main/run_eval.py
```

---

## 12. Định hướng mở rộng

* Áp dụng luật kết hợp k-item (level-3, level-4)
* Thêm mô hình collaborative filtering
* Online evaluation (A/B testing)
* Cá nhân hóa theo thời gian thực

---

## 13. Kết luận

Dự án minh họa một hệ thống gợi ý hoàn chỉnh theo hướng **Hybrid & Context-Aware**, phù hợp cho mục đích học tập, nghiên cứu và trình bày đồ án Machine Learning.

---

**Tác giả**: Long Pham
**Lĩnh vực**: Machine Learning – Recommendation Systems
