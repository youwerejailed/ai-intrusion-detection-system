# AI Intrusion Detection System (Supervised) — NSL-KDD

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Task](https://img.shields.io/badge/Task-Intrusion%20Detection-red)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-lightgrey)

Supervised ML pipeline that detects **Normal vs Attack** traffic using the **NSL-KDD** dataset.
Includes preprocessing, model training, evaluation (ROC/PR), threshold tuning, and model export.

---

## ✅ Key Results (NSL-KDD Test Set)

| Model | Threshold | ROC-AUC | PR-AUC | Attack Precision | Attack Recall | FP | FN |
|------|-----------:|--------:|-------:|-----------------:|--------------:|---:|---:|
| Random Forest | 0.25 | 0.9620 | 0.9656 | 0.9663 | 0.7385 | 331 | 3356 |
| Random Forest | 0.35 | 0.9620 | 0.9656 | 0.9684 | 0.6925 | 290 | 3946 |
| Logistic Regression (baseline) | 0.50 | 0.7907 | 0.8653 | 0.9169 | 0.6258 | 728 | 4802 |

**Interpretation:** Threshold controls the security tradeoff: higher **attack recall** vs lower **false positives**.

---

## 🧱 Architecture

```mermaid
flowchart LR
  A[NSL-KDD Raw Data<br/>KDDTrain+, KDDTest+] --> B[Preprocessing]
  B --> C1[Categorical: OneHot<br/>protocol_type, service, flag]
  B --> C2[Numeric: StandardScaler]
  C1 --> D[Model]
  C2 --> D[Model]
  D --> E[Probabilities]
  E --> F[Threshold Decision<br/>Attack if p >= threshold]
  F --> G[Metrics: ROC-AUC, PR-AUC<br/>Confusion Matrix, Report]
  D --> H[Export Pipeline<br/>joblib]
