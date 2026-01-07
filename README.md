# AI-Powered Intrusion Detection (Supervised) — NSL-KDD

Supervised machine learning pipeline for detecting **Normal vs Attack** network traffic using the **NSL-KDD** dataset.

## What this project does
- Loads NSL-KDD (KDDTrain+/KDDTest+)
- Converts multi-class attack labels into binary:
  - `normal` → 0
  - everything else → 1 (attack)
- Preprocessing:
  - One-hot encoding for categorical features (`protocol_type`, `service`, `flag`)
  - Standard scaling for numeric features
- Models:
  - Logistic Regression (baseline)
  - Random Forest (stronger)
- Evaluation:
  - ROC-AUC, PR-AUC
  - Confusion matrix, classification report
  - ROC/PR curves saved under `reports/figures/`

## Quickstart

### 1) Create & activate venv
**Git Bash (Windows):**
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt


<img width="748" height="560" alt="image" src="https://github.com/user-attachments/assets/cc155769-66ae-44c5-8aec-0759053e1475" />
