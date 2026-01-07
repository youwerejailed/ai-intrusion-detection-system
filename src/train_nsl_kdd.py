import argparse
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import joblib

# NSL-KDD: 41 feature + label + (opsiyonel) difficulty
COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

def load_nsl_kdd(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLS)

def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = (df["label"] != "normal").astype(int)
    return df

def build_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    categorical = ["protocol_type", "service", "flag"]
    numeric = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_name == "logreg":
        model = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
        )
    else:
        raise ValueError("model_name must be 'logreg' or 'rf'")

    return Pipeline([("preprocess", pre), ("model", model)])

def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/KDDTrain+.txt")
    ap.add_argument("--test", default="data/raw/KDDTest+.txt")
    ap.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--model_out", default=None, help="Optional output path. If not set, auto-named.")
    args = ap.parse_args()

    df_train = make_binary_label(load_nsl_kdd(args.train))
    df_test  = make_binary_label(load_nsl_kdd(args.test))

    drop_cols = ["label"]
    if "difficulty" in df_train.columns:
        drop_cols.append("difficulty")

    X_train = df_train.drop(columns=drop_cols + ["y"])
    y_train = df_train["y"].astype(int)

    X_test = df_test.drop(columns=drop_cols + ["y"])
    y_test = df_test["y"].astype(int)

    pipe = build_pipeline(X_train, args.model)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    roc = roc_auc_score(y_test, y_proba)
    pr  = average_precision_score(y_test, y_proba)

    print("=== Model ===")
    print(f"{args.model} (threshold={args.threshold})")

    print("\n=== Metrics ===")
    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC :", round(pr, 4))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    if not args.no_plots:
        # ROC
        plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve (NSL-KDD) - {args.model}")
        save_fig(Path(f"reports/figures/roc_curve_{args.model}.png"))
        plt.close()

        # PR
        plt.figure()
        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"Precision-Recall Curve (NSL-KDD) - {args.model}")
        save_fig(Path(f"reports/figures/pr_curve_{args.model}.png"))
        plt.close()

        print("\nSaved plots to reports/figures/")

    Path("models").mkdir(exist_ok=True)
    out_path = args.model_out or f"models/nslkdd_{args.model}_pipeline.joblib"
    joblib.dump(pipe, out_path)
    print(f"\nSaved model: {out_path}")

if __name__ == "__main__":
    main()