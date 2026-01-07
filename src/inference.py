import argparse
import pandas as pd
import joblib

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/nslkdd_logreg_pipeline.joblib")
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    pipe = joblib.load(args.model)

    df = pd.read_csv(args.input, header=None, names=COLS)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out = df.copy()
    out["attack_proba"] = proba
    out["pred_attack"] = pred

    print(out[["attack_proba", "pred_attack"]].head(20).to_string(index=False))
    print("\npred_attack: 1=attack, 0=normal")

if __name__ == "__main__":
    main()