"""
predict.py  -  predict the species for ONE new sample (used in the live demo)

A sample is 22 MFCC values (the same raw features as in the CSV, between -1 and 1).
The script scales them with the SAVED scaler (same transformation as training),
then prints the predicted species and the three most likely species.

Ways to supply a sample:
    python src/predict.py --features "0.15,0.2,...(22 numbers)..."
    python src/predict.py --demo            # uses one real row from the test set
    python src/predict.py --row 100         # uses row 100 of the CSV (0-based)
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from model import FrogMLP

DATA_PATH = os.environ.get("DATA_PATH", "data/Frogs_MFCCs.csv")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, help="22 comma-separated MFCC values")
    ap.add_argument("--row", type=int, help="row number of the CSV to use as the sample")
    ap.add_argument("--demo", action="store_true", help="use the first test-set row")
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(MODELS_DIR, "model.pt"), map_location="cpu",
                      weights_only=True)
    model = FrogMLP(n_features=ckpt["n_features"], hidden=tuple(ckpt["hidden"]),
                    n_classes=ckpt["n_classes"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    true_label = None
    if args.features:
        values = [float(v) for v in args.features.split(",")]
        if len(values) != len(feature_cols):
            raise SystemExit(f"Expected {len(feature_cols)} values, got {len(values)}.")
        x = np.array(values, dtype=float).reshape(1, -1)
    else:
        df = pd.read_csv(DATA_PATH)
        if args.row is not None:
            row = df.iloc[args.row]
        else:  # --demo (or nothing given): first row that belongs to the test set
            with open(os.path.join(PROCESSED_DIR, "recording_split.json")) as f:
                split = {int(k): v for k, v in json.load(f).items()}
            row = df[df["RecordID"].map(split) == "test"].iloc[0]
        x = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
        true_label = row["Species"]
        print("Using a sample from the CSV.")

    x_scaled = scaler.transform(pd.DataFrame(x, columns=feature_cols))
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x_scaled.astype(np.float32))), dim=1)[0].numpy()

    order = np.argsort(-probs)
    print(f"\nPredicted species: {le.classes_[order[0]]}  ({probs[order[0]] * 100:.1f}%)")
    print("Top 3:")
    for i in order[:3]:
        print(f"  {le.classes_[i]:<26}{probs[i] * 100:6.1f}%")
    if true_label is not None:
        print(f"\nTrue species in the dataset: {true_label}")


if __name__ == "__main__":
    main()
