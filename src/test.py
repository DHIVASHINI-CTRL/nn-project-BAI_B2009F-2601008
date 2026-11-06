"""
test.py  -  Task 5: evaluation (also the script that runs in the TEST container)

Loads the saved model, scaler and label encoder, rebuilds the test set from the
raw CSV using the saved recording split, and prints the final metrics.

IMPORTANT: the test set is for ONE final evaluation. Do not use these numbers to
change the model. The model was chosen on validation data in train.py.

It prints:
  - validation metrics (they must match the training log: reproducibility check)
  - test accuracy, macro precision / recall / F1, per-species report
  - comparison with the majority-class baseline and the Task 1 target
  - the 5 most confident WRONG predictions (for the error analysis)
and saves the confusion matrix and result files into outputs/.

Run:  python src/test.py
"""

import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from model import FrogMLP

SEED = 42
TARGET_MACRO_F1 = 0.85  # success criterion from Task 1 / registration form
DATA_PATH = os.environ.get("DATA_PATH", "data/Frogs_MFCCs.csv")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = FrogMLP(n_features=ckpt["n_features"], hidden=tuple(ckpt["hidden"]),
                    n_classes=ckpt["n_classes"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_proba(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X.astype(np.float32))), dim=1).numpy()


def main():
    # ---- load saved objects (the test container needs ALL of these)
    model, ckpt = load_model(os.path.join(MODELS_DIR, "model.pt"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "recording_split.json")) as f:
        rec_split = {int(k): v for k, v in json.load(f).items()}

    print(f"Loaded model: {ckpt['experiment']}, hidden={ckpt['hidden']}, "
          f"dropout={ckpt['dropout']}, best epoch={ckpt['best_epoch']}")

    # ---- rebuild splits from the raw CSV, then scale with the SAVED scaler
    df = pd.read_csv(DATA_PATH)
    df["split"] = df["RecordID"].map(rec_split)
    assert df["split"].notna().all(), "Some recordings are missing from recording_split.json"
    df["label"] = le.transform(df["Species"])

    def get(split):
        part = df[df["split"] == split]
        return part, scaler.transform(part[feature_cols]), part["label"].to_numpy()

    train_df, _, y_train = get("train")
    val_df, X_val, y_val = get("val")
    test_df, X_test, y_test = get("test")

    # ---- reproducibility check on validation data
    p_val = predict_proba(model, X_val).argmax(1)
    _, _, val_f1, _ = precision_recall_fscore_support(
        y_val, p_val, average="macro", zero_division=0)
    print("\n--- Reproducibility check (compare with outputs/experiment_log.csv) ---")
    print(f"Validation accuracy : {accuracy_score(y_val, p_val):.4f}")
    print(f"Validation macro F1 : {val_f1:.4f}")

    # ---- final test evaluation
    proba = predict_proba(model, X_test)
    pred = proba.argmax(1)
    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0)

    majority = np.bincount(y_train).argmax()
    baseline_acc = float((y_test == majority).mean())

    print("\n" + "=" * 60)
    print("TEST SET RESULTS (final, used once)")
    print("=" * 60)
    print(f"Test samples        : {len(y_test)}")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Macro precision     : {prec:.4f}")
    print(f"Macro recall        : {rec:.4f}")
    print(f"Macro F1            : {f1:.4f}")
    print(f"\nMajority-class baseline accuracy ('{le.classes_[majority]}'): {baseline_acc:.4f}")
    print(f"Improvement over baseline: {acc - baseline_acc:+.4f}")
    verdict = "MET" if f1 >= TARGET_MACRO_F1 else "NOT MET"
    print(f"Task 1 target (macro F1 >= {TARGET_MACRO_F1}): {verdict} (achieved {f1:.4f})")

    labels = list(range(len(le.classes_)))
    print("\nPer-species report:")
    print(classification_report(y_test, pred, labels=labels,
                                target_names=le.classes_, zero_division=0))

    # ---- confusion matrix
    cm = confusion_matrix(y_test, pred, labels=labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    names = list(le.classes_)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=8,
                     color="white" if cm[r, c] > thresh else "black")
    plt.xlabel("Predicted species")
    plt.ylabel("True species")
    plt.title("Confusion matrix on the test set")
    plt.tight_layout()
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, "figures", "confusion_matrix_test.png"), dpi=150)
    plt.close()

    # ---- error analysis: 5 most confident wrong predictions
    wrong = np.where(pred != y_test)[0]
    print(f"Wrong predictions: {len(wrong)} of {len(y_test)}")
    conf = proba[wrong, pred[wrong]]
    ordered = wrong[np.argsort(-conf)]  # most confident mistakes first
    # take one example per true species first, so the 5 cases are varied
    worst, seen = [], set()
    for i in ordered:
        if y_test[i] not in seen:
            worst.append(i)
            seen.add(y_test[i])
        if len(worst) == 5:
            break
    for i in ordered:  # fill up if fewer than 5 species were misclassified
        if len(worst) == 5:
            break
        if i not in worst:
            worst.append(i)
    rows = []
    print("\n5 most confident wrong predictions (for the error analysis):")
    for i in worst:
        top2 = np.argsort(-proba[i])[:2]
        rows.append({
            "RecordID": int(test_df.iloc[i]["RecordID"]),
            "True species": le.classes_[y_test[i]],
            "Predicted": le.classes_[pred[i]],
            "Confidence": round(float(proba[i, pred[i]]), 3),
            "Prob. of true species": round(float(proba[i, y_test[i]]), 3),
            "Top-2": ", ".join(le.classes_[top2]),
        })
    wrong_df = pd.DataFrame(rows)
    print(wrong_df.to_string(index=False))
    wrong_df.to_csv(os.path.join(OUT_DIR, "wrong_predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "test_results.json"), "w") as f:
        json.dump({"test_accuracy": acc, "macro_precision": prec, "macro_recall": rec,
                   "macro_f1": f1, "baseline_accuracy": baseline_acc,
                   "validation_macro_f1": val_f1, "n_test": int(len(y_test))}, f, indent=2)
    print("\nSaved: outputs/test_results.json, wrong_predictions.csv, "
          "figures/confusion_matrix_test.png")


if __name__ == "__main__":
    main()
