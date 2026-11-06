"""
prepare_data.py  -  Task 2: Data Preparation
Project : Frog Species Classification from Acoustic Call Features Using a Custom MLP
Dataset : Anuran Calls (MFCCs), UCI ML Repository, CC BY 4.0

What this script does
  1. Loads the CSV and checks shape, missing values, duplicates, class counts
  2. Counts outliers (IQR rule) per feature
  3. Saves 3 plots: class distribution, feature histograms, correlation heatmap
  4. Splits by RECORDING (not by row) inside each species, about 70/15/15
  5. Standardises the 22 MFCC features (scaler fitted on TRAIN only)
  6. Saves processed arrays, scaler and label encoder

Run from the project folder:
    python src/prepare_data.py
"""

import os
import json
import random

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed (works inside Docker)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------------------------------ settings
SEED = 42
DATA_PATH = os.environ.get("DATA_PATH", "data/Frogs_MFCCs.csv")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
FIG_DIR = os.environ.get("FIG_DIR", "outputs/figures")
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

random.seed(SEED)
np.random.seed(SEED)

for d in (PROCESSED_DIR, MODELS_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------ 1. load
df = pd.read_csv(DATA_PATH)
# column names look like "MFCCs_ 1" (with a space), so pick them by prefix
feature_cols = [c for c in df.columns if c.startswith("MFCCs_")]
label_col = "Species"
group_col = "RecordID"  # used ONLY for splitting, never as an input feature

print("=" * 60)
print("1. BASIC CHECKS")
print("=" * 60)
print("Shape                :", df.shape)
print("Number of features   :", len(feature_cols))
print("Missing values       :", int(df.isna().sum().sum()))
print("Duplicate rows       :", int(df.duplicated().sum()))
print("Number of species    :", df[label_col].nunique())
print("Number of recordings :", df[group_col].nunique())
print("\nSamples per species:")
print(df[label_col].value_counts().to_string())

# ------------------------------------------------------------------ 2. outliers
print("\n" + "=" * 60)
print("2. OUTLIERS (IQR rule: below Q1-1.5*IQR or above Q3+1.5*IQR)")
print("=" * 60)
X_all = df[feature_cols]
q1, q3 = X_all.quantile(0.25), X_all.quantile(0.75)
iqr = q3 - q1
is_outlier = (X_all < (q1 - 1.5 * iqr)) | (X_all > (q3 + 1.5 * iqr))
outliers_per_feature = is_outlier.sum()
rows_with_outlier = int(is_outlier.any(axis=1).sum())
print(outliers_per_feature.to_string())
print(f"\nRows with at least one outlier value: {rows_with_outlier} "
      f"({100 * rows_with_outlier / len(df):.1f}% of rows)")
print("Decision: keep them. Values are MFCCs already scaled to [-1, 1] and "
      "reflect real variation in frog calls, not errors.")

# ------------------------------------------------------------------ 3. plots
sns.set_theme(style="whitegrid")

# Plot 1: class distribution
counts = df[label_col].value_counts()
plt.figure(figsize=(9, 5))
sns.barplot(x=counts.values, y=counts.index, color="#3b7dd8")
for i, v in enumerate(counts.values):
    plt.text(v + 20, i, str(v), va="center", fontsize=9)
plt.title("Plot 1: Number of syllable samples per species")
plt.xlabel("Samples")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "plot1_class_distribution.png"), dpi=150)
plt.close()

# Plot 2: histograms of all 22 MFCC features
fig, axes = plt.subplots(4, 6, figsize=(16, 9))
for ax, col in zip(axes.ravel(), feature_cols):
    ax.hist(df[col], bins=30, color="#3b7dd8")
    ax.set_title(col.replace(" ", ""), fontsize=9)
    ax.tick_params(labelsize=7)
for ax in axes.ravel()[len(feature_cols):]:
    ax.axis("off")
fig.suptitle("Plot 2: Distribution of the 22 MFCC features", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "plot2_feature_histograms.png"), dpi=150)
plt.close()

# Plot 3: correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[feature_cols].corr(), cmap="coolwarm", center=0,
            xticklabels=[c.replace(" ", "") for c in feature_cols],
            yticklabels=[c.replace(" ", "") for c in feature_cols])
plt.title("Plot 3: Correlation between MFCC features")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "plot3_correlation_heatmap.png"), dpi=150)
plt.close()
print("\nSaved 3 plots to", FIG_DIR)

# ------------------------------------------------------------------ 4. split by recording
# Inside each species, whole recordings go to train / val / test. Recordings have
# very different numbers of syllables, so a random choice can give a species almost
# no training samples. Instead we try every possible assignment (at most a few
# thousand per species) and keep the one whose SAMPLE proportions are closest to
# 70/15/15. Ties are broken with a fixed seed so the result is repeatable.
from itertools import combinations

TARGET = {"train": 1 - VAL_FRACTION - TEST_FRACTION, "val": VAL_FRACTION, "test": TEST_FRACTION}
rng = np.random.RandomState(SEED)
rec_split = {}  # RecordID -> "train" / "val" / "test"

for species, g in df.groupby(label_col):
    sizes = g.groupby(group_col).size().to_dict()  # RecordID -> number of syllables
    recs = sorted(sizes)
    n = len(recs)
    total = sum(sizes.values())
    n_val = max(1, round(VAL_FRACTION * n))
    n_test = max(1, round(TEST_FRACTION * n))
    assert n - n_val - n_test >= 1, f"{species} has too few recordings ({n})"

    candidates = []
    for val_set in combinations(recs, n_val):
        rest = [r for r in recs if r not in val_set]
        for test_set in combinations(rest, n_test):
            train_set = [r for r in rest if r not in test_set]
            frac = {
                "train": sum(sizes[r] for r in train_set) / total,
                "val": sum(sizes[r] for r in val_set) / total,
                "test": sum(sizes[r] for r in test_set) / total,
            }
            cost = sum((frac[k] - TARGET[k]) ** 2 for k in TARGET)
            candidates.append((cost, rng.rand(), train_set, val_set, test_set))
    candidates.sort(key=lambda c: (round(c[0], 10), c[1]))
    _, _, train_set, val_set, test_set = candidates[0]
    for r in train_set:
        rec_split[r] = "train"
    for r in val_set:
        rec_split[r] = "val"
    for r in test_set:
        rec_split[r] = "test"

df["split"] = df[group_col].map(rec_split)

# leakage check: no recording may appear in more than one split
leak = df.groupby(group_col)["split"].nunique().max()
assert leak == 1, "A recording appears in more than one split (data leakage)!"

print("\n" + "=" * 60)
print("4. SPLIT BY RECORDING (no recording appears in two sets)")
print("=" * 60)
summary = (df.groupby([label_col, "split"]).size().unstack(fill_value=0)
           [["train", "val", "test"]])
summary["recordings (tr/va/te)"] = [
    "/".join(str(df[(df[label_col] == sp) & (df["split"] == s)][group_col].nunique())
             for s in ("train", "val", "test"))
    for sp in summary.index
]
print(summary.to_string())
tot = df["split"].value_counts()
print("\nOverall rows: " + ", ".join(
    f"{s} = {tot[s]} ({100 * tot[s] / len(df):.1f}%)" for s in ("train", "val", "test")))

# ------------------------------------------------------------------ 5. encode + scale
le = LabelEncoder()
df["label"] = le.fit_transform(df[label_col])

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])  # fit on TRAIN only
X_val = scaler.transform(val_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])

y_train = train_df["label"].to_numpy()
y_val = val_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()

# ------------------------------------------------------------------ 6. save
np.savez(os.path.join(PROCESSED_DIR, "frogs_processed.npz"),
         X_train=X_train.astype(np.float32), y_train=y_train,
         X_val=X_val.astype(np.float32), y_val=y_val,
         X_test=X_test.astype(np.float32), y_test=y_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# save which recording went where, so the split is documented and repeatable
with open(os.path.join(PROCESSED_DIR, "recording_split.json"), "w") as f:
    json.dump({str(k): v for k, v in sorted(rec_split.items())}, f, indent=2)
with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\n" + "=" * 60)
print("5-6. SCALING AND SAVING")
print("=" * 60)
print("Classes (label index -> species):")
for i, name in enumerate(le.classes_):
    print(f"  {i}: {name}")
print("Train mean (should be about 0):", np.round(X_train.mean(), 4))
print("Train std  (should be about 1):", np.round(X_train.std(), 4))
print("Shapes  train:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)

# baseline for Task 5: predicting the most common class in the test set
majority = np.bincount(y_train).argmax()
baseline_acc = float((y_test == majority).mean())
print(f"\nMajority-class baseline on test set (always predict "
      f"'{le.classes_[majority]}'): accuracy = {baseline_acc:.3f}")
print("Saved processed data, scaler.pkl, label_encoder.pkl.")
