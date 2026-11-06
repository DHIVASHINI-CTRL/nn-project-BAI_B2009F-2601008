"""
train.py  -  Task 4: training and experiments

Trains a baseline and three experiments. Each experiment changes ONE thing
compared with the baseline:

    Exp 0  baseline   hidden (64, 32), Adam lr=0.001, no dropout
    Exp 1  wider      hidden (128, 64)                      <- capacity
    Exp 2  dropout    dropout 0.3                           <- regularisation
    Exp 3  SGD        plain SGD lr=0.01 instead of Adam     <- optimiser

Models are compared on the VALIDATION set only. The test set is never used here;
it is used once, at the end, by test.py.

Outputs
    outputs/experiment_log.csv        results for Appendix B
    outputs/figures/curves_*.png      training vs validation loss / accuracy
    outputs/figures/compare_val_loss.png
    models/model.pt                   best model (by validation macro F1) + config
    models/history.json               per-epoch numbers for every experiment
"""

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import FrogMLP, count_parameters, architecture_table

# ------------------------------------------------------------------ settings
SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

EXPERIMENTS = [
    dict(id=0, name="Baseline", change="-",
         hidden=(64, 32), dropout=0.0, optimizer="adam", lr=1e-3),
    dict(id=1, name="Wider layers", change="hidden 64,32 -> 128,64",
         hidden=(128, 64), dropout=0.0, optimizer="adam", lr=1e-3),
    dict(id=2, name="Dropout 0.3", change="dropout 0 -> 0.3",
         hidden=(64, 32), dropout=0.3, optimizer="adam", lr=1e-3),
    dict(id=3, name="SGD optimiser", change="Adam lr 0.001 -> SGD lr 0.01",
         hidden=(64, 32), dropout=0.0, optimizer="sgd", lr=1e-2),
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # same numbers on different machines


def load_data():
    d = np.load(os.path.join(PROCESSED_DIR, "frogs_processed.npz"))
    return {k: d[k] for k in d.files}


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        loss = loss_fn(logits, torch.from_numpy(y)).item()
        pred = logits.argmax(1).numpy()
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    return loss, acc, f1


def run_experiment(cfg, data):
    set_seed()  # every experiment starts from the same seed
    model = FrogMLP(hidden=cfg["hidden"], dropout=cfg["dropout"])
    loss_fn = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])

    train_ds = TensorDataset(torch.from_numpy(data["X_train"]),
                             torch.from_numpy(data["y_train"]))
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "val_f1")}
    best = dict(f1=-1, epoch=0, state=None)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        tr_loss, tr_acc, _ = evaluate(model, data["X_train"], data["y_train"], loss_fn)
        va_loss, va_acc, va_f1 = evaluate(model, data["X_val"], data["y_val"], loss_fn)
        for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc, va_f1)):
            hist[k].append(v)
        if va_f1 > best["f1"]:
            best = dict(f1=va_f1, epoch=epoch,
                        state={k: v.clone() for k, v in model.state_dict().items()})

    return model, hist, best


def plot_curves(cfg, hist):
    ep = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="Training loss")
    ax[0].plot(ep, hist["val_loss"], label="Validation loss")
    ax[0].set_title(f"Exp {cfg['id']}: {cfg['name']} - loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cross-entropy loss")
    ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="Training accuracy")
    ax[1].plot(ep, hist["val_acc"], label="Validation accuracy")
    ax[1].set_title(f"Exp {cfg['id']}: {cfg['name']} - accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"curves_exp{cfg['id']}_{cfg['name'].split()[0].lower()}.png"),
                dpi=150)
    plt.close()


def main():
    data = load_data()
    print("Train/val sizes:", len(data["y_train"]), len(data["y_val"]))
    print("\nBaseline architecture (for the report):")
    print(f"{'Layer':<10}{'Size':<8}{'Activation':<24}{'Params':>8}")
    for name, size, act, p in architecture_table():
        print(f"{name:<10}{size:<8}{act:<24}{p:>8}")

    history, rows = {}, []
    overall_best = dict(f1=-1)
    for cfg in EXPERIMENTS:
        t0 = time.time()
        model, hist, best = run_experiment(cfg, data)
        history[cfg["id"]] = hist
        plot_curves(cfg, hist)

        gap = hist["train_acc"][-1] - hist["val_acc"][-1]
        rows.append({
            "Exp": cfg["id"], "Name": cfg["name"], "What changed": cfg["change"],
            "Epochs": EPOCHS, "LR": cfg["lr"], "Batch": BATCH_SIZE,
            "Params": count_parameters(model),
            "Best epoch (val F1)": best["epoch"],
            "Val loss (final)": round(hist["val_loss"][-1], 4),
            "Val acc (best epoch)": round(hist["val_acc"][best["epoch"] - 1], 4),
            "Val macro F1 (best)": round(best["f1"], 4),
            "Train-val acc gap (final)": round(gap, 4),
        })
        print(f"Exp {cfg['id']} {cfg['name']:<14} best val F1 {best['f1']:.4f} "
              f"at epoch {best['epoch']:>2} | final train acc "
              f"{hist['train_acc'][-1]:.3f}, val acc {hist['val_acc'][-1]:.3f} "
              f"({time.time() - t0:.0f}s)")
        if best["f1"] > overall_best["f1"]:
            overall_best = dict(f1=best["f1"], cfg=cfg, state=best["state"],
                                epoch=best["epoch"])

    # comparison plot
    plt.figure(figsize=(8, 5))
    for cfg in EXPERIMENTS:
        plt.plot(range(1, EPOCHS + 1), history[cfg["id"]]["val_loss"],
                 label=f"Exp {cfg['id']}: {cfg['name']}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation loss for all experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "compare_val_loss.png"), dpi=150)
    plt.close()

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "experiment_log.csv"), index=False)
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    # save the best model (chosen by VALIDATION macro F1, never by test data)
    best_cfg = overall_best["cfg"]
    torch.save({
        "state_dict": overall_best["state"],
        "hidden": list(best_cfg["hidden"]),
        "dropout": best_cfg["dropout"],
        "n_features": 22,
        "n_classes": 10,
        "experiment": best_cfg["name"],
        "best_epoch": overall_best["epoch"],
    }, os.path.join(MODELS_DIR, "model.pt"))
    with open(os.path.join(MODELS_DIR, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"\nBest model: Exp {best_cfg['id']} ({best_cfg['name']}), epoch "
          f"{overall_best['epoch']}, val macro F1 {overall_best['f1']:.4f} -> models/model.pt")


if __name__ == "__main__":
    main()
