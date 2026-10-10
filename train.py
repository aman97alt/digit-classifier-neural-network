"""Handwritten digit classification with a PyTorch MLP (64-100-50-10).

Run:  python train.py
Outputs go to ./outputs (metrics.json, loss_curve.png, confusion_matrix.png, model.pt)
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class DigitMLP(nn.Module):
    """Multi-Layer Perceptron: 64 -> 100 -> 50 -> 10 with ReLU activations."""

    def __init__(self, in_features=64, h1=100, h2=50, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # 1. Data: 1797 samples of 8x8 digit images (64 features)
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=args.seed
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_test_s = (scaler.transform(a) for a in (X_tr, X_val, X_test))

    def to_loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa, dtype=torch.float32), torch.tensor(ya, dtype=torch.long))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = to_loader(X_tr, y_tr, True)
    val_loader = to_loader(X_val, y_val, False)
    test_loader = to_loader(X_test_s, y_test, False)

    # 2. Model, loss, optimizer
    model = DigitMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training loop (backpropagation); keep the best model by validation loss
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train loss {history['train_loss'][-1]:.4f} "
                  f"| val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    model.load_state_dict(best_state)

    # 4. Final evaluation on the untouched test set
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_s, dtype=torch.float32)).argmax(1).numpy()
    mlp_acc = accuracy_score(y_test, preds)

    # 5. Baseline: Logistic Regression on the same split
    base = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    base_acc = accuracy_score(y_test, base.predict(X_test_s))

    print(f"\nPyTorch MLP test accuracy      : {mlp_acc:.4f}")
    print(f"Logistic Regression baseline   : {base_acc:.4f}\n")
    print(classification_report(y_test, preds, digits=3))

    # 6. Save plots, metrics and model
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "pytorch_mlp_test_accuracy": round(float(mlp_acc), 4),
                "logistic_regression_test_accuracy": round(float(base_acc), 4),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    torch.save(model.state_dict(), out_dir / "model.pt")
    print("Saved results to ./outputs")


if __name__ == "__main__":
    main()
