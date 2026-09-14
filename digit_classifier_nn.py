"""
Handwritten Digit Classification using a Neural Network (MLP)
----------------------------------------------------------------
A multi-class image classification project that recognizes handwritten
digits (0-9) from 8x8 pixel images, using a Multi-Layer Perceptron
(feedforward neural network) trained with backpropagation.

Dataset: Digits dataset (scikit-learn) — 1,797 samples, 8x8 grayscale
images (64 pixel features), 10 classes (digits 0-9). Same family of
problem as the classic MNIST digit recognition task, at a smaller scale.

Author: Aman Kumar Diwedi
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save plots to file
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_data():
    data = load_digits()
    X = data.data          # shape: (1797, 64) -- flattened 8x8 pixel images
    y = data.target        # shape: (1797,)    -- digit labels 0-9
    images = data.images   # shape: (1797, 8, 8) -- for visualization
    return X, y, images


def show_sample_digits(images, labels, path="sample_digits.png"):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, img, label in zip(axes.ravel(), images, labels):
        ax.imshow(img, cmap="gray_r")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved sample digit grid -> {path}")


def main():
    # 1. Load data
    X, y, images = load_data()
    print(f"Dataset shape: {X.shape}, Classes: {sorted(set(y))}")
    show_sample_digits(images[:10], y[:10])

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scale features (MLPs train better/faster on scaled inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Baseline: simple linear model, for comparison
    baseline = LogisticRegression(max_iter=3000, random_state=42)
    baseline.fit(X_train_scaled, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test_scaled))
    print(f"\nBaseline (Logistic Regression) test accuracy: {baseline_acc:.4f}")

    # 5. Neural Network: Multi-Layer Perceptron
    #    - 2 hidden layers (64 -> 32 neurons)
    #    - ReLU activation
    #    - Adam optimizer, trained via backpropagation
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=1000,
        random_state=42,
        early_stopping=False,
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 55}")
    print("Neural Network (MLP) Results")
    print(f"{'=' * 55}")
    print(f"Architecture: 64 (input) -> 100 -> 50 -> 10 (output)")
    print(f"Training epochs run: {mlp.n_iter_}")
    print(f"Test accuracy: {mlp_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Neural Network — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    print("Saved confusion matrix -> confusion_matrix.png")

    print(f"\n{'=' * 55}")
    print("Summary")
    print(f"{'=' * 55}")
    print(f"Logistic Regression (baseline): {baseline_acc:.4f}")
    print(f"Neural Network (MLP):           {mlp_acc:.4f}")
    print(f"Improvement: {(mlp_acc - baseline_acc) * 100:+.2f} percentage points")


if __name__ == "__main__":
    main()
