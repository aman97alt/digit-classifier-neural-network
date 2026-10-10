# Handwritten Digit Classification (PyTorch)

A Multi-Layer Perceptron (64 -> 100 -> 50 -> 10, ReLU, Adam) built in PyTorch to
recognize handwritten digits from the scikit-learn digits dataset (1,797 samples, 8x8 images).
It is compared against a Logistic Regression baseline on the same split.

## Run
```
pip install -r requirements.txt
python train.py
```
Optional flags: `--epochs 60 --batch-size 32 --lr 0.001 --seed 42`

## What it does
- Stratified train / validation / test split, StandardScaler on the features
- Training loop with backpropagation (CrossEntropyLoss + Adam), best model kept by validation loss
- Test accuracy vs Logistic Regression baseline, per-class precision/recall
- Saves `outputs/loss_curve.png`, `outputs/confusion_matrix.png`, `outputs/metrics.json`, `outputs/model.pt`

## Results
 PyTorch MLP: 97.78% test accuracy. Logistic Regression baseline: 96.67%. Stratified 80/20 split on 1,797 samples.
