# PyTorch Training Pipeline

A complete, beginner-friendly walkthrough of building a neural network training pipeline **from scratch** using PyTorch. Every step is written manually — no black boxes — so you understand exactly what is happening and why.

The example solves a real-world problem: classifying breast cancer tumors as **malignant** or **benign** using 30 numerical features from medical scans.

---

## The Big Picture

Training a neural network always follows the same pipeline, no matter the problem:

```
Raw Data  →  Preprocessing  →  Model Definition  →  Training Loop  →  Evaluation
```

Each of these stages is covered step by step below.

---

## What You Will Learn

- How to load and clean a real-world CSV dataset
- How to split data into training and test sets (and why)
- How to scale features so the model trains properly
- How to encode text labels into numbers
- How to convert NumPy arrays into PyTorch tensors
- How to define a neural network class from scratch (without `nn.Module`)
- How **Binary Cross-Entropy** loss works and why we need it
- How to implement the full **5-step training loop**
- How to evaluate accuracy on unseen data

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

| Property | Value |
|---|---|
| Source | `gscdit/Breast-Cancer-Detection` on GitHub |
| Rows | 569 patient samples |
| Features | 30 numerical measurements of cell nuclei (radius, texture, perimeter, area, etc.) |
| Target | `diagnosis` column — `M` (Malignant) or `B` (Benign) |
| Dropped columns | `id` (not predictive), `Unnamed: 32` (empty artifact) |

After dropping those two columns:
```
Shape: (569, 31)  →  1 label column + 30 feature columns
```

---

## Prerequisites

Install the required libraries before running the notebook:

```bash
pip install torch numpy pandas scikit-learn
```

---

## Step-by-Step Walkthrough

---

### Step 1 — Import Libraries

```python
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
```

| Library | Role |
|---|---|
| `numpy` | Numerical arrays and math |
| `pandas` | Loading and inspecting CSV data |
| `torch` | Creating tensors, building and training the neural network |
| `train_test_split` | Splitting dataset into training and test portions |
| `StandardScaler` | Normalizing feature values to the same scale |
| `LabelEncoder` | Converting string labels (`M`/`B`) into integers (`1`/`0`) |

---

### Step 2 — Load and Clean the Data

```python
df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')

# Remove columns that add no value
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
```

**Why drop those columns?**

- `id` is just a patient identifier — it has no relationship to whether a tumor is cancerous
- `Unnamed: 32` is an accidental empty column in the CSV file — keeping it would add noise

After cleaning, the DataFrame has **31 columns**: 1 label + 30 features.

---

### Step 3 — Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:],   # X: all feature columns (everything after column 0)
    df.iloc[:, 0],    # y: the label column (diagnosis)
    test_size=0.2     # hold out 20% for testing
)
```

**Why split the data?**

If you train and test on the same data, the model can simply memorize it and still achieve 100% — but it would fail on new patients. The test set simulates real unseen data.

```
Total samples : 569
Training set  : ~455 samples  (80%)
Test set      : ~114 samples  (20%)
```

- `X` = the 30 input features (measurements)
- `y` = the target label (`M` or `B`)

---

### Step 4 — Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  # learn the mean and std from training data, then scale
X_test  = scaler.transform(X_test)       # scale test data using the SAME mean and std
```

**Why scale features?**

Consider two features:
- `area_mean` — values like `500`, `1200`, `2000`
- `smoothness_mean` — values like `0.05`, `0.09`, `0.12`

Without scaling, the model pays much more attention to `area_mean` just because its numbers are bigger. That is unfair and slows down training.

`StandardScaler` transforms every feature to have:
- **Mean = 0**
- **Standard deviation = 1**

This puts all features on equal footing.

> **Golden rule:** Always `fit` on training data only, then `transform` both sets.  
> Fitting on test data would let information from the future "leak" into your model.

---

### Step 5 — Label Encoding

```python
encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)  # fit on training labels and convert
y_test  = encoder.transform(y_test)       # convert using the same mapping
```

Neural networks need numbers, not strings. `LabelEncoder` creates a simple mapping:

```
B  (Benign)    →  0
M  (Malignant) →  1
```

---

### Step 6 — Convert to PyTorch Tensors

```python
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor  = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor  = torch.from_numpy(y_test)
```

PyTorch works with **tensors**, not NumPy arrays. `torch.from_numpy()` does the conversion efficiently — the tensor and the original NumPy array share the same memory block (no copying).

```python
X_train_tensor.shape
# → torch.Size([455, 30])
#   455 training samples, each with 30 features
```

---

### Step 7 — Define the Model

This is a **Logistic Regression** model (a single-layer neural network) implemented as a plain Python class:

```python
class MySimpleNN():

    def __init__(self, X):
        # One weight for each feature, initialized randomly
        self.weights = torch.randn(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
        # Single bias term, initialized to zero
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def forward(self, X):
        # Linear combination: z = X·W + b
        z = torch.matmul(X, self.weights) + self.bias
        # Sigmoid squashes z into a probability between 0 and 1
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y):
        # Clamp to avoid log(0) which is undefined (-infinity)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # Binary Cross-Entropy Loss
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss
```

**Explaining each part:**

| Part | What it does |
|---|---|
| `self.weights` | Shape `[30, 1]` — one learnable weight per input feature. Starts random. |
| `self.bias` | Shape `[1]` — a single learnable offset. Starts at 0. |
| `requires_grad=True` | Tells PyTorch to automatically track gradients for these parameters during backprop |
| `torch.matmul(X, self.weights)` | Multiplies inputs by weights: produces a score `z` for each sample |
| `torch.sigmoid(z)` | Converts any real number into a probability: values close to 1 → likely malignant, close to 0 → likely benign |
| `torch.clamp(...)` | Prevents the prediction from ever being exactly 0 or 1, which would break `log()` |
| Binary Cross-Entropy | Measures how wrong each prediction is. Wrong confident predictions are punished heavily. |

**How the data flows through the model:**

```
30 features per sample
        ↓
  z = X · W + b          (linear combination, one number per sample)
        ↓
  ŷ = sigmoid(z)         (convert to probability 0–1)
        ↓
  probability output
  (> 0.5 → malignant, < 0.5 → benign)
```

**Binary Cross-Entropy Loss formula:**

```
Loss = - [ y · log(ŷ) + (1 - y) · log(1 - ŷ) ]
```

- If the true label is `1` (malignant) and prediction is `0.9` → small loss (good)
- If the true label is `1` (malignant) and prediction is `0.1` → large loss (bad)

---

### Step 8 — Set Hyperparameters

```python
learning_rate = 0.1
epochs = 25
```

| Hyperparameter | What it controls |
|---|---|
| `learning_rate` | The size of each weight update step. Too large → training explodes. Too small → training is very slow. `0.1` is a reasonable starting point. |
| `epochs` | How many times we loop through the entire training dataset. More epochs = more learning (up to a point). |

---

### Step 9 — The Training Loop

This is the core of machine learning — the loop that teaches the model:

```python
# Create the model
model = MySimpleNN(X_train_tensor)

for epoch in range(epochs):

    # ── Step 1: Forward Pass ──────────────────────────────
    # Feed all training data through the model to get predictions
    y_pred = model.forward(X_train_tensor)

    # ── Step 2: Compute Loss ──────────────────────────────
    # Measure how wrong the predictions are compared to true labels
    loss = model.loss_function(y_pred, y_train_tensor)

    # ── Step 3: Backward Pass ─────────────────────────────
    # Compute how much each weight contributed to the loss (gradients)
    loss.backward()

    # ── Step 4: Update Parameters ─────────────────────────
    # Adjust weights and bias to reduce the loss
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias    -= learning_rate * model.bias.grad

    # ── Step 5: Zero the Gradients ────────────────────────
    # Clear gradient values so they don't carry over to the next epoch
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    print(f'Epoch: {epoch + 1}, loss: {loss.item():.4f}')
```

**Breaking down each step:**

**Step 1 — Forward Pass**  
Run the input data through the model to produce predictions. The model uses its current (imperfect) weights.

**Step 2 — Compute Loss**  
Compare predictions to the true labels using the loss function. A high loss means the model is wrong. A low loss means it is doing well.

**Step 3 — Backward Pass (`loss.backward()`)**  
PyTorch automatically computes the **gradient** of the loss with respect to every parameter marked with `requires_grad=True`. A gradient tells us: *"if I increase this weight a little, does the loss go up or down?"*

**Step 4 — Parameter Update (`torch.no_grad()`)**  
Nudge each weight in the direction that reduces the loss:
```
new_weight = old_weight - learning_rate × gradient
```
We use `torch.no_grad()` here because we are doing math on tensors, but we do **not** want PyTorch to track this as part of the computation graph.

**Step 5 — Zero the Gradients**  
PyTorch **accumulates** gradients by default (adds new gradients on top of old ones). If we don't reset them, the next epoch's update will use the wrong (inflated) gradients. Always call `.zero_()` before the next forward pass.

**Expected output:**
```
Epoch: 1,  loss: 2.2553
Epoch: 2,  loss: 2.1972
Epoch: 3,  loss: 2.1437
...
Epoch: 25, loss: 1.6073
```

Loss decreases each epoch — the model is actively learning.

---

### Step 10 — Evaluation

After training, we test the model on data it has never seen before:

```python
with torch.no_grad():
    # Get predictions on test data
    y_pred = model.forward(X_test_tensor)

    # Convert probabilities to class labels using a 0.9 threshold
    # If the model is 90%+ confident it's malignant → predict 1, else → predict 0
    y_pred = (y_pred > 0.9).float()

    # Compare predictions to true labels
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')
```

**What the threshold means:**
- `y_pred > 0.5` → Standard threshold (anything above 50% probability is classified as malignant)
- `y_pred > 0.9` → Strict threshold (only classify as malignant when the model is very confident)

In medical contexts, it can be safer to use a lower threshold to reduce false negatives (missing a real cancer).

**Why `torch.no_grad()` during evaluation?**  
We are not training — there are no weight updates — so we don't need to compute or store gradients. Disabling gradient tracking uses less memory and runs faster.

---

## Full Pipeline at a Glance

```
┌─────────────────────────────────────────────────┐
│  Raw CSV (569 rows, 32 columns)                 │
│       ↓  drop 'id' and 'Unnamed: 32'            │
│  Clean DataFrame (569 rows, 31 columns)         │
│       ↓  train_test_split(test_size=0.2)        │
│  X_train [455×30]    X_test [114×30]            │
│  y_train [455]       y_test [114]               │
│       ↓  StandardScaler                         │
│  Features normalized (mean=0, std=1)            │
│       ↓  LabelEncoder                           │
│  Labels encoded  M→1, B→0                       │
│       ↓  torch.from_numpy()                     │
│  PyTorch Tensors (float64)                      │
│       ↓  MySimpleNN (weights + bias + sigmoid)  │
│  Training Loop × 25 epochs:                     │
│    forward → loss → backward → update → zero    │
│       ↓                                         │
│  Evaluate accuracy on test set                  │
└─────────────────────────────────────────────────┘
```

---

## Common Questions

**Q: Why does `fit_transform` go on training data but only `transform` on test data?**  
A: `fit` learns statistics (mean, std, label mappings) from the data. Using test data during `fit` would leak information about the future into your model — making evaluation unreliable.

**Q: Why do we need `requires_grad=True`?**  
A: PyTorch's autograd engine tracks operations on tensors with `requires_grad=True` to compute gradients automatically during `loss.backward()`. Without it, PyTorch wouldn't know how to compute the gradient for that parameter.

**Q: Why use `torch.no_grad()` during weight updates and evaluation?**  
A: Gradient computation consumes memory and processing time. During weight updates and evaluation, we are just doing arithmetic on tensors — we don't need the computation graph for those operations.

**Q: Why do gradients need to be zeroed each epoch?**  
A: PyTorch adds (accumulates) new gradients on top of existing ones. If you skip `.zero_()`, the stored gradients from the previous epoch are included in the next update, making the update incorrect.

---

## File

| File | Description |
|---|---|
| `pipeline.ipynb` | Jupyter notebook with full working code and outputs |
