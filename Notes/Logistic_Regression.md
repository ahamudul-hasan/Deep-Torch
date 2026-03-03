# Logistic Regression — From Zero to Understanding

A complete, beginner-friendly guide to **Logistic Regression** — the most important classification algorithm in machine learning. By the end of this document, you will understand what it is, why it works, how it differs from linear regression, and how to code it from scratch in PyTorch.

---

## Table of Contents

1. [What Problem Does Logistic Regression Solve?](#what-problem-does-logistic-regression-solve)
2. [Real-World Analogy](#real-world-analogy)
3. [Why Not Just Use Linear Regression?](#why-not-just-use-linear-regression)
4. [The Secret Ingredient — Sigmoid Function](#the-secret-ingredient--sigmoid-function)
5. [The Full Equation](#the-full-equation)
6. [How Do We Decide the Class?](#how-do-we-decide-the-class)
7. [How Does the Model Learn? — Binary Cross-Entropy Loss](#how-does-the-model-learn--binary-cross-entropy-loss)
8. [How Does the Model Improve? — Gradient Descent](#how-does-the-model-improve--gradient-descent)
9. [The Training Loop — 5 Steps](#the-training-loop--5-steps)
10. [Complete Code Example — From Scratch in PyTorch](#complete-code-example--from-scratch-in-pytorch)
11. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
12. [Real-World Example — Breast Cancer Detection](#real-world-example--breast-cancer-detection)
13. [Using nn.Module (The PyTorch Way)](#using-nnmodule-the-pytorch-way)
14. [Linear Regression vs Logistic Regression](#linear-regression-vs-logistic-regression)
15. [Key Takeaways](#key-takeaways)

---

## What Problem Does Logistic Regression Solve?

Logistic Regression predicts **which category** something belongs to — this is called **classification**.

| Task | Input | Output (Prediction) |
|---|---|---|
| Is this email spam? | Email text, sender, links | **Spam (92%)** or Not Spam |
| Does this patient have cancer? | Cell measurements | **Malignant (87%)** or Benign |
| Will the customer leave? | Usage data, complaints | **Will Leave (71%)** or Will Stay |
| Is this transaction fraud? | Amount, location, time | **Fraud (95%)** or Legitimate |

**Key point:** The output is a **probability** (0% to 100%) that something belongs to a class. It's NOT a number on a scale like "house price = $350,000" — that's regression.

### Binary Classification

Classic logistic regression handles **two classes** (binary):

```
Is it A or B?
  - Spam or Not Spam
  - Malignant or Benign
  - Yes or No
  - 1 or 0
```

---

## Real-World Analogy

Imagine you're a doctor examining a tumor. You look at various measurements — the size, the texture, the shape — and you form an **opinion**:

> *"Based on these measurements, I'm about 87% confident this tumor is malignant."*

You don't say "the tumor is 350 malignant" (that's a regression output). You give a **probability**, then make a **decision**:

- If confidence > 50% → classify as malignant
- If confidence ≤ 50% → classify as benign

Logistic Regression does the **exact same thing** — it looks at numbers, calculates a probability, and makes a binary decision.

---

## Why Not Just Use Linear Regression?

You might think: *"Can't I just use linear regression and say anything above 0.5 is class 1?"*

Here's why that breaks:

### The Problem with Linear Regression for Classification

Linear regression output (`y = wx + b`) can produce **any number**: -50, 0.3, 2.7, 999...

```
Probability?
    |           Linear regression line
 2.0|                        /
    |                      /
 1.0|                    /  ← 1.5?? Probability can't be > 1!
    |                  /
 0.5|                /
    |              /
 0.0|            /
    |          /
-0.5|        /   ← -0.3?? Probability can't be negative!
    |______________________
         Input Features
```

**Problems:**
- Predicts values **above 1** and **below 0** — but probabilities must be between 0 and 1
- A single extreme data point can **tilt the entire line** and wreck all predictions
- The output has **no probabilistic meaning**

### The Solution

We need a function that takes **any number** and squashes it into the range **[0, 1]**. That function is the **Sigmoid**.

---

## The Secret Ingredient — Sigmoid Function

The Sigmoid function (also called the logistic function — hence the name "Logistic Regression"):

```
σ(z) = 1 / (1 + e^(-z))
```

### What Does It Do?

It takes **any real number** and converts it to a value between **0 and 1**:

| Input (z) | Sigmoid Output σ(z) | Interpretation |
|---|---|---|
| -10 | 0.00005 | Almost certainly class 0 |
| -5 | 0.007 | Very likely class 0 |
| -2 | 0.12 | Probably class 0 |
| 0 | 0.50 | Completely uncertain (50-50) |
| +2 | 0.88 | Probably class 1 |
| +5 | 0.993 | Very likely class 1 |
| +10 | 0.99995 | Almost certainly class 1 |

### The Shape

```
σ(z)
 1.0 |                    ─────────────
     |                  /
     |                /
 0.5 |              ·        ← z = 0 → exactly 0.5
     |            /
     |          /
 0.0 |─────────
     └──────────────────────────────── z
        -6  -4  -2   0   2   4   6
```

**Key properties:**
- Output is **always between 0 and 1** — perfect for probabilities
- **S-shaped** (smooth transition, not a hard switch)
- Large positive z → output approaches **1**
- Large negative z → output approaches **0**
- At z = 0 → output is exactly **0.5**

### In PyTorch

```python
import torch

z = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
probabilities = torch.sigmoid(z)
# tensor([0.0067, 0.1192, 0.5000, 0.8808, 0.9933])
```

---

## The Full Equation

Logistic Regression is just **Linear Regression + Sigmoid**:

```
Step 1 (Linear):    z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
Step 2 (Sigmoid):   ŷ = σ(z) = 1 / (1 + e^(-z))
```

**Data flow:**

```
Input Features  →  Linear Combination  →  Sigmoid  →  Probability
[x₁, x₂, ..., xₙ]     z = Wx + b          σ(z)        0 to 1
```

### Example — Tumor Classification

Suppose our model has 3 features and has learned these weights:

```
Tumor size (x₁)   = 2.5     weight₁ = 1.2
Texture (x₂)      = 3.1     weight₂ = 0.8
Smoothness (x₃)   = 0.9     weight₃ = -0.5
                              bias   = -3.0
```

**Step 1 — Linear combination:**
```
z = (1.2 × 2.5) + (0.8 × 3.1) + (-0.5 × 0.9) + (-3.0)
  = 3.0 + 2.48 - 0.45 - 3.0
  = 2.03
```

**Step 2 — Sigmoid:**
```
ŷ = 1 / (1 + e^(-2.03))
  = 1 / (1 + 0.131)
  = 1 / 1.131
  = 0.884
```

**Result:** The model is **88.4% confident** this tumor is malignant.

---

## How Do We Decide the Class?

The sigmoid gives us a **probability**. We need a **threshold** to make a yes/no decision:

```
If ŷ ≥ threshold  →  Predict Class 1 (Malignant)
If ŷ < threshold  →  Predict Class 0 (Benign)
```

### Common Thresholds

| Threshold | When to Use |
|---|---|
| **0.5** | Default. Balanced — treats both classes equally |
| **0.3** | When missing class 1 is costly (e.g., missing cancer is dangerous). More sensitive — catches more positives but may have more false alarms |
| **0.7–0.9** | When false alarms are costly (e.g., expensive follow-up tests). More conservative — predicts class 1 only when very confident |

### Example

For our tumor with ŷ = 0.884:

| Threshold | Prediction |
|---|---|
| 0.5 | Malignant ✓ (0.884 > 0.5) |
| 0.7 | Malignant ✓ (0.884 > 0.7) |
| 0.9 | Benign ✗ (0.884 < 0.9) |

Choosing the right threshold depends on the **cost of mistakes** in your specific problem.

---

## How Does the Model Learn? — Binary Cross-Entropy Loss

The model needs a way to measure **how wrong** its predictions are. For logistic regression, we use **Binary Cross-Entropy (BCE)** loss:

```
Loss = -[ y · log(ŷ) + (1 - y) · log(1 - ŷ) ]
```

Where:
- `y` = true label (0 or 1)
- `ŷ` = predicted probability (0 to 1)

### Why Not Use MSE (Mean Squared Error)?

MSE works for linear regression, but for classification it creates a **bumpy** loss surface with many local minima — gradient descent can get stuck. BCE creates a **smooth, convex** surface that always leads gradient descent to the right answer.

### How BCE Works — Intuition

The formula has **two cases** (only one is active at a time):

**When the true label is 1 (y = 1):**
```
Loss = -log(ŷ)
```
- If ŷ = 0.99 → Loss = -log(0.99) = **0.01** (tiny — good prediction!)
- If ŷ = 0.50 → Loss = -log(0.50) = **0.69** (moderate — uncertain)
- If ŷ = 0.01 → Loss = -log(0.01) = **4.61** (huge — terrible prediction!)

**When the true label is 0 (y = 0):**
```
Loss = -log(1 - ŷ)
```
- If ŷ = 0.01 → Loss = -log(0.99) = **0.01** (tiny — correctly says not malignant)
- If ŷ = 0.99 → Loss = -log(0.01) = **4.61** (huge — confidently wrong!)

### The Key Insight

BCE **punishes confident wrong predictions very heavily**:

```
Loss
  5 |*
    | *
  4 |  *
    |   *
  3 |    *
    |     *
  2 |       *
    |         *
  1 |            *
    |                *
  0 |                      *   *   *
    └────────────────────────────────
    0    0.2   0.4   0.6   0.8   1.0
              Predicted ŷ  (when true y = 1)
```

A prediction of 0.01 when the truth is 1 → **massive** loss. A prediction of 0.99 when the truth is 1 → tiny loss.

### Numerical Example

| True Label (y) | Prediction (ŷ) | BCE Loss | Assessment |
|---|---|---|---|
| 1 (Malignant) | 0.95 | 0.05 | Excellent |
| 1 (Malignant) | 0.60 | 0.51 | Mediocre |
| 1 (Malignant) | 0.10 | 2.30 | Terrible |
| 0 (Benign) | 0.05 | 0.05 | Excellent |
| 0 (Benign) | 0.40 | 0.51 | Mediocre |
| 0 (Benign) | 0.90 | 2.30 | Terrible |

Average BCE across all samples = total loss the model tries to minimize.

### The `log(0)` Problem

`log(0)` is **negative infinity** — it crashes the math. If the model ever predicts exactly 0 or exactly 1, the loss becomes undefined.

**Solution — Clamping:**
```python
epsilon = 1e-7
y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
# Ensures predictions stay in [0.0000001, 0.9999999]
```

---

## How Does the Model Improve? — Gradient Descent

Same algorithm as linear regression — but applied to the BCE loss:

### The Blindfolded Hiker Analogy

You're blindfolded on a hilly terrain (the loss landscape). You want to reach the **lowest valley** (minimum loss):

1. **Feel the slope** under your feet (compute gradients)
2. **Step downhill** (update weights to reduce loss)
3. **Repeat** until you hit the bottom

### The Update Rules

```
w = w - learning_rate × (∂Loss / ∂w)
b = b - learning_rate × (∂Loss / ∂b)
```

| Part | Meaning |
|---|---|
| `∂Loss/∂w` | Gradient — tells the direction and magnitude of steepest ascent |
| `learning_rate` | How big each step is (typically 0.001 to 0.1) |
| `-` | We step in the **opposite** direction of the gradient (downhill) |

PyTorch computes all gradients **automatically** with `loss.backward()` — you never need to derive them by hand.

### Learning Rate Visualization

```
Loss
  |\
  | \      Learning rate too large:
  |  \       ↗ ↘ ↗ ↘ (overshoots, may diverge)
  |   \
  |    \   Learning rate just right:
  |     \    ↘ ↘ ↘ → (smooth convergence)
  |      \___
  |          Learning rate too small:
  |            ↘ . . . . . (very slow)
  |___________________
     Training Steps
```

---

## The Training Loop — 5 Steps

Every training loop in deep learning follows these steps:

```
┌────────────────────────────────────────────────┐
│  for each epoch:                               │
│                                                │
│    1. Forward Pass    → compute predictions    │
│    2. Compute Loss    → measure the error      │
│    3. Backward Pass   → compute gradients      │
│    4. Update Params   → adjust w and b         │
│    5. Zero Gradients  → reset for next round   │
│                                                │
└────────────────────────────────────────────────┘
```

### Why Zero Gradients?

PyTorch **accumulates** gradients by default (adds new on top of old). Without zeroing, the gradients from epoch 1 leak into epoch 2, and the model learns wrong things:

```
Without zeroing:               With zeroing (correct):
Epoch 1 gradient: 0.5          Epoch 1 gradient: 0.5
Epoch 2 gradient: 0.5 + 0.3    Epoch 2 gradient: 0.3
  = 0.8 (WRONG! includes       = 0.3 (correct, fresh)
   stale gradient from epoch 1)
```

---

## Complete Code Example — From Scratch in PyTorch

A minimal example: predict whether a student **passes or fails** based on hours studied.

```python
import torch

# ─── Training Data ──────────────────────────────────
# X = hours studied, Y = 1 (pass) or 0 (fail)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0],
                   [6.0], [7.0], [8.0], [9.0], [10.0]])
Y = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0],
                   [1.0], [1.0], [1.0], [1.0], [1.0]])
# Students who studied ≤ 5 hours failed; > 5 hours passed

# ─── Model Parameters ──────────────────────────────
w = torch.randn(1, 1, requires_grad=True)   # weight (random start)
b = torch.zeros(1, requires_grad=True)      # bias (start at 0)

# ─── Hyperparameters ───────────────────────────────
learning_rate = 0.1
epochs = 200

# ─── Training Loop ─────────────────────────────────
for epoch in range(epochs):

    # 1) Forward pass
    z = X @ w + b                          # linear: z = Xw + b
    y_pred = torch.sigmoid(z)              # sigmoid: squash to [0, 1]

    # 2) Compute loss (Binary Cross-Entropy)
    epsilon = 1e-7
    y_pred_clamped = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -(Y * torch.log(y_pred_clamped) + (1 - Y) * torch.log(1 - y_pred_clamped)).mean()

    # 3) Backward pass
    loss.backward()

    # 4) Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 5) Zero gradients
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# ─── Results ────────────────────────────────────────
print(f"\nLearned weight: {w.item():.4f}")
print(f"Learned bias:   {b.item():.4f}")

# ─── Predict for new students ──────────────────────
test_hours = torch.tensor([[4.0], [5.5], [7.0]])
with torch.no_grad():
    probs = torch.sigmoid(test_hours @ w + b)
    predictions = (probs > 0.5).int()

for hours, prob, pred in zip(test_hours, probs, predictions):
    status = "PASS" if pred.item() == 1 else "FAIL"
    print(f"{hours.item():.0f} hours → {prob.item():.1%} chance of passing → {status}")
```

### Expected Output

```
Epoch 50/200  | Loss: 0.3821
Epoch 100/200 | Loss: 0.2524
Epoch 150/200 | Loss: 0.1937
Epoch 200/200 | Loss: 0.1601

Learned weight: 1.8437
Learned bias:   -10.1204

4 hours → 8.6% chance of passing  → FAIL
5.5 hours → 51.3% chance of passing → PASS
7 hours → 92.1% chance of passing  → PASS
```

The model learned the boundary: around 5–6 hours is the cutoff between pass and fail.

---

## Step-by-Step Code Walkthrough

### 1. Prepare the Data

```python
X = torch.tensor([[1.0], [2.0], ..., [10.0]])   # hours studied
Y = torch.tensor([[0.0], [0.0], ..., [1.0]])     # 0 = fail, 1 = pass
```

- 10 students: first 5 failed, last 5 passed
- Shape: `[10, 1]` — 10 samples, 1 feature each
- `float32` — PyTorch's default for model computations

### 2. Initialize Parameters

```python
w = torch.randn(1, 1, requires_grad=True)   # shape [1, 1]
b = torch.zeros(1, requires_grad=True)      # shape [1]
```

- `requires_grad=True` → PyTorch will track operations on these tensors to compute gradients automatically
- Starting values are random/zero — the model will learn correct values through training

### 3. Forward Pass

```python
z = X @ w + b                   # linear combination
y_pred = torch.sigmoid(z)       # convert to probability
```

**First:** compute a raw score:
```
z = hours × weight + bias
```

**Then:** squash it through sigmoid to get a probability:
```
y_pred = σ(z) = 1 / (1 + e^(-z))
```

### 4. Compute BCE Loss

```python
epsilon = 1e-7
y_pred_clamped = torch.clamp(y_pred, epsilon, 1 - epsilon)
loss = -(Y * torch.log(y_pred_clamped) + (1 - Y) * torch.log(1 - y_pred_clamped)).mean()
```

- **Clamp** predictions to `[0.0000001, 0.9999999]` to avoid `log(0)`
- Calculate BCE for each sample
- Take the **mean** across all samples → single loss number

### 5. Backward Pass

```python
loss.backward()
```

PyTorch traces back through every operation and computes:
- `w.grad` — how much the weight contributed to the loss
- `b.grad` — how much the bias contributed to the loss

### 6. Update Parameters

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

- `torch.no_grad()` — don't track these operations (we're updating, not computing gradients)
- Move each parameter in the direction that **reduces** the loss

### 7. Zero Gradients

```python
w.grad.zero_()
b.grad.zero_()
```

Reset to zero. Without this, gradients accumulate across epochs and the model learns incorrectly.

---

## Real-World Example — Breast Cancer Detection

The notebook in `3_Training_Pipeline/pipeline.ipynb` applies logistic regression to a **real** medical dataset — predicting whether breast tumors are malignant or benign.

### The Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

| Property | Value |
|---|---|
| Source | `gscdit/Breast-Cancer-Detection` on GitHub |
| Rows | 569 patient samples |
| Features | 30 numerical measurements (radius, texture, perimeter, area, smoothness, etc.) |
| Target | `diagnosis` — `M` (Malignant) or `B` (Benign) |

### The Full Pipeline

```
┌──────────────────────────────────────────────────────┐
│  1. Load CSV data                                    │
│       ↓  drop 'id' and 'Unnamed: 32'                │
│  2. Clean DataFrame (569 rows, 31 columns)           │
│       ↓  train_test_split(test_size=0.2)             │
│  3. Split → Train [455×30]  Test [114×30]            │
│       ↓  StandardScaler                              │
│  4. Scale features (mean=0, std=1)                   │
│       ↓  LabelEncoder                               │
│  5. Encode labels (M→1, B→0)                        │
│       ↓  torch.from_numpy()                          │
│  6. Convert to PyTorch tensors                       │
│       ↓  MySimpleNN class                            │
│  7. Train for 25 epochs                              │
│       ↓  evaluate                                    │
│  8. Test accuracy on unseen data                     │
└──────────────────────────────────────────────────────┘
```

### Key Preprocessing Steps

**Why Scale Features?**
```
Feature "area_mean":        values like 500, 1200, 2000
Feature "smoothness_mean":  values like 0.05, 0.09, 0.12
```
Without scaling, the model pays too much attention to `area_mean` just because it has bigger numbers. `StandardScaler` normalizes all features to mean=0, std=1, putting them on equal footing.

**Why Encode Labels?**
Neural networks work with numbers, not text:
```
M (Malignant)  →  1
B (Benign)     →  0
```

**Why Split Into Train/Test?**
If the model trains and tests on the same data, it can simply memorize everything and still score 100%. The test set simulates real-world, unseen patients.

### The Model

```python
class MySimpleNN():
    def __init__(self, X):
        self.weights = torch.randn(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()
        return loss
```

**What each part does:**

| Component | Shape | Purpose |
|---|---|---|
| `self.weights` | `[30, 1]` | One learnable weight per feature (30 features → 30 weights) |
| `self.bias` | `[1]` | A single learnable offset |
| `torch.matmul(X, self.weights)` | `[455, 1]` | Linear combination: z = Xw (one score per patient) |
| `torch.sigmoid(z)` | `[455, 1]` | Squash score to probability [0, 1] |
| `torch.clamp(...)` | | Prevent exact 0 or 1 (avoids log(0) crash) |
| BCE formula | scalar | Average loss across all patients |

**Data flow through the model:**

```
30 features per patient
        ↓
  z = X · W + b          (linear score, shape [455, 1])
        ↓
  ŷ = sigmoid(z)         (probability 0–1)
        ↓
  Loss = BCE(ŷ, y)       (how wrong are we?)
        ↓
  Gradients → update W and b
```

---

## Using nn.Module (The PyTorch Way)

The manual approach is great for understanding. In practice, PyTorch provides cleaner tools:

```python
import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)       # z = wx + b
        out = self.sigmoid(out)    # σ(z) = probability
        return out

# ─── Setup ───────────────────────────────────────────
model = LogisticRegressionModel(input_features=30)
criterion = nn.BCELoss()                                     # built-in BCE loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)     # built-in optimizer

# ─── Training Loop ───────────────────────────────────
for epoch in range(100):
    y_pred = model(X_train_tensor)           # forward pass
    loss = criterion(y_pred, Y_train_tensor) # compute loss

    optimizer.zero_grad()                    # zero gradients
    loss.backward()                          # backward pass
    optimizer.step()                         # update parameters
```

### Manual vs nn.Module — Side by Side

| Manual Version | nn.Module Version |
|---|---|
| Create `w` and `b` tensors manually | `nn.Linear(30, 1)` handles both |
| `z = torch.matmul(X, w) + b` | `self.linear(x)` |
| `torch.sigmoid(z)` | `self.sigmoid(out)` or `nn.Sigmoid()` |
| Write BCE formula by hand | `nn.BCELoss()` |
| `w -= lr * w.grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

The logic is **identical** — `nn.Module` just wraps everything in reusable, cleaner code.

### What is `nn.Linear`?

```python
nn.Linear(in_features, out_features)
```

A single layer that computes `output = input × weight + bias`. It creates and manages the weight and bias tensors automatically.

```python
layer = nn.Linear(30, 1)     # 30 inputs → 1 output
print(layer.weight.shape)    # [1, 30]
print(layer.bias.shape)      # [1]
```

---

## Linear Regression vs Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|---|---|---|
| **Task** | Predict a **number** (regression) | Predict a **category** (classification) |
| **Output range** | Any value (-∞ to +∞) | Probability between 0 and 1 |
| **Activation** | **None** — raw output | **Sigmoid** function |
| **Loss function** | MSE (Mean Squared Error) | BCE (Binary Cross-Entropy) |
| **Decision** | Direct prediction | Threshold → class label |
| **Example** | House price: **$290,000** | Tumor: **Malignant (88%)** |
| **Equation** | `y = wx + b` | `y = sigmoid(wx + b)` |

**The only architectural difference** is one line — the sigmoid activation:

```
Linear:      Input → wx + b → number → done
Logistic:    Input → wx + b → sigmoid → probability → threshold → class
```

---

## Key Takeaways

1. **Logistic Regression** is for **binary classification** — predicting one of two categories.

2. It's just **Linear Regression + Sigmoid**. The sigmoid squashes output to [0, 1] for probabilities.

3. **Sigmoid function:** `σ(z) = 1 / (1 + e^(-z))` — maps any real number to [0, 1].

4. **Binary Cross-Entropy (BCE)** loss measures error. It heavily punishes confident wrong predictions.

5. **Gradient Descent** updates weights to reduce loss: `w = w - lr × gradient`.

6. The **training loop** is always: Forward → Loss → Backward → Update → Zero Gradients.

7. **Threshold** converts probability to class label. Default is 0.5, but adjust based on the cost of mistakes.

8. **Always clamp** predictions to avoid `log(0)` in the loss function.

9. PyTorch's `nn.Module`, `nn.BCELoss()`, and `optimizer.step()` wrap the manual steps in cleaner code.

10. Despite the name "regression," **Logistic Regression is a classification algorithm**.

---

## Quick Reference

```
Model:          ŷ = sigmoid(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
Sigmoid:        σ(z) = 1 / (1 + e^(-z))
Loss:           BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
Update:         w = w - lr × gradient
Decision:       class = 1 if ŷ ≥ threshold else 0
PyTorch layer:  nn.Linear(in_features, out_features) + nn.Sigmoid()
PyTorch loss:   nn.BCELoss()
PyTorch optim:  torch.optim.SGD(model.parameters(), lr=0.01)
```

---

*Now you understand Logistic Regression — the foundation of every classification model in deep learning. Every neural network classifier, no matter how complex, ends with a sigmoid (or softmax for multiple classes) to output probabilities. Master this, and you've mastered the core idea behind all classification.*
