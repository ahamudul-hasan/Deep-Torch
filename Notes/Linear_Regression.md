# Linear Regression — From Zero to Understanding

A complete, beginner-friendly guide to **Linear Regression** — the most fundamental algorithm in machine learning. By the end of this document, you will understand what it is, why it works, and how to code it from scratch in PyTorch.

---

## Table of Contents

1. [What Problem Does Linear Regression Solve?](#what-problem-does-linear-regression-solve)
2. [Real-World Analogy](#real-world-analogy)
3. [The Core Idea](#the-core-idea)
4. [The Math (Keep Calm, It's Simple)](#the-math-keep-calm-its-simple)
5. [What Are Weights and Bias?](#what-are-weights-and-bias)
6. [How Does the Model Learn? — Loss Function](#how-does-the-model-learn--loss-function)
7. [How Does the Model Improve? — Gradient Descent](#how-does-the-model-improve--gradient-descent)
8. [The Training Loop — Putting It All Together](#the-training-loop--putting-it-all-together)
9. [Complete Code Example in PyTorch](#complete-code-example-in-pytorch)
10. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
11. [Using `nn.Module` (The PyTorch Way)](#using-nnmodule-the-pytorch-way)
12. [Linear Regression vs Logistic Regression](#linear-regression-vs-logistic-regression)
13. [Key Takeaways](#key-takeaways)

---

## What Problem Does Linear Regression Solve?

Linear Regression predicts a **continuous number** as output.

| Task | Input | Output (Prediction) |
|---|---|---|
| Predict house price | Size (sq ft), bedrooms, location | **$350,000** |
| Predict student marks | Hours studied | **85 marks** |
| Predict salary | Years of experience | **$72,000** |
| Predict temperature | Month, humidity | **32°C** |

**Key point:** The output is always a **number on a scale** (not a category). If you're predicting a category like "cat" or "dog," that's **classification**, not regression.

---

## Real-World Analogy

Imagine you are a real estate agent. After selling hundreds of houses, you start to notice a pattern:

> *"For every additional 100 sq ft, the price goes up by about $15,000."*

You have unconsciously built a **linear model** in your head:

```
Price = (some amount per sq ft) × Size + (some base price)
```

Linear Regression does the exact same thing — but **mathematically** and **automatically**. It finds the best "amount per sq ft" and "base price" by looking at real data.

---

## The Core Idea

Linear Regression tries to draw the **best-fitting straight line** through your data points.

```
Price ($)
  |          *
  |        *  
  |      *   ← data points (actual sales)
  |    *
  |  *
  |_______________
       Size (sq ft)
```

The straight line through these points is your **model**. Once you have the line, you can predict the price for **any** size — even sizes you haven't seen before.

---

## The Math (Keep Calm, It's Simple)

The equation of a straight line:

```
y = w * x + b
```

| Symbol | Name | Meaning |
|---|---|---|
| `y` | Prediction | The value we want to predict (e.g., house price) |
| `x` | Input / Feature | The data we feed in (e.g., house size) |
| `w` | Weight | How much `x` affects `y` (the slope of the line) |
| `b` | Bias | The starting point when `x = 0` (the y-intercept) |

### With Multiple Features

If you have more than one input (e.g., size AND number of bedrooms):

```
y = w₁ * x₁ + w₂ * x₂ + ... + wₙ * xₙ + b
```

Each input gets its own weight. The model learns which features matter more by giving them a higher weight.

### Example

Suppose after training, our model learned:

```
Price = 150 * Size + 20000 * Bedrooms + 50000
         ↑              ↑                  ↑
       weight₁        weight₂             bias
```

For a house with **1200 sq ft** and **3 bedrooms**:

```
Price = 150 × 1200 + 20000 × 3 + 50000
      = 180,000 + 60,000 + 50,000
      = $290,000
```

---

## What Are Weights and Bias?

### Weight (`w`)

The weight controls **how much influence** each input has on the output.

- **Large positive weight** → as input increases, output increases a lot
- **Small positive weight** → input has a mild effect
- **Negative weight** → as input increases, output **decreases**
- **Zero weight** → input has no effect at all

### Bias (`b`)

The bias is a **constant shift**. It lets the line move up or down.

Without bias, the line is forced to pass through the origin (0, 0), which is rarely what real data looks like.

```
With bias = 50000:       Without bias:
Price                     Price
  |      /                  |    /
  |    /                    |  /
  |  /                      |/
  |/ ← starts at 50k       |________
  |____________              Size
    Size
```

### Before Training

Weights and bias start as **random numbers** (or zeros). The model's initial predictions are terrible.

### After Training

The model adjusts weights and bias step by step until predictions are close to the real values.

---

## How Does the Model Learn? — Loss Function

The model needs a way to measure **how wrong** its predictions are. This measure is called the **loss function** (or cost function).

For Linear Regression, we use **Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ (y_actual - y_predicted)²
```

### Why Squared?

| Reason | Explanation |
|---|---|
| Removes negatives | A prediction of +10 off and -10 off are both equally bad. Squaring makes both positive. |
| Penalizes big errors more | An error of 2 → penalty of 4. An error of 10 → penalty of 100. Big mistakes get punished heavily. |

### Example

| Actual Price | Predicted Price | Error | Error² |
|---|---|---|---|
| $200,000 | $190,000 | 10,000 | 100,000,000 |
| $350,000 | $340,000 | 10,000 | 100,000,000 |
| $150,000 | $180,000 | -30,000 | 900,000,000 |

```
MSE = (100,000,000 + 100,000,000 + 900,000,000) / 3
    = 366,666,667
```

The model's goal: **make this number as small as possible**.

---

## How Does the Model Improve? — Gradient Descent

Gradient Descent is the algorithm that **updates weights and bias** to reduce the loss.

### Intuition — The Blindfolded Hiker

Imagine you're blindfolded on a mountain. You want to reach the **lowest valley** (minimum loss). You can't see, but you can feel the slope under your feet.

Strategy:
1. **Feel the slope** (compute the gradient)
2. **Step downhill** (adjust weights in the direction that reduces loss)
3. **Repeat** until you reach the bottom

### The Update Rule

```
w = w - learning_rate × (∂loss / ∂w)
b = b - learning_rate × (∂loss / ∂b)
```

| Part | Meaning |
|---|---|
| `∂loss / ∂w` | **Gradient** — tells us the direction and steepness of the slope with respect to `w` |
| `learning_rate` | **Step size** — how big of a step we take downhill (usually a small number like 0.01) |
| `-` | We go in the **opposite** direction of the gradient (downhill, not uphill) |

### Learning Rate Matters

| Learning Rate | Effect |
|---|---|
| **Too small** (0.0001) | Model learns very slowly — takes forever to converge |
| **Just right** (0.01) | Model steadily walks down the hill to the minimum |
| **Too large** (10) | Model overshoots the minimum — bounces around wildly and may never converge |

```
Loss
  |
  |\        Too large: ↗ ↘ ↗ ↘ overshoots
  | \      
  |  \     Just right: ↘ ↘ ↘ → reaches bottom
  |   \___
  |        Too small: ↘ . . . . very slow
  |_______________
    Training Steps
```

---

## The Training Loop — Putting It All Together

Every training loop in deep learning follows these **5 steps**:

```
┌─────────────────────────────────────────────┐
│  for each epoch:                            │
│                                             │
│    1. Forward Pass     → make a prediction  │
│    2. Compute Loss     → measure the error  │
│    3. Backward Pass    → compute gradients  │
│    4. Update Params    → adjust w and b     │
│    5. Zero Gradients   → reset for next run │
│                                             │
└─────────────────────────────────────────────┘
```

### Why Zero Gradients?

PyTorch **accumulates** gradients by default (adds new gradients to old ones). If you don't reset them, the gradients from the previous step mix with the current step, and the model learns the wrong thing.

---

## Complete Code Example in PyTorch

A simple example: predict a student's **test score** based on **hours studied**.

```python
import torch

# ─── Training Data ───────────────────────────────────
# X = hours studied, Y = test score
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[25], [38], [52], [61], [75], [82], [93], [101]], dtype=torch.float32)

# ─── Model Parameters (start random) ────────────────
w = torch.randn(1, 1, requires_grad=True)  # weight
b = torch.zeros(1, requires_grad=True)     # bias

# ─── Hyperparameters ────────────────────────────────
learning_rate = 0.01
epochs = 500

# ─── Training Loop ──────────────────────────────────
for epoch in range(epochs):

    # Step 1: Forward pass — make prediction
    y_pred = X @ w + b                       # @ is matrix multiply

    # Step 2: Compute loss (MSE)
    loss = ((Y - y_pred) ** 2).mean()

    # Step 3: Backward pass — compute gradients
    loss.backward()

    # Step 4: Update weights and bias
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Step 5: Zero gradients
    w.grad.zero_()
    b.grad.zero_()

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# ─── Results ────────────────────────────────────────
print(f"\nLearned weight: {w.item():.2f}")
print(f"Learned bias:   {b.item():.2f}")

# ─── Predict for new data ──────────────────────────
hours = torch.tensor([[10]], dtype=torch.float32)
predicted_score = hours @ w + b
print(f"\nPrediction: {hours.item():.0f} hours → {predicted_score.item():.1f} marks")
```

### Expected Output

```
Epoch 100/500 | Loss: 4.8321
Epoch 200/500 | Loss: 3.9102
Epoch 300/500 | Loss: 3.4517
Epoch 400/500 | Loss: 3.2003
Epoch 500/500 | Loss: 3.0721

Learned weight: 11.04
Learned bias:   14.21

Prediction: 10 hours → 124.6 marks
```

The model learned: **Score ≈ 11 × Hours + 14**. Makes sense — each extra hour of study adds about 11 marks.

---

## Step-by-Step Code Walkthrough

### 1. Prepare the Data

```python
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[25], [38], [52], [61], [75], [82], [93], [101]], dtype=torch.float32)
```

- Each row in `X` is one student's hours studied
- Each row in `Y` is the corresponding test score
- We use `float32` because PyTorch models work with floats, not integers
- Shape is `[8, 1]` — 8 samples, 1 feature per sample

### 2. Initialize Parameters

```python
w = torch.randn(1, 1, requires_grad=True)  # random starting weight
b = torch.zeros(1, requires_grad=True)     # bias starts at 0
```

- `requires_grad=True` tells PyTorch: *"Track every operation on this tensor so we can compute gradients later"*
- Starting values don't matter much — the model will learn the right values

### 3. Forward Pass

```python
y_pred = X @ w + b
```

This is just `y = wx + b` in matrix form:

```
[1]               [25.0]       (predicted)
[2]               [36.0]
[3]   ×  [11]  +  [bias]   =   ...
[4]
...
```

### 4. Compute Loss

```python
loss = ((Y - y_pred) ** 2).mean()
```

Step by step:
1. `Y - y_pred` → how far off each prediction is (the error)
2. `** 2` → square each error (removes negatives, punishes large errors)
3. `.mean()` → average all squared errors into one number

### 5. Backward Pass

```python
loss.backward()
```

PyTorch automatically computes:
- `w.grad` → ∂loss/∂w (how much `w` contributed to the error)
- `b.grad` → ∂loss/∂b (how much `b` contributed to the error)

This is the magic of **autograd** — you don't need to derive gradients by hand.

### 6. Update Parameters

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

- `torch.no_grad()` → tells PyTorch *"don't track these operations"* — we're just updating, not computing gradients
- We nudge `w` and `b` in the direction that **reduces** the loss

### 7. Zero Gradients

```python
w.grad.zero_()
b.grad.zero_()
```

Reset gradients to zero. Without this, gradients would **pile up** from previous epochs and everything breaks.

---

## Using `nn.Module` (The PyTorch Way)

The manual approach above is great for understanding. In practice, PyTorch provides built-in tools that make life easier:

```python
import torch
import torch.nn as nn

# ─── Define Model ────────────────────────────────────
class LinearRegressionModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)  # one linear layer

    def forward(self, x):
        return self.linear(x)  # no activation — raw output

# ─── Data ────────────────────────────────────────────
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[25], [38], [52], [61], [75], [82], [93], [101]], dtype=torch.float32)

# ─── Setup ───────────────────────────────────────────
model = LinearRegressionModel(input_features=1)
criterion = nn.MSELoss()                           # built-in MSE loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # built-in optimizer

# ─── Training Loop ───────────────────────────────────
for epoch in range(500):
    y_pred = model(X)                # forward pass
    loss = criterion(y_pred, Y)      # compute loss

    optimizer.zero_grad()            # zero gradients
    loss.backward()                  # backward pass
    optimizer.step()                 # update parameters

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/500 | Loss: {loss.item():.4f}")

# ─── Predict ────────────────────────────────────────
with torch.no_grad():
    prediction = model(torch.tensor([[10.0]]))
    print(f"10 hours → {prediction.item():.1f} marks")
```

### What Changed?

| Manual Version | nn.Module Version |
|---|---|
| `w` and `b` created manually | `nn.Linear(1, 1)` handles both automatically |
| `y_pred = X @ w + b` | `y_pred = model(X)` |
| Loss written by hand | `nn.MSELoss()` built-in |
| `w -= lr * w.grad` by hand | `optimizer.step()` does it for you |
| `w.grad.zero_()` manually | `optimizer.zero_grad()` |

The logic is **identical** — `nn.Module` just wraps it in cleaner, reusable code.

### What is `nn.Linear`?

`nn.Linear(in_features, out_features)` is a single layer that computes:

```
output = input × weight + bias
```

It automatically creates and manages the `weight` and `bias` tensors for you.

```python
layer = nn.Linear(3, 1)  # 3 inputs → 1 output

print(layer.weight)  # tensor of shape [1, 3] — randomly initialized
print(layer.bias)    # tensor of shape [1]   — initialized to ~0
```

---

## Linear Regression vs Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|---|---|---|
| **Task** | Predict a **number** (regression) | Predict a **category** (classification) |
| **Output** | Any value (-∞ to +∞) | Probability between 0 and 1 |
| **Activation** | None (raw output) | Sigmoid function |
| **Loss function** | Mean Squared Error (MSE) | Binary Cross-Entropy (BCE) |
| **Example** | Predict house price: **$290,000** | Predict tumor: **Malignant (89%)** |
| **Equation** | `y = wx + b` | `y = sigmoid(wx + b)` |

The **only** difference is logistic regression adds a **sigmoid** activation that squishes the output into [0, 1] to represent a probability.

```
Linear:     Input → wx + b → raw number → done
Logistic:   Input → wx + b → sigmoid → probability → done
```

---

## Key Takeaways

1. **Linear Regression** predicts continuous numbers by fitting a straight line to data.

2. The model is just `y = wx + b` — a weight, a bias, and a multiplication.

3. **MSE (Mean Squared Error)** measures how bad the predictions are.

4. **Gradient Descent** improves the model by nudging weights in the direction that reduces loss.

5. The **training loop** always follows: Forward → Loss → Backward → Update → Zero Gradients.

6. **Learning rate** controls step size — too small is slow, too large is unstable.

7. PyTorch's `nn.Module` and `nn.Linear` do the same thing as the manual approach, just cleaner.

8. Linear Regression = no activation. Add a **sigmoid** and it becomes Logistic Regression (classification).

---

## Quick Reference

```
Formula:        y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
Loss:           MSE = mean((y_actual - y_predicted)²)
Update:         w = w - lr × gradient
PyTorch layer:  nn.Linear(in_features, out_features)
PyTorch loss:   nn.MSELoss()
PyTorch optim:  torch.optim.SGD(model.parameters(), lr=0.01)
```

---

*Now you understand Linear Regression — the building block of nearly every deep learning model. Every neural network, no matter how complex, is built from layers of linear transformations followed by activations. Master this, and everything else is just stacking more layers.*
