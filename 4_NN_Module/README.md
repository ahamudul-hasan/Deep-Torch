# PyTorch nn.Module — Building Neural Networks the Right Way

This folder shows how to **upgrade** the manual training pipeline from scratch into a clean, professional PyTorch implementation using built-in tools. The same Breast Cancer classification problem is solved — but now using PyTorch's `nn` module properly.

---

## What Changed From the Manual Approach?

The previous approach built everything by hand — weights, bias, loss formula, gradient zeroing. This approach replaces all of that with PyTorch's built-in tools:

| Step | Manual (Before) | nn.Module (Now) |
|---|---|---|
| Model definition | Plain Python class | `class Model(nn.Module)` |
| Activation function | `torch.sigmoid(z)` written manually | `nn.Sigmoid()` built-in |
| Loss function | BCE formula written by hand | `nn.BCELoss()` built-in |
| Optimizer / weight update | `w -= lr * w.grad` by hand | `torch.optim.SGD` built-in |
| Zero gradients | `w.grad.zero_()` manually | `optimizer.zero_grad()` |

The logic and math are **identical** — the tools just make it cleaner, safer, and easier to extend.

---

## The Four Improvements

### 1. Building the Neural Network Using `nn.Module`

`nn.Module` is the **base class** for every neural network in PyTorch. All your custom models must inherit from it.

**Why inherit from `nn.Module`?**

- Makes the model **callable** — you can call `model(X)` instead of `model.forward(X)`
- Automatically tracks all layers and their parameters
- Gives access to helper methods like `model.parameters()`, `model.train()`, `model.eval()`
- Required for the optimizer to find your weights and biases

```python
class MySimpleNN(nn.Module):        # ← inherit from nn.Module

    def __init__(self, num_features):
        super().__init__()           # ← initialize the parent class
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):     # ← define the forward pass
        out = self.linear(features)
        out = self.sigmoid(out)
        return out
```

**What is `nn.Linear`?**

`nn.Linear(in_features, out_features)` is a fully connected layer. It computes:

```
output = input × weight + bias
```

It automatically creates and manages the `weight` and `bias` tensors — no need to create them manually.

```python
self.linear = nn.Linear(30, 1)
# 30 input features → 1 output (probability)
# Internally creates: weight of shape [1, 30] and bias of shape [1]
```

**What is `forward()`?**

The `forward` method defines what happens when data flows through the model. PyTorch calls it automatically when you do `model(X)`.

```
model(X_train_tensor)
    ↓  calls forward() automatically
out = self.linear(features)   → linear combination: z = Xw + b
out = self.sigmoid(out)       → squash to probability [0, 1]
return out                    → predicted probability per sample
```

---

### 2. Using Built-in Activation Functions

Instead of writing `torch.sigmoid(z)` manually, PyTorch provides `nn.Sigmoid()` as a layer you can plug into your model.

```python
self.sigmoid = nn.Sigmoid()
```

**Available activation functions in `torch.nn`:**

| Activation | Class | Use Case |
|---|---|---|
| Sigmoid | `nn.Sigmoid()` | Binary classification — output between 0 and 1 |
| ReLU | `nn.ReLU()` | Hidden layers in deep networks — most common |
| Tanh | `nn.Tanh()` | Output between -1 and 1 |
| Softmax | `nn.Softmax()` | Multi-class classification — outputs sum to 1 |

Using built-in activations as layers means they become part of the model's architecture — visible in the model summary and properly handled during training.

---

### 3. Using Built-in Loss Functions

Instead of writing the Binary Cross-Entropy formula by hand, use `nn.BCELoss()`:

```python
loss_function = nn.BCELoss()
```

Then use it during training:

```python
loss = loss_function(y_pred, y_train_tensor.view(-1, 1))
```

**Why `y_train_tensor.view(-1, 1)`?**

The model outputs shape `[N, 1]` (N rows, 1 column). The labels `y_train_tensor` are shape `[N]` (flat). `.view(-1, 1)` reshapes the labels to `[N, 1]` so shapes match.

**Common built-in loss functions:**

| Loss Function | Class | Use Case |
|---|---|---|
| Binary Cross-Entropy | `nn.BCELoss()` | Binary classification (2 classes) |
| Cross-Entropy | `nn.CrossEntropyLoss()` | Multi-class classification |
| Mean Squared Error | `nn.MSELoss()` | Regression (predicting numbers) |
| Mean Absolute Error | `nn.L1Loss()` | Regression (less sensitive to outliers) |

---

### 4. Using Built-in Optimizer

Instead of manually nudging weights with `w -= lr * w.grad`, use `torch.optim`:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Then in the training loop:

```python
optimizer.zero_grad()    # clear old gradients
loss.backward()          # compute new gradients
optimizer.step()         # update all parameters
```

**What is `model.parameters()`?**

`model.parameters()` returns an **iterator over all trainable tensors** in the model — all the weights and biases of every layer. The optimizer uses these to know what to update.

```python
for param in model.parameters():
    print(param.shape)
# tensor of shape [1, 30]  ← the weight of nn.Linear
# tensor of shape [1]      ← the bias of nn.Linear
```

**Common optimizers:**

| Optimizer | Class | Description |
|---|---|---|
| SGD | `torch.optim.SGD` | Classic gradient descent. Simple, reliable. |
| Adam | `torch.optim.Adam` | Adaptive learning rate. Usually trains faster. |
| RMSprop | `torch.optim.RMSprop` | Good for recurrent networks. |

---

## The `torch.nn` Module — Key Components

`torch.nn` is PyTorch's neural network library. Here's a map of what it provides:

```
torch.nn
├── Modules (Layers)
│   ├── nn.Module       → base class for all models
│   ├── nn.Linear       → fully connected layer (y = Wx + b)
│   ├── nn.Conv2d       → 2D convolutional layer (for images)
│   └── nn.LSTM         → recurrent layer (for sequences)
│
├── Activation Functions
│   ├── nn.Sigmoid      → squash to [0, 1]
│   ├── nn.ReLU         → max(0, x) — most popular for hidden layers
│   └── nn.Tanh         → squash to [-1, 1]
│
├── Loss Functions
│   ├── nn.BCELoss      → binary cross-entropy (2-class classification)
│   ├── nn.CrossEntropyLoss → multi-class classification
│   └── nn.MSELoss      → mean squared error (regression)
│
├── Containers
│   └── nn.Sequential   → stack layers in order without writing forward()
│
└── Regularization
    ├── nn.Dropout      → randomly drops neurons to prevent overfitting
    └── nn.BatchNorm2d  → normalizes layer outputs to stabilize training
```

---

## The `torch.optim` Module

`torch.optim` handles **parameter updates** during training. Key points:

- Takes `model.parameters()` so it knows which tensors to update
- Supports extra features like **learning rate scheduling** and **weight decay** (regularization)
- Replaces the manual `with torch.no_grad(): w -= lr * w.grad` pattern

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# During each training step:
optimizer.zero_grad()   # 1. clear accumulated gradients
loss.backward()         # 2. compute gradients via autograd
optimizer.step()        # 3. apply: param = param - lr × grad
```

---

## Complete Code Walkthrough — `pipeline.ipynb`

### Step 1 — Imports

```python
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

`torch.nn` is imported as `nn` — the standard convention in all PyTorch code.

---

### Step 2 — Data Preparation

Same as before: load CSV, drop unused columns, split, scale, encode labels, convert to tensors.

```python
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor  = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float()
y_test_tensor  = torch.from_numpy(y_test).float()
```

`.float()` converts from NumPy's default `float64` to PyTorch's `float32` — required because `nn.Linear` uses `float32`.

---

### Step 3 — Define the Model

```python
class MySimpleNN(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)   # 30 → 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        out = self.linear(features)   # linear combination
        out = self.sigmoid(out)       # convert to probability
        return out
```

Data flow:
```
[batch_size, 30]
      ↓  nn.Linear(30, 1)
[batch_size, 1]    ← raw score (any number)
      ↓  nn.Sigmoid()
[batch_size, 1]    ← probability between 0 and 1
```

---

### Step 4 — Set Up Training Tools

```python
learning_rate = 0.1
epochs = 25

loss_function = nn.BCELoss()
model = MySimpleNN(X_train_tensor.shape[1])   # num_features = 30
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

---

### Step 5 — Training Loop

```python
for epoch in range(epochs):

    # 1. Forward pass — get predictions
    y_pred = model(X_train_tensor)

    # 2. Compute loss — measure how wrong we are
    loss = loss_function(y_pred, y_train_tensor.view(-1, 1))

    # 3. Zero gradients — clear stale values from last epoch
    optimizer.zero_grad()

    # 4. Backward pass — compute gradients
    loss.backward()

    # 5. Update parameters — nudge weights to reduce loss
    optimizer.step()

    print(f'Epoch: {epoch + 1}, loss: {loss.item()}')
```

**Notice the order:** zero → backward → step. Always in this order.

> **Why zero gradients BEFORE backward?**
> PyTorch accumulates gradients. If you don't zero them first, the new gradients pile on top of the old ones and weights get updated with the wrong values.

---

### Step 6 — Evaluation

```python
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.9).float()    # threshold: 90% confident = malignant
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f'Accuracy: {accuracy.item()}')
```

`torch.no_grad()` disables gradient tracking during evaluation — we're not training, so we don't need gradients. This saves memory and runs faster.

---

## Full Pipeline at a Glance

```
┌──────────────────────────────────────────────────────┐
│  Raw CSV (569 rows, 32 columns)                      │
│       ↓  drop 'id' and 'Unnamed: 32'                │
│  Clean data (569 rows, 31 columns)                   │
│       ↓  train_test_split(test_size=0.2)             │
│  X_train [455×30]  /  X_test [114×30]               │
│       ↓  StandardScaler + LabelEncoder              │
│  Scaled features + numeric labels (0/1)              │
│       ↓  .float() tensors                            │
│  PyTorch float32 tensors                             │
│       ↓  MySimpleNN(nn.Module)                       │
│  nn.Linear(30, 1) + nn.Sigmoid()                    │
│       ↓  nn.BCELoss + optim.SGD                      │
│  Training loop × 25 epochs:                          │
│    forward → loss → zero_grad → backward → step      │
│       ↓                                              │
│  Evaluate on test set → accuracy                     │
└──────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|---|---|
| `pipeline.ipynb` | Full working notebook — data loading, preprocessing, model, training, evaluation |
| `NN_Module.ipynb` | Quick demo of building a model with `nn.Module` and checking its structure |

---

## Quick Reference

```python
# Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in, out)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Setup
model         = MyModel()
loss_function = nn.BCELoss()
optimizer     = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
y_pred = model(X)                        # forward
loss   = loss_function(y_pred, y)        # loss
optimizer.zero_grad()                    # zero grads
loss.backward()                          # backward
optimizer.step()                         # update
```
