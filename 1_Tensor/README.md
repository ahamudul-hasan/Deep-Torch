# Tensors in PyTorch

## What is a Tensor?

A **tensor** is a multi-dimensional array — the fundamental building block of PyTorch and deep learning. Think of it as a generalization of numbers, lists, and tables into any number of dimensions.

| Structure   | Dimensions | Example                          |
|-------------|------------|----------------------------------|
| Scalar      | 0D         | `5.0`                            |
| Vector      | 1D         | `[1.0, 2.0, 3.0]`               |
| Matrix      | 2D         | `[[1, 2], [3, 4]]`              |
| 3D Tensor   | 3D         | RGB image                        |
| 4D Tensor   | 4D         | Batch of RGB images              |
| 5D Tensor   | 5D         | Batch of video clips             |

---

## Tensor Dimensions — With Real-World Examples

### 1. Scalar — 0D Tensor
A single number with no dimensions.

**Real-World Use:** The **loss value** after a forward pass in training — a single number that tells you how wrong your model is.

```python
import torch

loss = torch.tensor(5.0)
print(loss)        # tensor(5.)
print(loss.shape)  # torch.Size([])
print(loss.ndim)   # 0
```

---

### 2. Vector — 1D Tensor
A flat list of numbers.

**Real-World Use:** A **word embedding** — each word in a sentence is converted into a list of numbers that captures its meaning (e.g., Word2Vec, GloVe).

```python
import torch

# A word embedding vector from a pre-trained model
word_embedding = torch.tensor([0.12, -0.84, 0.33, 0.57, -0.21])
print(word_embedding)        # tensor([ 0.1200, -0.8400,  0.3300,  0.5700, -0.2100])
print(word_embedding.shape)  # torch.Size([5])
print(word_embedding.ndim)   # 1
```

---

### 3. Matrix — 2D Tensor
A grid of numbers with rows and columns.

**Real-World Use:** A **grayscale image** — each pixel is a number between 0 (black) and 255 (white).

```python
import torch

# A tiny 2x3 grayscale image (2 rows, 3 columns of pixel values)
grayscale_image = torch.tensor([
    [  0, 255, 128],
    [ 34,  90, 180]
])
print(grayscale_image)        # tensor([[  0, 255, 128], [ 34,  90, 180]])
print(grayscale_image.shape)  # torch.Size([2, 3])
print(grayscale_image.ndim)   # 2
```

---

### 4. 3D Tensor — Coloured (RGB) Images
A stack of three 2D matrices — one for each colour channel: Red, Green, Blue.

**Shape:** `[Height, Width, Channels]` or `[Channels, Height, Width]` (PyTorch convention)

**Real-World Use:** A **single RGB image**.

```python
import torch

# A small 4x4 RGB image: shape [3, 4, 4] → (Channels, Height, Width)
rgb_image = torch.randint(0, 256, (3, 4, 4))
print(rgb_image.shape)  # torch.Size([3, 4, 4])
print(rgb_image.ndim)   # 3

# Real-world size: a 256x256 RGB image
real_image = torch.zeros(3, 256, 256)
print(real_image.shape)  # torch.Size([3, 256, 256])
```

> Shape `[3, 256, 256]` means: 3 colour channels, 256 pixels tall, 256 pixels wide.

---

### 5. 4D Tensor — Batch of RGB Images
A collection of multiple RGB images grouped together for efficient parallel processing.

**Shape:** `[Batch Size, Channels, Height, Width]`

**Real-World Use:** During model training, images are processed in **batches** (groups), not one by one.

```python
import torch

# A batch of 32 RGB images, each 128x128
batch_of_images = torch.zeros(32, 3, 128, 128)
print(batch_of_images.shape)  # torch.Size([32, 3, 128, 128])
print(batch_of_images.ndim)   # 4
```

> Shape `[32, 3, 128, 128]` means: 32 images, each with 3 colour channels, 128px × 128px.

---

### 6. 5D Tensor — Video Data
Video is just a sequence of image frames over time. Adding a **time dimension** to 4D gives us 5D.

**Shape:** `[Batch Size, Frames, Channels, Height, Width]`

**Real-World Use:** A batch of **video clips** for action recognition (e.g., detecting if someone is running or jumping).

```python
import torch

# 10 video clips, each with 16 frames, 3 colour channels, 64x64 resolution
video_batch = torch.zeros(10, 16, 3, 64, 64)
print(video_batch.shape)  # torch.Size([10, 16, 3, 64, 64])
print(video_batch.ndim)   # 5
```

> Shape `[10, 16, 3, 64, 64]` means: 10 clips × 16 frames each × 3 channels × 64px × 64px.

---

## Why Are Tensors Useful?

### 1. Mathematical Operations
Tensors support all the operations needed in neural networks — addition, multiplication, dot products, matrix multiplication — and they do it efficiently.

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)       # tensor([5., 7., 9.])
print(a * b)       # tensor([ 4., 10., 18.])
print(torch.dot(a, b))  # tensor(32.)  → dot product: 1×4 + 2×5 + 3×6
```

### 2. Represent Any Real-World Data

| Data Type | Tensor Shape Example             |
|-----------|----------------------------------|
| Image     | `[3, 224, 224]`                  |
| Text      | `[Sequence Length, Embedding Size]` |
| Video     | `[Frames, 3, H, W]`              |
| Audio     | `[Channels, Samples]`            |

### 3. GPU Acceleration
Tensors can be moved to a **GPU** with one line of code, making computations orders of magnitude faster.

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = tensor.to(device)

print(f"Tensor is on: {tensor.device}")
```

---

## Where Are Tensors Used in Deep Learning?

### 1. Storing Training Data
All input data — images, text, audio — is loaded into tensors before being fed into a model.

```python
import torch

# 100 training samples, each with 20 features
training_data = torch.randn(100, 20)
print(training_data.shape)  # torch.Size([100, 20])
```

### 2. Weights and Biases
The **learnable parameters** of a neural network are stored as tensors. During training, these tensors are updated to minimise the loss.

```python
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=2)
print(layer.weight)         # Tensor of shape [2, 4]
print(layer.weight.shape)   # torch.Size([2, 4])
print(layer.bias.shape)     # torch.Size([2])
```

### 3. Matrix Operations
The core of a neural network is **matrix multiplication** — multiplying input tensors by weight tensors to produce outputs.

```python
import torch

# Input: 1 sample with 4 features
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape [1, 4]

# Weight matrix: maps 4 features → 2 outputs
W = torch.randn(4, 2)                       # shape [4, 2]

# Matrix multiplication (the core of a neural network layer)
output = x @ W                              # shape [1, 2]
print(output.shape)  # torch.Size([1, 2])
```

### 4. The Training Process

```
Input Data (tensor)
       ↓
  Forward Pass  → predictions flow through the network as tensors
       ↓
  Loss Function → computes a scalar tensor (the error)
       ↓
 Backward Pass  → gradients are computed and stored as tensors
       ↓
  Optimiser     → updates weight tensors to reduce the loss
```

```python
import torch

# Simple example: predict y = 2x
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(1.0, requires_grad=True)  # learnable weight

# Forward pass
y_pred = w * x

# Loss (Mean Squared Error)
loss = ((y_pred - y_true) ** 2).mean()
print(f"Loss: {loss.item():.4f}")

# Backward pass — compute gradients
loss.backward()
print(f"Gradient of w: {w.grad.item():.4f}")  # tells us which direction to adjust w
```

---

## Quick Reference — Tensor Shapes

```
Scalar          →  shape: []            →  a single number
Vector          →  shape: [N]           →  N numbers in a line
Matrix          →  shape: [H, W]        →  H rows, W columns
RGB Image       →  shape: [C, H, W]     →  C=3 channels, H height, W width
Image Batch     →  shape: [B, C, H, W]  →  B images
Video Batch     →  shape: [B, T, C, H, W] →  B clips, T frames each
```

---

## Summary

- A **tensor** is an n-dimensional array optimised for computation.
- Scalars, vectors, and matrices are all special cases of tensors.
- Tensors power everything in deep learning: data storage, model parameters, and gradient computation.
- PyTorch tensors can run on **CPUs or GPUs**, making large-scale training feasible.
