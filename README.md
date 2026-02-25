<div align="center">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="PyTorch Logo" width="200"/>

  # PyTorch Learning Journey

  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

  *A structured deep dive into PyTorch — from fundamentals to distributed training.*
</div>

---

## Overview

This repository documents a progressive study of **PyTorch**, one of the most powerful and flexible deep learning frameworks. The curriculum covers core computational concepts, hardware acceleration, automatic differentiation, and production-scale distributed training.

---

## Syllabus

### 1. Tensor Computations
> The foundation of all PyTorch operations.

- Creating and manipulating tensors
- Tensor operations: arithmetic, slicing, reshaping, broadcasting
- Data types and device management
- Comparison with NumPy arrays

---

### 2. GPU Acceleration
> Leveraging hardware for high-performance computation.

- Moving tensors between CPU and GPU (`.to(device)`, `.cuda()`)
- Writing device-agnostic code
- Performance benchmarking: CPU vs GPU
- Memory management and optimization

---

### 3. Dynamic Computation Graph
> Understanding PyTorch's define-by-run paradigm.

- What is a computation graph?
- Static (TensorFlow) vs Dynamic (PyTorch) graphs
- How PyTorch builds graphs at runtime
- Practical implications for model design and debugging

---

### 4. Automatic Differentiation
> Autograd — the engine behind neural network training.

- `torch.autograd` fundamentals
- Forward and backward passes
- Gradient accumulation and zeroing
- Custom gradient functions with `torch.autograd.Function`
- Gradient tracking: `requires_grad`, `detach()`, `no_grad()`

---

### 5. Distributed Training
> Scaling training across multiple devices and machines.

- `torch.distributed` package overview
- Data Parallelism: `DataParallel` vs `DistributedDataParallel`
- Process groups and communication backends (NCCL, Gloo)
- Model parallelism strategies
- Multi-node training setup

---

### 6. Interoperability with Other Libraries
> PyTorch in the broader ML ecosystem.

- **NumPy**: Tensor ↔ ndarray conversions
- **ONNX**: Exporting models for cross-framework deployment
- **Hugging Face Transformers**: Using pretrained models
- **TorchScript**: Serializing and deploying PyTorch models
- **scikit-learn**: Bridging classical ML with deep learning pipelines

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| CUDA (optional) | ≥ 11.7 |

Install PyTorch:
```bash
# CPU only
pip install torch torchvision torchaudio

# With CUDA support (adjust cu121 to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
---