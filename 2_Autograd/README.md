# PyTorch Autograd

## What is Autograd?

Autograd is PyTorch's **automatic differentiation engine**. It is the core mechanism that makes training neural networks possible. Instead of manually calculating derivatives (gradients), PyTorch does it for you automatically.

In simple terms: you define a computation, run it forward, and then Autograd figures out how to compute the gradients of the output with respect to any input — all automatically.

---

## Why Do We Need Gradients?

When training a neural network, we need to minimize a **loss function**. To do that, we use an optimization algorithm (like Gradient Descent) that needs to know:

> *"If I change this weight slightly, how much does the loss change?"*

That rate of change is called a **gradient**. Autograd computes these gradients for us.

---

## The Computational Graph

Every time you perform an operation on a tensor in PyTorch, it builds a **computational graph** behind the scenes. Think of it like a flowchart that tracks:

- What tensors were used
- What operations were performed on them
- In what order the operations happened

This graph is used later to compute gradients by working **backwards** through the operations — a process called **backpropagation**.

### Forward Pass vs Backward Pass

| Phase | What Happens |
|---|---|
| **Forward Pass** | Input data flows through the network, operations are recorded, and the loss is computed |
| **Backward Pass** | Gradients are computed by traversing the graph in reverse order |

---

## `requires_grad` — Telling PyTorch What to Track

By default, PyTorch does not track operations on every tensor. You have to explicitly tell it which tensors need gradients by setting `requires_grad=True`.

- If a tensor has `requires_grad=True`, all operations involving that tensor will be tracked.
- The result of those operations will also have `requires_grad=True` automatically.
- Tensors that represent **raw input data** or **labels** typically do not need gradients.
- Tensors that represent **model parameters** (weights and biases) always need gradients.

---

## `grad_fn` — The Operation That Created a Tensor

When a tensor is created through an operation and is part of the tracked graph, PyTorch attaches a `grad_fn` to it. This is a reference to the function that created the tensor (e.g., `AddBackward`, `MulBackward`, `MeanBackward`).

- Tensors created directly by the user (leaf tensors) have `grad_fn = None`.
- Tensors produced by operations on tracked tensors will have a `grad_fn`.

---

## `leaf` Tensors vs Non-Leaf Tensors

### Leaf Tensors
A **leaf tensor** is one that was created directly — not as the result of an operation. Model weights and biases are leaf tensors. Their gradients are stored after `.backward()` is called and can be accessed via `.grad`.

### Non-Leaf Tensors
A **non-leaf tensor** is one produced by an operation (e.g., the output of a layer, the computed loss). Gradients for non-leaf tensors are typically not stored by default to save memory. You can force PyTorch to keep them using `.retain_grad()`.

---

## `.backward()` — Triggering Gradient Computation

Once you have a scalar output (like a loss value), you call `.backward()` on it. This tells PyTorch to:

1. Start from that output tensor
2. Traverse the computational graph in reverse
3. Apply the **chain rule** of calculus at each step
4. Accumulate the computed gradients into the `.grad` attribute of each leaf tensor

### The Chain Rule

Autograd is essentially a very efficient, automated application of the **chain rule** from calculus. If you have a function composed of many smaller functions, the chain rule lets you compute the overall derivative step by step through each layer.

This is exactly how gradients flow backwards through the layers of a neural network — from the loss, back through each layer, all the way to the input weights.

---

## `.grad` — Where the Gradients Are Stored

After calling `.backward()`, the gradient for each leaf tensor with `requires_grad=True` is stored in its `.grad` attribute. The optimizer then uses these gradients to update the parameters.

**Important:** Gradients **accumulate** by default. Each call to `.backward()` adds to the existing `.grad` values instead of replacing them. This is why you must call `optimizer.zero_grad()` (or manually zero out `.grad`) before each training step — to prevent gradients from building up across iterations.

---

## `torch.no_grad()` — Disabling Gradient Tracking

There are situations where you do not need gradients at all, such as:

- **Inference / Evaluation** — when you just want predictions, not training
- **Validation loops** — checking performance without updating weights

Wrapping code in `torch.no_grad()` tells PyTorch to stop building the computational graph. This:

- Saves memory (no graph is stored)
- Speeds up computation
- Is best practice for any code that does not involve training

---

## `detach()` — Separating a Tensor from the Graph

If you want to use a tensor's values but do not want it to be part of the computational graph, you can call `.detach()` on it. The returned tensor shares the same data but has no gradient history.

This is useful when:
- You want to inspect values during training without affecting the graph
- You want to use a tensor as a fixed (non-trainable) input to another part of your code

---

## Summary of Key Concepts

| Concept | Description |
|---|---|
| `requires_grad` | Marks a tensor for gradient tracking |
| Computational Graph | A dynamic record of all operations performed |
| `grad_fn` | The function that created a tensor (its history) |
| Leaf Tensor | A directly created tensor (e.g., model weights) |
| `.backward()` | Triggers backpropagation and computes gradients |
| `.grad` | Stores the computed gradient on a leaf tensor |
| `torch.no_grad()` | Disables gradient computation (for inference) |
| `.detach()` | Removes a tensor from the computational graph |
| Chain Rule | The mathematical principle Autograd applies |

---

## The Training Loop in Context

To see how Autograd fits into the bigger picture, here is the typical flow of one training iteration:

1. **Zero out gradients** — clear accumulated gradients from the previous step
2. **Forward pass** — run the input through the model, compute the loss
3. **Backward pass** — call `.backward()` to compute all gradients
4. **Optimizer step** — use the gradients to update the model weights
5. **Repeat**

Autograd handles step 3 entirely. You only need to call `.backward()` and it takes care of the rest.

---

## Dynamic vs Static Graphs

PyTorch uses a **dynamic computational graph** (also called "define-by-run"). This means:

- The graph is built fresh on every forward pass
- You can use normal Python control flow (if statements, loops) inside your model
- Debugging is much easier because the graph reflects actual runtime behavior

This is in contrast to frameworks that use **static graphs** (defined once before running), which are less flexible.

---
