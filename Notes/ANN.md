# Artificial Neural Network (ANN) Pipeline Explained

This document explains the full pipeline of training an Artificial Neural Network (ANN) using a dataset stored in a CSV file.

Pipeline Overview:

```
CSV File
   ↓
Load with Pandas
   ↓
Split features (X) and labels (y)
   ↓
Train / Test Split (80/20)
   ↓
Normalize pixel values (÷255)
   ↓
CustomDataset → DataLoader (batch size 32)
   ↓
Define ANN Architecture
   ↓
Training Loop (Epochs)
   ↓
Evaluate on Test Set
```

---

# 1. CSV File

The dataset is stored in a **CSV file**.

Example dataset:

| label | pixel1 | pixel2 | pixel3 | ... |
|------|------|------|------|------|
| 7 | 0 | 255 | 120 | ... |
| 2 | 45 | 90 | 12 | ... |

- **Label** → Correct answer
- **Pixels / Features** → Input data

Goal:
Store the dataset in a structured format.

---

# 2. Load Data with Pandas

We load the CSV file using pandas.

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

Goal:
Convert the CSV file into a **dataframe** that Python can easily process.

---

# 3. Split Features (X) and Labels (y)

The dataset contains both inputs and outputs.

```python
X = df.iloc[:,1:].values
y = df.iloc[:,0].values
```

- **X** → input features
- **y** → correct labels

Example:

```
X = image pixels
y = digit number
```

Goal:
Separate **input data** from **target values**.

---

# 4. Train/Test Split (80/20)

We split the dataset into two parts.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **Training data (80%)** → used to train the model
- **Testing data (20%)** → used to evaluate the model

Goal:
Check if the model performs well on **new unseen data**.

---

# 5. Normalize Pixel Values

Pixel values range from **0 to 255**.

Neural networks work better with smaller numbers, so we normalize them.

```python
X_train = X_train / 255
X_test = X_test / 255
```

Example:

```
255 → 1.0
128 → 0.5
64  → 0.25
```

Goal:
Make training **faster and more stable**.

---

# 6. Custom Dataset and DataLoader

We create a custom dataset class.

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.X = features
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
```

Then we create DataLoaders.

```python
from torch.utils.data import DataLoader

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
```

Goal:

- Load data in **small batches**
- Reduce memory usage
- Speed up training

---

# 7. Define ANN Architecture

We define the neural network model.

```python
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)
```

Layers:

```
Input Layer
   ↓
Hidden Layer
   ↓
Hidden Layer
   ↓
Output Layer
```

Goal:
Create the **structure of the neural network**.

---

# 8. Training Setup

We define the loss function and optimizer.

```python
import torch.optim as optim

model = Model(X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Components:

- **Loss Function** → measures prediction error
- **Optimizer** → updates model weights

Goal:
Prepare the model for training.

---

# 9. Training Loop

The training loop is where the model learns.

```python
epochs = 100

for epoch in range(epochs):

    for batch_features, batch_labels in train_loader:

        outputs = model(batch_features)

        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

Steps inside the loop:

1. Model makes predictions
2. Calculate error (loss)
3. Backpropagation computes gradients
4. Optimizer updates weights

Goal:
Gradually improve the model predictions.

---

# 10. Evaluate on Test Set

After training, we check the model performance.

```python
correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:

        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

print("Test Accuracy:", accuracy)
```

Example result:

```
Test Accuracy: 0.97
```

Goal:
Measure how well the model works on **unseen data**.

---

# Final Goal of the Pipeline

The complete pipeline helps us:

```
Raw Data
   ↓
Prepare Data
   ↓
Train Neural Network
   ↓
Learn Patterns
   ↓
Predict New Data
```

Artificial Neural Networks can be used for:

- Image recognition
- Handwritten digit recognition
- Speech recognition
- Medical diagnosis
- Self-driving cars

The ultimate goal is to build a model that can **learn patterns from data and make accurate predictions automatically**.