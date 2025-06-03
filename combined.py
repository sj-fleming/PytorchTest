import torch
import torch.nn as nn
import torch.nn.functional as F
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np

# Set manual seed
torch.manual_seed(41)

# Load dataset, replace class names with integers
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets.replace({
    'class': {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
})

# Split dataset into training and test sets
# Note: X and y are pandas DataFrames/Series, so we need to convert them to numpy arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert to tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.LongTensor(np.array(y_train).squeeze())
y_test = torch.LongTensor(np.array(y_test).squeeze())

# Define model class
class Model(nn.Module):
    def __init__(self, input_features=4, h1=8, h2=8, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Initialize model, loss, and optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
epochs = 100
losses = []

for i in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.item()}")

# Check gradients
print("\nfc1 weight gradients:\n", model.fc1.weight.grad)

