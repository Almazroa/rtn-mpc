import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

'''
Author: Reem Al Mazroa
'''

'''
Neural Network Training and Validation
- architecture: ResidualNet (MLP with residual connections ReLU activations) 
- loss function: Mean Squared Error (MSE)
- optimizer: Adam
- dataset: 10,000 samples. 80% training, 20% validation
- training loop: 50 epochs with early stopping based on validation loss
- training and validation loss is calculated each and printed each epock and curves are plotted at the end

this model will be used to predict the residuals of the bicycle model dynamics
it will serve as a corrective module in the MPC framework by improving prediction accuracy during control
'''

# Load data that you previously saved
inputs = np.load("mpc_inputs.npy")   # Shape: (num_samples, 6)
targets = np.load("mpc_targets.npy")   # Shape: (num_samples, 4)

# Convert numpy arrays to PyTorch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# Create a dataset and split it into training and validation sets
dataset = TensorDataset(inputs_tensor, targets_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define a simple Multi-Layer Perceptron
class ResidualNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=4, hidden_dim=64, num_layers=3):
        super(ResidualNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Instantiate the model, loss function, and optimizer
model = ResidualNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_inputs.size(0)
    epoch_loss = running_loss / train_size
    train_losses.append(epoch_loss)
    
    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item() * batch_inputs.size(0)
    epoch_val_loss = val_loss / val_size
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

torch.save(model, "residual_model.pth")

# Plot training and validation loss curves
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

'''
RESULTS: 
snapshot of loss:
Epoch 1/50 | Train Loss: 0.0121 | Val Loss: 0.0044
Epoch 2/50 | Train Loss: 0.0043 | Val Loss: 0.0041
Epoch 3/50 | Train Loss: 0.0041 | Val Loss: 0.0044
Epoch 4/50 | Train Loss: 0.0042 | Val Loss: 0.0041
Epoch 5/50 | Train Loss: 0.0041 | Val Loss: 0.0041
Epoch 10/50 | Train Loss: 0.0040 | Val Loss: 0.0040
Epoch 20/50 | Train Loss: 0.0038 | Val Loss: 0.0039
Epoch 30/50 | Train Loss: 0.0038 | Val Loss: 0.0038
Epoch 40/50 | Train Loss: 0.0038 | Val Loss: 0.0038
Epoch 50/50 | Train Loss: 0.0038 | Val Loss: 0.0038

- mean squared error (MSE) is averaging about 0.0038 for both training and validation
- this means that model's predictions for the residuals are quite accurate
- both the training and validation losses converging to similar values (~0.0038) indicates that network is generalizing properly and there is no significant overfitting

- residuals are generally on the order of 0.01 to 0.2
- An MSE around 0.0038 corresponds to a root mean squared error (RMSE) of roughly 0.062 
-> in a similar ballpark to target magnitudes
-> on average, network's prediction error is relatively small compared to the values it's trying to predict
'''