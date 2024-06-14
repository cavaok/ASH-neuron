import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep import mlp_load_and_preprocess_data

# Loading and preprocessing the data
X_train, y_train, X_test, y_test, X_safe, y_safe = mlp_load_and_preprocess_data('./exploratory_worms_complete.csv')

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_safe = scaler.transform(X_safe)

# Function to convert any DataFrame or NumPy array to a PyTorch tensor (not sure if needed
# to be set up like this but put this together while debugging...
def convert_to_tensor(data, unsqueeze=False):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()  # Convert DataFrame to NumPy array if necessary
    tensor = torch.tensor(data, dtype=torch.float32)
    if unsqueeze:
        tensor = tensor.unsqueeze(1)  # Add dimension for PyTorch compatibility if required
    return tensor

# Converting and using the function
X_train = convert_to_tensor(X_train)
y_train = convert_to_tensor(y_train)
X_test = convert_to_tensor(X_test)
y_test = convert_to_tensor(y_test)
X_safe = convert_to_tensor(X_safe)
y_safe = convert_to_tensor(y_safe)

# Check shapes
print("Shapes:", X_train.shape, y_train.shape, X_safe.shape, y_safe.shape)

# Dataloaders (need to change batch size)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
safe_loader = DataLoader(TensorDataset(X_safe, y_safe), batch_size=64, shuffle=False)

# MLP class
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Dropout(0.01),  # Optionally retain dropout as a form of regularization
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        return self.network(x)


# Model, loss fcn, and optimizer
model = MLP(input_size=X_train.shape[1], output_size=y_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training the model
epoch_losses = []
num_epochs = 300
for epoch in range(num_epochs):
    total_loss = 0
    count_batches = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count_batches += 1
    average_loss = total_loss / count_batches
    epoch_losses.append(average_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')


# Evaluate the Model on Test and Safe Data using MSE
def evaluate_model(data_loader):
    model.eval()
    total_loss = 0
    total_samples = 0  # Keep track of the total number of samples
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size
            total_samples += inputs.size(0)
    average_loss = total_loss / total_samples
    return average_loss

# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label='Training Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

print("Test Data MSE:", evaluate_model(test_loader))
print("Safe Data MSE:", evaluate_model(safe_loader))
