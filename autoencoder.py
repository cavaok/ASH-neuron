import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataprep import autoencoder_dataprep4
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Data loading and prep
data, noisy_data, safe_data, noisy_safe_data = autoencoder_dataprep4('./exploratory_worms_complete.csv',
                                                                     undersample_glycerol=True,
                                                                     noisy_features=False, # zero out 0-5.9 and 7-29.9
                                                                     crazy_noisy_features=True) # zero out 6-6.9
# For the confusion matrix
stimuli_names = data.columns[300:304]

# Dataset class
class DenoisingDataset(Dataset):
    def __init__(self, clean_data, noisy_data):
        self.clean_data = clean_data
        self.noisy_data = noisy_data

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        x_noisy = self.noisy_data.iloc[idx].values.astype('float32')
        x_clean = self.clean_data.iloc[idx].values.astype('float32')
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# Autoencoder class
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(304, 200), torch.nn.ELU(),
            torch.nn.Linear(200, 100), torch.nn.ELU(),
            torch.nn.Linear(100, 50), torch.nn.ELU(),
            torch.nn.Linear(50, 10), torch.nn.ELU())
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 50), torch.nn.ELU(),
            torch.nn.Linear(50, 100), torch.nn.ELU(),
            torch.nn.Linear(100, 200), torch.nn.ELU(),
            torch.nn.Linear(200, 304))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Trains the autoencoder function
def train_autoencoder(loader, model, epochs):
    learn_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_function = torch.nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        for (x_noisy, x_clean) in loader:
            x_noisy = x_noisy.view(-1, 304)
            x_clean = x_clean.view(-1, 304)
            optimizer.zero_grad()
            reconstructed = model(x_noisy)
            loss = loss_function(reconstructed, x_clean)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses

# Function to test autoencoder
def test_autoencoder(loader, model):
    predicted_labels = []
    actual_labels = []
    with torch.no_grad():
        for x_noisy, x_clean in loader:
            x_noisy = x_noisy.view(-1, 304)
            reconstructed = model(x_noisy)
            predicted_indices = torch.argmax(reconstructed[:, 300:304], dim=1)
            actual_indices = torch.argmax(x_clean[:, 300:304], dim=1)
            predicted_labels.extend(stimuli_names[predicted_indices])
            actual_labels.extend(stimuli_names[actual_indices])
    accuracy = sum(1 for x, y in zip(predicted_labels, actual_labels) if x == y) / len(predicted_labels)
    cm = confusion_matrix(actual_labels, predicted_labels, labels=stimuli_names)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    plot_confusion_matrix(cm, stimuli_names)
    return accuracy

# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Function to plot loss
def plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='MSE Loss')
    plt.title('MSE Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Usage

dataset = DenoisingDataset(data, noisy_data)
safe_dataset = DenoisingDataset(safe_data, noisy_data)

# Dataloaders
loader = DataLoader(dataset, batch_size=len(data), shuffle=True)
safe_loader = DataLoader(safe_dataset, batch_size=len(data), shuffle=True)

model = AE() # initiating the model

# Loss and accuracy
losses = train_autoencoder(loader, model, 1000)
plot_loss(losses)
accuracy = test_autoencoder(loader, model)

safe_dataset = DenoisingDataset(safe_data, noisy_safe_data)
safe_loader = DataLoader(safe_dataset, batch_size=len(safe_data), shuffle=False)
safe_accuracy = test_autoencoder(safe_loader, model)
