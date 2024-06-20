import wandb
wandb.login()

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataprep import autoencoder_dataprep4
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from torch import nn


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
            torch.nn.Linear(50, 10), torch.nn.ELU()) # 10 might be too adventurous
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
def train_autoencoder(loader, model, epochs, learn_rate, ):
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_function = nn.MSELoss()
    epoch_losses = []  # List to store average loss per epoch

    for epoch in range(epochs):
        batch_losses = []  # List to store losses for each batch
        for (x_noisy, x_clean) in loader:
            x_noisy = x_noisy.view(-1, 304)
            x_clean = x_clean.view(-1, 304)
            optimizer.zero_grad()
            reconstructed = model(x_noisy)
            loss = loss_function(reconstructed, x_clean)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # Calculate and log the average loss of the epoch
        average_loss = sum(batch_losses) / len(batch_losses)
        wandb.log({"epoch": epoch + 1, "loss": average_loss})
        epoch_losses.append(average_loss)  # Append average loss of the epoch to the list

    print(f"Training complete. Final average loss: {average_loss:.4f}")
    return epoch_losses  # Return list of average losses per epoch

# Function to test autoencoder
def test_autoencoder(loader, model, dataset_name):
    predicted_indices = []
    actual_indices = []
    with torch.no_grad():
        for x_noisy, x_clean in loader:
            x_noisy = x_noisy.view(-1, 304)
            x_clean = x_clean.view(-1, 304)
            reconstructed = model(x_noisy)
            preds = torch.argmax(reconstructed[:, 300:304], dim=1)
            actuals = torch.argmax(x_clean[:, 300:304], dim=1)

            predicted_indices.extend(preds.cpu().numpy())
            actual_indices.extend(actuals.cpu().numpy())

    accuracy = np.mean(np.array(predicted_indices) == np.array(actual_indices))

    # Log the confusion matrix and accuracy to wandb with a label
    wandb.log({
        f"{dataset_name}_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=actual_indices,
            preds=predicted_indices,
            class_names=stimuli_names
        ),
        f"{dataset_name}_accuracy": accuracy
    })
    print(f"Accuracy for {dataset_name}: {accuracy * 100:.2f}%")
    return accuracy


# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Data loading and prep
data, noisy_data, safe_data, noisy_safe_data = autoencoder_dataprep4('./exploratory_worms_complete.csv',
                                                                     apply_undersample_glycerol=True,
                                                                     apply_noisy_features=False, # zero out 0-5.9 and 7-29.9
                                                                     apply_crazy_noisy_features=True) # zero out 6-6.9
# For the confusion matrix
stimuli_names = data.columns[300:304]

# Set epochs and learn rate
learn_rate = 1e-4
epochs = 1000

# Wandb run
run = wandb.init(
    # Set the project where this run will be logged
    project="ash_neuron-autoencoder",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learn_rate,
        "epochs": epochs,
    },
)

# Usage
dataset = DenoisingDataset(data, noisy_data)
safe_dataset = DenoisingDataset(safe_data, noisy_data)

# Dataloaders
loader = DataLoader(dataset, batch_size=len(data), shuffle=True)
safe_loader = DataLoader(safe_dataset, batch_size=len(data), shuffle=True)

model = AE() # initiating the model

# Training Loss
losses = train_autoencoder(loader, model, epochs, learn_rate)

# Accuracy
accuracy = test_autoencoder(loader, model, "train")
safe_accuracy = test_autoencoder(safe_loader, model, "test")

wandb.finish()