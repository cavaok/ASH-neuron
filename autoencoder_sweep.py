import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
from dataprep import autoencoder_dataprep4  # Assuming this is correctly defined elsewhere

# Initialize Weights & Biases
wandb.login()


class DenoisingDataset(Dataset):
    def __init__(self, clean_data, noisy_data):
        self.clean_data = clean_data
        self.noisy_data = noisy_data
        self.num_features = clean_data.shape[1]

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        x_noisy = self.noisy_data.iloc[idx].values.astype('float32')
        x_clean = self.clean_data.iloc[idx].values.astype('float32')
        return torch.tensor(x_noisy), torch.tensor(x_clean)


class AE(nn.Module):
    def __init__(self, num_features, num_hidden_units):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden_units), nn.ELU(),
            nn.Linear(num_hidden_units, num_hidden_units // 2), nn.ELU(),
            nn.Linear(num_hidden_units // 2, 10), nn.ELU())
        self.decoder = nn.Sequential(
            nn.Linear(10, num_hidden_units // 2), nn.ELU(),
            nn.Linear(num_hidden_units // 2, num_hidden_units), nn.ELU(),
            nn.Linear(num_hidden_units, num_features))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(config, loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    epoch_losses = []
    for epoch in range(config.epochs):
        batch_losses = []
        for (x_noisy, x_clean) in loader:
            x_noisy = x_noisy.view(-1, loader.dataset.num_features)
            x_clean = x_clean.view(-1, loader.dataset.num_features)
            optimizer.zero_grad()
            reconstructed = model(x_noisy)
            loss = loss_function(reconstructed, x_clean)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        average_loss = sum(batch_losses) / len(batch_losses)
        wandb.log({"epoch": epoch + 1, "loss": average_loss})
        epoch_losses.append(average_loss)
    print(f"Training complete. Final average loss: {average_loss:.4f}")
    return epoch_losses


def test_autoencoder(loader, model, dataset_name):
    predicted_indices = []
    actual_indices = []
    with torch.no_grad():
        for x_noisy, x_clean in loader:
            x_noisy = x_noisy.view(-1, loader.dataset.num_features)
            x_clean = x_clean.view(-1, loader.dataset.num_features)
            reconstructed = model(x_noisy)
            preds = torch.argmax(reconstructed[:, -4:], dim=1)
            actuals = torch.argmax(x_clean[:, -4:], dim=1)

            predicted_indices.extend(preds.cpu().numpy())
            actual_indices.extend(actuals.cpu().numpy())

    accuracy = np.mean(np.array(predicted_indices) == np.array(actual_indices))

    # Log the confusion matrix and accuracy to wandb with a label
    wandb.log({
        f"{dataset_name}_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=actual_indices,
            preds=predicted_indices,
            class_names=["Class1", "Class2", "Class3", "Class4"]  # Adjust class names accordingly
        ),
        f"{dataset_name}_accuracy": accuracy
    })
    print(f"Accuracy for {dataset_name}: {accuracy * 100:.2f}%")
    return accuracy


def sweep_train():
    with wandb.init() as run:
        config = run.config
        dataset = DenoisingDataset(data, noisy_data)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        model = AE(num_features=dataset.num_features, num_hidden_units=config.num_hidden_units)
        train_autoencoder(config, loader, model)
        test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        test_autoencoder(test_loader, model, "validation")


data, noisy_data, _, _ = autoencoder_dataprep4('./exploratory_worms_complete.csv', apply_undersample_glycerol=True)

sweep_config = {
    'method': 'random',
    'metric': {'name': 'loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-3, 'distribution': 'log_uniform_values'},
        'num_hidden_units': {'values': [50, 100, 150]},
        'epochs': {'value': 500}
    }
}

sweep_id = wandb.sweep(sweep_config, project="ash_neuron-autoencoder_sweeps")
wandb.agent(sweep_id, sweep_train)
