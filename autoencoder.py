import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataprep import autoencoder_dataprep
import wandb
from accuracy import wasserstein_accuracy

# Note: data and noisy_data have shape (355, 310)

# Data loading and prep
data, noisy_data = autoencoder_dataprep('./exploratory_worms_complete.csv')

# Class for denoising the dataset - - - - - - - - - - - - - - - - - - - - - - - - -
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

dataset = DenoisingDataset(clean_data=data, noisy_data=noisy_data) # creates dataset
loader = DataLoader(dataset, batch_size=355, shuffle=True) # creates loader for training

# Class for the autoencoder - - - - - - - - - - - - - - - - - - - - - - - - - -
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(310, 250),
            torch.nn.ELU(),
            torch.nn.Linear(250, 200),
            torch.nn.ELU(),
            torch.nn.Linear(200, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 30),
            torch.nn.ELU(),
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(30, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 200),
            torch.nn.ELU(),
            torch.nn.Linear(200, 250),
            torch.nn.ELU(),
            torch.nn.Linear(250, 310)
        )

    # Stepper
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Running the model - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
learn_rate = 1e-3 # setting learn rate
epochs = 300 # setting epochs
outputs = []
losses = []
model = AE() # initializes the model
loss_function = torch.nn.MSELoss() # initialize the loss function
optimizer = torch.optim.RMSprop(model.parameters(), lr=learn_rate)

run = wandb.init(
    project="ASH-neuron",
    config={
        "learning_rate": learn_rate,
        "epochs": epochs,
    },
)

for epoch in range(epochs):
    for (x_noisy, x_clean) in loader:
        x_noisy = x_noisy.view(-1, 310)
        x_clean = x_clean.view(-1, 310)

        optimizer.zero_grad()
        reconstructed = model(x_noisy)
        loss = loss_function(reconstructed, x_clean)
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss.item())
        acc = wasserstein_accuracy(x_clean, reconstructed)
        wandb.log({"accuracy": acc, "loss": loss})


