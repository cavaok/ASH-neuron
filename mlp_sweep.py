import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataprep import mlp4_load_and_preprocess_data
import wandb

wandb.login()

# MLP Class
class MLP(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, dropout_rate):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(num_features, num_hidden)
        self.dropout = nn.Dropout(p=dropout_rate)  # Using dropout rate from config
        self.linear_out = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        x = F.elu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_out(x)
        return torch.sigmoid(x)

# MLP dataset class
class MLPDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return X, y

# Accuracy function
def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            logits = net(features)
            predicted_labels = torch.argmax(logits, 1)
            true_labels = torch.argmax(targets, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == true_labels).sum()
    return correct_pred.float() / num_examples * 100

# Training function for sweeps
def train():
    with wandb.init() as run:
        config = wandb.config
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data prep
        X_train, y_train, X_test, y_test = mlp4_load_and_preprocess_data('./exploratory_worms_complete.csv', test_size=0.3, random_state=config.random_seed)
        train_dataset = MLPDataset(X_train.values, y_train.values)
        test_dataset = MLPDataset(X_test.values, y_test.values)
        train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        # Setting up the model
        model = MLP(num_features=300, num_hidden=config.num_hidden, num_classes=4, dropout_rate=config.dropout)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # Training loop --> getting num of epochs from configuration at the bottom
        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            for batch_idx, (features, targets) in enumerate(train_loader):
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                logits = model(features)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Logging to wandb
            train_accuracy = compute_accuracy(model, train_loader)
            wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_loader), "train_accuracy": train_accuracy})

        # Evaluate the model
        test_accuracy = compute_accuracy(model, test_loader)
        wandb.log({"test_accuracy": test_accuracy})


sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'num_hidden': {'values': [10, 15, 20]},
        'dropout': {'values': [0.3, 0.4, 0.5]},
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.00001, 'max': 0.01},
        'epochs': {'value': 500},  # Adjusted number of epochs
        'random_seed': {'value': 42},
        #'weight_decay': {'distribution': 'uniform', 'min': 0, 'max': 0.001}  # L2 regularization
    }
}

# Initialize/run the sweep
sweep_id = wandb.sweep(sweep_config, project="ash_neuron-mlp")
wandb.agent(sweep_id, train, count=10) # adjust this for number of sweeps
