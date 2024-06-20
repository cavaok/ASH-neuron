import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataprep import mlp4_load_and_preprocess_data
import wandb
wandb.login()


# MLP Class
class MLP(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(MLP, self).__init__()
        self.num_classes = num_classes

        # 1st hidden layer
        self.linear_1 = nn.Linear(num_features, num_hidden)
        #self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% drop probability
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()

        # Output layer
        self.linear_out = nn.Linear(num_hidden, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):
        out = self.linear_1(x)
        out = F.elu(out)  # Apply ELU activation
        #out = self.dropout(out)  # Apply dropout
        logits = self.linear_out(out)
        probas = torch.sigmoid(logits)  # Apply sigmoid to the output layer
        return probas


# Dataset class
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
            features = features.view(-1, num_features).to(DEVICE)
            targets = targets.to(DEVICE)
            logits = net(features)
            predicted_labels = torch.argmax(logits, 1)
            true_labels = torch.argmax(targets, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == true_labels).sum()
    return correct_pred.float() / num_examples * 100


# Training function
def train_mlp(model, train_loader, num_epochs, optimizer):
    mse_loss = torch.nn.MSELoss()
    epoch_cost = []
    train_acc = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.view(-1, num_features).to(DEVICE)
            if targets.ndim == 1:
                targets = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=num_classes)
            targets = targets.float().to(DEVICE)
            logits = model(features)
            loss = mse_loss(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_accuracy = compute_accuracy(model, train_loader)
        epoch_cost.append(average_loss)
        train_acc.append(train_accuracy)

        wandb.log({"epoch": epoch, "train_loss": average_loss, "train_accuracy": train_accuracy})

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    return epoch_cost, train_acc


# Test function
def test_mlp(model, data_loader, dataset_name):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, num_features).to(DEVICE)
            if targets.ndim == 1:
                targets = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=num_classes)
            targets = targets.float().to(DEVICE)
            logits = model(features)
            predicted_labels = torch.argmax(logits, dim=1)
            true_labels = torch.argmax(targets, dim=1)
            all_preds.extend(predicted_labels.cpu().numpy())
            all_targets.extend(true_labels.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() * 100
    wandb.log({
        f"{dataset_name}_confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds,
            y_true=all_targets,
            class_names=class_names),
        f"{dataset_name}_accuracy": accuracy
    })

    print(f"{dataset_name} - Accuracy: {accuracy:.2f}%")
    return accuracy


# Usage - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data preparation
X_train, y_train, X_test, y_test = mlp4_load_and_preprocess_data('./exploratory_worms_complete.csv', test_size=0.3, undersample_glycerol=True)

class_names = y_train.columns.tolist()
num_classes = 4  # Ensure this matches the dataprep fcn choice

# Settings
rand_seed = 15
epochs = 1000
learn_rate = 1e-4
num_hidden = 15
num_features = 300

# Initialize Weights & Biases run
run = wandb.init(
    project="ash_neuron-mlp",
    config={
        "learning_rate": learn_rate,
        "epochs": epochs,
        "random_seed": rand_seed,
        "hidden_layers": num_hidden,
    },
)

# Datasets
train_dataset = MLPDataset(X_train.values, y_train.values)
test_dataset = MLPDataset(X_test.values, y_test.values)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

# Model setup
torch.manual_seed(rand_seed)
model = MLP(num_features=num_features, num_hidden=num_hidden, num_classes=num_classes)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# Training mlp
epoch_cost, train_acc = train_mlp(model, train_loader, epochs, optimizer)

# Testing mlp
train_accuracy = test_mlp(model, train_loader, "train")
test_accuracy = test_mlp(model, test_loader, "test")

wandb.finish()
