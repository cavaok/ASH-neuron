# Some code structure and snippets from Sebastian Raschka @rasbt

from dataprep import mlp_load_and_preprocess_data
import torch
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MlpSigmoidMSE(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(MlpSigmoidMSE, self).__init__()

        self.num_classes = num_classes

        ### 1st hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()

        ### Output layer
        self.linear_out = torch.nn.Linear(num_hidden, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):
        out = self.linear_1(x)
        out = torch.sigmoid(out)
        logits = self.linear_out(out)
        probas = torch.sigmoid(logits)
        return logits, probas


def compute_mse(net, data_loader):
    curr_mse, num_examples = torch.zeros(model.num_classes).float(), 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 300).to(DEVICE)
            logits, probas = net.forward(features)
            probas = probas.to(torch.device('cpu'))
            loss = torch.sum((targets - probas)**2, dim=0)
            num_examples += targets.size(0)
            curr_mse += loss

        curr_mse = torch.mean(curr_mse/num_examples, dim=0)
        return curr_mse

def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 300).to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = net.forward(features)
            predicted_labels = torch.argmax(probas, 1)
            true_labels = torch.argmax(targets, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == true_labels).sum()
        return correct_pred.float() / num_examples * 100


# Settings and device setup
RANDOM_SEED = 1
BATCH_SIZE = 255
NUM_EPOCHS = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Getting data all ready and prepped
X_train, y_train, X_test, y_test, X_safe, y_safe = mlp_load_and_preprocess_data('./exploratory_worms_complete.csv')

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return X, y


# Create dataset instances
train_dataset = TimeSeriesDataset(X_train.values, y_train.values)
test_dataset = TimeSeriesDataset(X_test.values, y_test.values)

# Create dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
torch.manual_seed(RANDOM_SEED)
model = MlpSigmoidMSE(num_features=300,
                      num_hidden=100, # need to try different ones/experiment
                      num_classes=10)

# Loss function and optimizer
model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
start_time = time.time()
minibatch_cost = []
epoch_cost = []
train_acc = []
test_acc = []

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.view(-1, 300).to(DEVICE)  # Adjusting to 300 features
        targets = targets.to(DEVICE)  # Targets are already one-hot encoded

        ### FORWARD AND BACK PROP
        logits, probas = model(features)

        cost = F.mse_loss(probas, targets)
        optimizer.zero_grad()

        cost.backward()
        minibatch_cost.append(cost.item())
        ### UPDATE MODEL PARAMETERS
        optimizer.step()

    cost = compute_mse(model, train_loader)
    epoch_cost.append(cost.item())

    train_accuracy = compute_accuracy(model, train_loader)
    test_accuracy = compute_accuracy(model, test_loader)
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

# Compute final MSE on test set
test_mse = compute_mse(model, test_loader)
print(f'Test MSE: {test_mse:.4f}')

# Compute final accuracy on training and test sets
train_accuracy = compute_accuracy(model, train_loader)
test_accuracy = compute_accuracy(model, test_loader)
print(f'Training Accuracy: {train_accuracy:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Plotting the losses and accuracies with Matplotlib
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(range(NUM_EPOCHS), epoch_cost, label='Epoch cost', color='blue', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(range(NUM_EPOCHS), train_acc, label='Train Accuracy', color='green')
ax2.plot(range(NUM_EPOCHS), test_acc, label='Test Accuracy', color='orange')
ax2.set_ylabel('Accuracy (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

fig.suptitle('Training Loss and Accuracy Over Time')
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()