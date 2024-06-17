# class MlpSigmoidMSE and other snippets from Sebastian Raschka @rasbt
from dataprep import mlp_load_and_preprocess_data
from dataprep import mlp4_load_and_preprocess_data  # only want to look at 4 chemicals
import torch
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    curr_mse, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 300).to(DEVICE)
            logits, probas = net.forward(features)
            probas = probas.to(torch.device('cpu'))
            loss = torch.sum((targets - probas) ** 2, dim=0)
            num_examples += targets.size(0)
            curr_mse += loss

        curr_mse = torch.mean(curr_mse / num_examples, dim=0)
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


def plot_confusion_matrix(net, data_loader, class_names):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 300).to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = net.forward(features)
            predicted_labels = torch.argmax(probas, 1)
            true_labels = torch.argmax(targets, 1)
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# Settings and device setup
RANDOM_SEED = 15
NUM_EPOCHS = 1000
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Getting data all ready and prepped/grabbing class names for confusion matrix later
X_train, y_train, X_test, y_test, X_safe, y_safe = mlp_load_and_preprocess_data('./exploratory_worms_complete.csv',
                                                                                undersample_glycerol=False)
class_names = y_train.columns.tolist()

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


# Dataset instances
train_dataset = TimeSeriesDataset(X_train.values, y_train.values)
test_dataset = TimeSeriesDataset(X_test.values, y_test.values)
safe_dataset = TimeSeriesDataset(X_safe.values, y_safe.values)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
safe_loader = DataLoader(dataset=safe_dataset, batch_size=len(safe_dataset), shuffle=False)

# Setup for the model
torch.manual_seed(RANDOM_SEED)
model = MlpSigmoidMSE(num_features=300,
                      num_hidden=70,  # need to try different ones/experiment
                      num_classes=10)  # if you use mlp4 true this has to be 4 otherwise 10

# Loss function and optimizer
model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
start_time = time.time()
epoch_cost = []
train_acc = []

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.view(-1, 300).to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)

        cost = F.mse_loss(probas, targets)
        optimizer.zero_grad()

        cost.backward()
        optimizer.step()

    # Calculate epoch cost and training accuracy
    cost = compute_mse(model, train_loader)
    epoch_cost.append(cost.item())
    train_accuracy = compute_accuracy(model, train_loader)
    train_acc.append(train_accuracy)

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

# Compute final MSE and accuracy on test and safe sets
test_mse = compute_mse(model, test_loader)
safe_mse = compute_mse(model, safe_loader)
test_accuracy = compute_accuracy(model, test_loader)
safe_accuracy = compute_accuracy(model, safe_loader)

print(f'Test MSE: {test_mse:.4f}')
print(f'Safe MSE: {safe_mse:.4f}')
print(f'Training Accuracy: {train_acc[-1]:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Safe Accuracy: {safe_accuracy:.2f}%')

# Plotting the losses and accuracies
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(range(NUM_EPOCHS), epoch_cost, label='Epoch cost', color='blue', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(range(NUM_EPOCHS), train_acc, label='Train Accuracy', color='green')
ax2.set_ylabel('Accuracy (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

fig.suptitle('Training Loss and Accuracy Over Time')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.show()

plot_confusion_matrix(model, train_loader, class_names)
plot_confusion_matrix(model, test_loader, class_names)
plot_confusion_matrix(model, safe_loader, class_names)
