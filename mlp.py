import torch
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataprep import mlp_load_and_preprocess_data
from dataprep import mlp4_load_and_preprocess_data


# Define the neural network class
class MlpSigmoidMSE(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(MlpSigmoidMSE, self).__init__()

        self.num_classes = num_classes

        # 1st hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()

        # Output layer
        self.linear_out = torch.nn.Linear(num_hidden, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):
        out = self.linear_1(x)
        out = torch.sigmoid(out)
        logits = self.linear_out(out)
        probas = torch.sigmoid(logits)
        return logits, probas

# Data preparation function
def prepare_data(file_path, undersample_glycerol=False):
    X_train, y_train, X_test, y_test, X_safe, y_safe = mlp_load_and_preprocess_data(file_path, test_size=0.2, safe_data_fraction=0.1, undersample_glycerol=undersample_glycerol)
    return X_train, y_train, X_test, y_test, X_safe, y_safe

# Dataset class
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

# MSE computation function
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

# Accuracy function
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

# Function that plots the confusion matrix
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

# Train the model function
def train_mlp(model, train_loader, num_epochs, optimizer):
    epoch_cost = []
    train_acc = []
    start_time = time.time()

    for epoch in range(num_epochs):
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
    return epoch_cost, train_acc

# Evaluates the model
def evaluate_model(model, data_loader):
    mse = compute_mse(model, data_loader)
    accuracy = compute_accuracy(model, data_loader)
    return mse, accuracy

# Usage
RANDOM_SEED = 15
NUM_EPOCHS = 1000
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

file_path = './exploratory_worms_complete.csv'
X_train, y_train, X_test, y_test, X_safe, y_safe = prepare_data(file_path, undersample_glycerol=True)
class_names = y_train.columns.tolist()

# Datasets
train_dataset = TimeSeriesDataset(X_train.values, y_train.values)
test_dataset = TimeSeriesDataset(X_test.values, y_test.values)
safe_dataset = TimeSeriesDataset(X_safe.values, y_safe.values)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
safe_loader = DataLoader(dataset=safe_dataset, batch_size=len(safe_dataset), shuffle=False)

# Model setup
torch.manual_seed(RANDOM_SEED)
model = MlpSigmoidMSE(num_features=300, num_hidden=70, num_classes=10)
model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
epoch_cost, train_acc = train_mlp(model, train_loader, NUM_EPOCHS, optimizer)

# Evaluating
test_mse, test_accuracy = evaluate_model(model, test_loader)
safe_mse, safe_accuracy = evaluate_model(model, safe_loader)

print(f'Test MSE: {test_mse:.4f}')
print(f'Safe MSE: {safe_mse:.4f}')
print(f'Training Accuracy: {train_acc[-1]:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Safe Accuracy: {safe_accuracy:.2f}%')

# Plotting
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
