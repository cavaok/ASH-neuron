import torch
from scipy.stats import wasserstein_distance

def custom_accuracy(x_clean, reconstructed, threshold=0.1):
    with torch.no_grad():
        correct = (torch.abs(x_clean - reconstructed) < threshold).float()
        return correct.mean()

def wasserstein_accuracy(output, target, max_distance=10):
    batch_size = output.shape[0]
    distances = []
    for i in range(batch_size):
        dist = wasserstein_distance(output[i].detach().cpu().numpy(), target[i].detach().cpu().numpy())
        distances.append(dist)
    # Normalize distances by a chosen max_distance that makes sense for your use case
    normalized_distances = torch.tensor(distances) / max_distance
    # Ensure no value exceeds 1
    normalized_distances = torch.clamp(normalized_distances, 0, 1)
    # Invert distances to convert to accuracy-like metric
    accuracy = 1 - normalized_distances.mean()
    return accuracy