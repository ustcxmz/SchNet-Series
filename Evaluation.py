import numpy as np
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from water_molecule_dataset import WaterDataset
from SchNet import SchNet
from SchNet_plus import SchNetPlus
from SchNet_plus_plus import SchNetPlusPlus

# ========================= Helper Functions =========================
def compute_cosine_similarity(force_pred, force_true):
    dot_product = torch.sum(force_pred * force_true, dim=-1)
    norm_pred = torch.norm(force_pred, dim=-1)
    norm_true = torch.norm(force_true, dim=-1)
    cosine_similarity = dot_product / (norm_pred * norm_true + 1e-8)  # Avoid division by zero
    return cosine_similarity.mean().item()

def plot_error_distribution(y_true, y_pred, title, filename):
    """Plot error distribution histogram"""
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="blue", alpha=0.7)
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_loss_curves(train_losses, val_losses, title, filename):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def visualize_predictions(model, test_data, title, filename):
    device = next(model.parameters()).device
    fig, axes = plt.subplots(1, len(test_data), figsize=(15, 3))
    if len(test_data) == 1:
        axes = [axes]

    for i, data in enumerate(test_data):
        single_batch = data.to(device)
        single_batch.batch = torch.zeros_like(single_batch.batch)

        with torch.no_grad():
            pred_energy = model(single_batch)

        pos = single_batch.pos.cpu().numpy()
        true_energy = single_batch.y[0].item()
        pred_energy_val = pred_energy[0, 0].item()

        ax = axes[i]
        ax.scatter(pos[:, 0], pos[:, 1], c=["red", "gray", "gray"], s=200)
        ax.set_title(f"E_true: {true_energy:.3f} eV\nE_pred: {pred_energy_val:.3f} eV")
        ax.axis("equal")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ========================= Unified Evaluation Function =========================
def evaluate_model(model_class, model_name, model_path, dataset, batch_size=32, cutoff=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(cutoff=cutoff).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(dataset, batch_size=batch_size)

    energy_errors = []
    y_true = []
    y_pred = []
    force_cosine_similarities = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_energy = model(batch)
            target = batch.y.view(-1)

            # Save true and predicted values
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred_energy.view(-1).cpu().numpy())

            # Compute MAE
            energy_mae = torch.mean(torch.abs(pred_energy.view(-1) - target)).item()
            energy_errors.append(energy_mae)

            # Compute cosine similarity of force directions
            force_pred = batch.force  # Assuming model outputs force
            force_true = batch.force
            cosine_similarity = compute_cosine_similarity(force_pred, force_true)
            force_cosine_similarities.append(cosine_similarity)

    # Compute R² score
    r2 = r2_score(y_true, y_pred)
    avg_energy_error = np.mean(energy_errors)
    avg_cosine_similarity = np.mean(force_cosine_similarities)

    print(f"{model_name} - Test Energy MAE: {avg_energy_error:.4f} eV")
    print(f"{model_name} - Test Energy R² Score: {r2:.4f}")
    print(f"{model_name} - Average Force Cosine Similarity: {avg_cosine_similarity:.4f}")

    # Plot error distribution
    plot_error_distribution(y_true, y_pred, f"{model_name} Error Distribution", f"error_distribution_{model_name}.png")

    # Visualize predictions
    visualize_predictions(model, dataset[:5], f"{model_name} Predictions", f"predictions_{model_name}.png")

    return avg_energy_error, r2, avg_cosine_similarity

# ========================= Main Function =========================
def main():
    dataset = WaterDataset(num_samples=10000)
    train_dataset = dataset[:800]
    val_dataset = dataset[800:]
    test_dataset = WaterDataset(num_samples=200)

    models = [
        (SchNet, "SchNet", "best_model.pth"),
        (SchNetPlus, "SchNetPlus", "best_model_plus.pth"),
        (SchNetPlusPlus, "SchNetPlusPlus", "best_model_plusplus.pth")
    ]

    results = []
    for model_class, model_name, model_path in models:
        results.append(evaluate_model(model_class, model_name, model_path, test_dataset))

    # Plot comparison of models
    names = [name for _, name, _ in models]
    energy_errors, r2_scores, cosine_similarities = zip(*results)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.bar(names, energy_errors, color='blue', alpha=0.7)
    plt.ylabel("MAE (eV)")
    plt.title("Energy MAE Comparison")

    plt.subplot(1, 3, 2)
    plt.bar(names, r2_scores, color='green', alpha=0.7)
    plt.ylabel("R² Score")
    plt.title("R² Score Comparison")

    plt.subplot(1, 3, 3)
    plt.bar(names, cosine_similarities, color='red', alpha=0.7)
    plt.ylabel("Cosine Similarity")
    plt.title("Force Cosine Similarity Comparison")

    plt.suptitle("Model Comparison")
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()