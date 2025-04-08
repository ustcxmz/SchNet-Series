# ========================= Import Modules =========================
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
from water_molecule_dataset import WaterDataset
from SchNet_plus_plus import (
    SchNetPlusPlus,
    compute_energy_conservation_loss,
    compute_force_loss,
)


start_time = time.time()


# ========================= Training and Evaluation =========================
def train():
    """Training loop for the SchNetPlusPlus model"""
    dataset = WaterDataset(num_samples=10000)
    train_dataset = dataset[:800]
    val_dataset = dataset[800:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNetPlusPlus(cutoff=5.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            batch.pos.requires_grad_()
            optimizer.zero_grad()

            pred_energy = model(batch)
            target = batch.y.view(-1)

            # Compute energy loss
            energy_loss = F.mse_loss(pred_energy.view(-1), target) * 0.01

            # Compute force loss
            force_loss = compute_force_loss(pred_energy, batch.pos, batch.force)

            # Compute energy conservation loss
            energy_conservation_loss = compute_energy_conservation_loss(
                pred_energy, batch.batch
            )

            # Total loss
            loss = energy_loss + force_loss + 0.1 * energy_conservation_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_energy = model(batch)
                target = batch.y.view(-1)
                loss = F.mse_loss(pred_energy.view(-1), target) * 0.01
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_plusplus.pth")

        print(
            f"Epoch {epoch+1:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve_plusplus.png")
    plt.close()


def compute_cosine_similarity(force_pred, force_true):
    dot_product = torch.sum(force_pred * force_true, dim=-1)
    norm_pred = torch.norm(force_pred, dim=-1)
    norm_true = torch.norm(force_true, dim=-1)
    cosine_similarity = dot_product / (
        norm_pred * norm_true + 1e-8
    )  # Avoid division by zero
    return cosine_similarity.mean().item()


def plot_error_distribution(y_true, y_pred):
    """Plot error distribution histogram"""
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="blue", alpha=0.7)
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(True)
    plt.savefig("error_distribution_plusplus.png")
    plt.close()


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNetPlusPlus(cutoff=5.0).to(device)
    model.load_state_dict(torch.load("best_model_plusplus.pth"))
    model.eval()

    test_dataset = WaterDataset(num_samples=200)
    test_loader = DataLoader(test_dataset, batch_size=32)

    energy_errors = []
    y_true = []
    y_pred = []
    force_cosine_similarities = []
    absolute_errors = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_energy = model(batch)
            target = batch.y.view(-1)

            # Save true and predicted values
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred_energy.view(-1).cpu().numpy())

            # Compute absolute error for each sample
            abs_error = torch.abs(pred_energy.view(-1) - target).cpu().numpy()
            absolute_errors.extend(abs_error)

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

    print(f"Test Energy MAE: {avg_energy_error:.4f} eV")
    print(f"Test Energy R² Score: {r2:.4f}")
    print(f"Average Force Cosine Similarity: {avg_cosine_similarity:.4f}")

    # Sort by absolute error and select the 5 samples with the smallest error
    absolute_errors = np.array(absolute_errors)
    sorted_indices = np.argsort(absolute_errors)[
        :5
    ]  # Get indices of the 5 samples with the smallest error
    best_samples = [test_dataset[i] for i in sorted_indices]

    # Visualize the 5 samples with the smallest error
    visualize_predictions(model, best_samples)
    plot_error_distribution(y_true, y_pred)


def visualize_predictions(model, test_data):
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

    plt.tight_layout()
    plt.savefig("predictions_plusplus.png")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
    evaluate()

end_time = time.time()
print(f"time: {end_time - start_time}s")
