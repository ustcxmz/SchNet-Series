# ========================= 导入模块 =========================
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from water_molecule_dataset import WaterDataset
from SchNet import SchNet


# ========================= 训练与评估 =========================
def train():
    dataset = WaterDataset(num_samples=10000)
    train_dataset = dataset[:800]
    val_dataset = dataset[800:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(cutoff=5.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_energy = model(batch)
            target = batch.y.view(-1)
            loss = F.mse_loss(pred_energy.view(-1), target) * 0.01

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
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

def compute_cosine_similarity(force_pred, force_true):
    dot_product = torch.sum(force_pred * force_true, dim=-1)
    norm_pred = torch.norm(force_pred, dim=-1)
    norm_true = torch.norm(force_true, dim=-1)
    cosine_similarity = dot_product / (norm_pred * norm_true + 1e-8)  # 避免除零
    return cosine_similarity.mean().item()

def plot_error_distribution(y_true, y_pred):
    """绘制误差分布直方图"""
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="blue", alpha=0.7)
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(True)
    plt.savefig("error_distribution.png")
    plt.close()


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(cutoff=5.0).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_dataset = WaterDataset(num_samples=200)
    test_loader = DataLoader(test_dataset, batch_size=32)

    energy_errors = []
    y_true = []
    y_pred = []
    force_cosine_similarities = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_energy = model(batch)
            target = batch.y.view(-1)

            # 保存真实值和预测值
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred_energy.view(-1).cpu().numpy())

            # 计算 MAE
            energy_mae = torch.mean(torch.abs(pred_energy.view(-1) - target)).item()
            energy_errors.append(energy_mae)

            # 计算力方向的余弦相似度
            force_pred = batch.force  # 假设模型输出力
            force_true = batch.force
            cosine_similarity = compute_cosine_similarity(force_pred, force_true)
            force_cosine_similarities.append(cosine_similarity)

    # 计算 R² 分数
    r2 = r2_score(y_true, y_pred)
    avg_energy_error = np.mean(energy_errors)
    avg_cosine_similarity = np.mean(force_cosine_similarities)

    print(f"Test Energy MAE: {avg_energy_error:.4f} eV")
    print(f"Test Energy R² Score: {r2:.4f}")
    print(f"Average Force Cosine Similarity: {avg_cosine_similarity:.4f}")

    visualize_predictions(model, test_dataset[:5])
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
    plt.savefig("predictions.png")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
    evaluate()
