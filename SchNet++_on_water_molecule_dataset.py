import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, global_add_pool
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


# 1. 数据集模块
class WaterDataset(Dataset):
    """Quantum material dataset"""

    def __init__(self, data_dir="data", train=True, test_size=0.2, random_state=42):
        super().__init__()
        self.data_dir = data_dir
        self.train = train

        # 生成模拟数据（在实际应用中，应从文件加载真实数据）
        self._generate_simulated_data()

        # 将数据集分为训练集和测试集
        indices = np.arange(len(self.energies))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        self.indices = train_idx if train else test_idx

    def _generate_simulated_data(self):
        """生成模拟数据"""
        num_samples = 1000
        max_atoms = 10
        self.atomic_numbers = []
        self.positions = []
        self.energies = []

        # 生成水分子（H2O）数据
        for _ in range(num_samples):
            # 随机原子数（3-10个原子）
            num_atoms = np.random.randint(3, max_atoms + 1)

            # 原子编号（1-H, 8-O）
            z = np.random.choice([1, 8], size=num_atoms, p=[0.7, 0.3])

            # 随机位置（具有一定的物理合理性）
            pos = np.random.randn(num_atoms, 3) * 2.0

            # 确保O-H键长在合理范围内
            o_indices = np.where(z == 8)[0]
            h_indices = np.where(z == 1)[0]
            for o_idx in o_indices:
                for h_idx in h_indices:
                    dist = np.linalg.norm(pos[o_idx] - pos[h_idx])
                    if dist > 1.5:  # 如果O-H距离过大，调整H的位置
                        pos[h_idx] = (
                            pos[o_idx] + (pos[h_idx] - pos[o_idx]) * 0.96 / dist
                        )

            # 计算能量（简化模型）
            energy = self._calculate_energy(z, pos)

            self.atomic_numbers.append(torch.tensor(z, dtype=torch.long))
            self.positions.append(torch.tensor(pos, dtype=torch.float32))
            self.energies.append(torch.tensor([energy], dtype=torch.float32))

    def _calculate_energy(self, z, pos):
        """简化能量计算函数"""
        energy = 0.0
        num_atoms = len(z)

        # 键能贡献
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(pos[i] - pos[j])
                if z[i] == 8 and z[j] == 1 or z[i] == 1 and z[j] == 8:  # O-H键
                    energy += -1.0 * np.exp(-((dist - 0.96) ** 2) / 0.1)
                elif z[i] == 1 and z[j] == 1:  # H-H键
                    energy += 0.5 * np.exp(-((dist - 1.5) ** 2) / 0.2)
                else:  # 其他
                    energy += 0.1 / (dist + 1e-6)

        # 角度能贡献（简化）
        o_indices = np.where(z == 8)[0]
        for o_idx in o_indices:
            h_indices = np.where(z == 1)[0]
            if len(h_indices) >= 2:
                # 计算H-O-H角度
                vec1 = pos[h_indices[0]] - pos[o_idx]
                vec2 = pos[h_indices[1]] - pos[o_idx]
                cos_angle = np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2)
                )
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                # 104.5°是水分子的理想角度
                energy += 0.5 * (angle - np.deg2rad(104.5)) ** 2

        return energy

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            self.atomic_numbers[real_idx],
            self.positions[real_idx],
            self.energies[real_idx],
        )


# 2. 数据预处理和批处理
def collate_fn(batch):
    """将多个样本批处理成一个图"""
    z, pos, energy = zip(*batch)

    # 计算每个样本的原子数
    num_atoms = [len(x) for x in z]

    # 合并所有原子
    z = torch.cat(z, dim=0)
    pos = torch.cat(pos, dim=0)
    energy = torch.stack(energy, dim=0).squeeze(-1)  # 确保能量是一维张量

    # 创建批次索引
    batch_idx = torch.repeat_interleave(
        torch.arange(len(num_atoms)), torch.tensor(num_atoms)
    )

    return z, pos, batch_idx, energy


# 3. 模型模块
class SingleSchNetPlusPlus(MessagePassing):
    """Single SchNet++"""

    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
    ):
        super(SingleSchNetPlusPlus, self).__init__(aggr="add")

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        # 原子嵌入
        self.embedding = nn.Embedding(100, hidden_channels)

        # 距离展开
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # 简化的特征提取
        self.feature_nn = nn.Sequential(
            nn.Linear(hidden_channels, num_filters),
            nn.LeakyReLU(0.01),
            nn.Linear(num_filters, num_filters),
            nn.LeakyReLU(0.01),
        )

        # 交互模块
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # 输出层
        self.lin1 = nn.Linear(num_filters, num_filters // 2)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.lin2 = nn.Linear(num_filters // 2, 1)

        self.reset_parameters()

    # 初始化参数
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, z, pos, batch):
        # 原子嵌入
        h = self.embedding(z)

        # 特征提取
        h = self.feature_nn(h)

        # 计算边索引
        edge_index = knn_graph(pos, batch)
        row, col = edge_index

        # 计算距离
        edge_attr = pos[row] - pos[col]
        edge_length = torch.norm(edge_attr, dim=1)
        edge_attr = self.distance_expansion(edge_length)

        # 交互模块
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr, edge_length)

        # 聚合特征
        h = global_add_pool(h, batch)

        # 输出层
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # 确保输出是[batch_size]而不是[batch_size, 1]
        return h.view(-1)


class InteractionBlock(MessagePassing):
    """Interaction block"""

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(num_filters, num_filters),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(num_filters, hidden_channels),
        )
        self.cutoff = cutoff
        self.reset_parameters()

    # 初始化参数
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.mlp[4].weight)
        self.mlp[4].bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, edge_length):
        # 计算边的应力（距离）
        C = 0.5 * (torch.cos(edge_length * torch.pi / self.cutoff) + 1.0)
        W = self.mlp(edge_attr) * C.view(-1, 1)

        # 消息传递
        return self.propagate(edge_index, x=x, W=W)

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    """径向基函数"""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SchNetPlusPlus(nn.Module):
    """SchNet++主模块"""

    def __init__(
        self,
        batch_size=32,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        num_heads=3
    ):
        super(SchNetPlusPlus, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads

        # 初始化SchNet++模块
        self.schnet_plus_plus = SingleSchNetPlusPlus(
            hidden_channels, num_filters, num_interactions, num_gaussians, cutoff
        )

    def forward(self, z, pos, batch):
        # 对每个样本进行多次预测并取平均值
        size = z.size(0)  # 获取批次大小
        energy = self.schnet_plus_plus(z, pos, batch)

        for _ in range(self.num_heads - 1):
            energy += self.schnet_plus_plus(z, pos, batch)

        return energy / self.num_heads


class LossWithPhysicalConstraint(nn.Module):
    """损失函数"""

    def __init__(self, pred, target):
        super(LossWithPhysicalConstraint, self).__init__()
        self.pred = pred
        self.target = target

    def forward(self):
        numerical_loss = F.mse_loss(self.pred, self.target)
        physical_loss = self.physical_loss()
        return numerical_loss + physical_loss

    def physical_loss(self):
        # 实现物理约束损失
        # 为了简化，我们假设能量应该是负值（因为原子之间的结合是释放能量的）
        # 并且预测值不应该偏离目标值太多
        physical_constraint = torch.relu(self.pred)  # 惩罚正能量预测
        return 0.1 * torch.mean(physical_constraint)  # 权重系数为0.1


# 4. 训练和评估模块
class Trainer:
    """模型训练和评估类"""

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.loss_history = {"train": [], "val": []}
        self.mae_history = {"train": [], "val": []}

    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0

            for z, pos, batch, energy in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}"
            ):
                z, pos, batch, energy = (
                    z.to(self.device),
                    pos.to(self.device),
                    batch.to(self.device),
                    energy.to(self.device),
                )

                optimizer.zero_grad()
                pred = self.model(z, pos, batch)
                # 确保维度匹配
                pred = pred.view(-1)  # 将预测结果转换为[batch_size]
                loss = F.mse_loss(pred, energy)
                mae = F.l1_loss(pred, energy)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(energy)
                train_mae += mae.item() * len(energy)

            # 验证阶段
            val_loss, val_mae = self.evaluate(val_loader)

            # 记录历史
            train_loss /= len(train_loader.dataset)
            train_mae /= len(train_loader.dataset)
            self.loss_history["train"].append(train_loss)
            self.mae_history["train"].append(train_mae)
            self.loss_history["val"].append(val_loss)
            self.mae_history["val"].append(val_mae)

            # 调整学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model_plusplus.pth")

            # 打印进度
            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
            )

    def evaluate(self, loader):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0

        with torch.no_grad():
            for z, pos, batch, energy in loader:
                z, pos, batch, energy = (
                    z.to(self.device),
                    pos.to(self.device),
                    batch.to(self.device),
                    energy.to(self.device),
                )

                pred = self.model(z, pos, batch)
                # 确保维度匹配
                pred = pred.view(-1)  # 将预测结果转换为[batch_size]
                loss = F.mse_loss(pred, energy)
                mae = F.l1_loss(pred, energy)

                total_loss += loss.item() * len(energy)
                total_mae += mae.item() * len(energy)

        return total_loss / len(loader.dataset), total_mae / len(loader.dataset)

    def predict(self, loader):
        """生成预测"""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for z, pos, batch, energy in loader:
                z, pos, batch, energy = (
                    z.to(self.device),
                    pos.to(self.device),
                    batch.to(self.device),
                    energy.to(self.device),
                )

                pred = self.model(z, pos, batch)
                # 确保维度匹配
                pred = pred.view(-1)  # 将预测结果转换为[batch_size]
                all_preds.append(pred.cpu())
                all_targets.append(energy.cpu())

        return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


# 5. 可视化模块
class Visualizer:
    """训练过程可视化"""

    @staticmethod
    def plot_training_history(trainer):
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(trainer.loss_history["train"], label="Train Loss")
        plt.plot(trainer.loss_history["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        # MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(trainer.mae_history["train"], label="Train MAE")
        plt.plot(trainer.mae_history["val"], label="Validation MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Training and Validation MAE")
        plt.legend()

        plt.tight_layout()
        plt.savefig("loss_curve_plusplus.png")
        plt.close()

    @staticmethod
    def plot_predictions(preds, targets):
        """绘制预测值与真实值"""
        plt.figure(figsize=(6, 6))
        plt.scatter(targets.numpy(), preds.numpy(), alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Predictions vs True Values")
        plt.savefig("predictions_plusplus.png")
        plt.close()


# 6. 主程序
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化数据集和数据加载器
    train_dataset = WaterDataset(train=True)
    val_dataset = WaterDataset(train=False)

    # 使用PyTorch的DataLoader，并传入自定义的collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # 初始化模型
    model = SchNetPlusPlus(
        batch_size=32,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
    )

    # 训练模型
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader, epochs=50)

    # 可视化训练过程
    Visualizer.plot_training_history(trainer)

    # 评估最终模型
    train_preds, train_targets = trainer.predict(train_loader)
    val_preds, val_targets = trainer.predict(val_loader)

    print(f"\nFinal Training MAE: {F.l1_loss(train_preds, train_targets):.4f}")
    print(f"Final Validation MAE: {F.l1_loss(val_preds, val_targets):.4f}")

    # 可视化预测结果
    Visualizer.plot_predictions(val_preds, val_targets)


# 添加自定义的kNN函数
def knn_graph(x, batch, k=12):
    """构建k近邻图"""
    # 计算每个样本的起始索引
    batch_size = int(batch.max()) + 1
    ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=batch.device)
    for i in range(batch_size):
        ptr[i + 1] = (batch == i).sum() + ptr[i]

    # 构建边
    edge_index = []
    for i in range(batch_size):
        # 获取当前批次的原子
        start, end = ptr[i], ptr[i + 1]
        if end > start:  # 确保有原子
            # 当前批次的原子坐标
            pos_i = x[start:end]
            # 计算原子间距离
            dist = torch.cdist(pos_i, pos_i)
            # 对每个原子找出k个最近的原子(不包括自己)
            _, topk_idx = dist.topk(k=min(k + 1, dist.size(1)), dim=1, largest=False)
            # 移除自连接（通常是第一个，因为距离为0）
            topk_idx = topk_idx[:, 1:] if topk_idx.size(1) > 1 else topk_idx

            # 构建边
            for j in range(pos_i.size(0)):
                for idx in topk_idx[j]:
                    # 添加无向边 (source, target)
                    edge_index.append([j + start, idx.item() + start])

    # 转换为张量
    if not edge_index:  # 如果没有边
        return torch.zeros(2, 0, dtype=torch.long, device=batch.device)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=batch.device).t()
    return edge_index


if __name__ == "__main__":
    main()
