import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool


class CNNResNetBlock(nn.Module):
    """CNN + MaxPooling + Linear + ResNet
    这个模块结合了卷积神经网络、自适应平均池化、批量归一化和残差连接。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNResNetBlock, self).__init__()

        # 卷积层，用于提取局部特征
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 使用自适应平均池化来减少特征图的尺寸
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # 简化结构，移除了线性层，直接使用批量归一化
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 残差连接，如果输入和输出通道数不同，则添加一个卷积层和批量归一化层
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels),
            )

        # 使用LeakyReLU作为激活函数
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # 确保输入维度正确，如果输入是二维的，则添加一个维度
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # 添加一个维度
        
        # 保存残差连接
        residual = self.shortcut(x)

        # 通过卷积层和批量归一化层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # 通过自适应平均池化层
        out = self.pool(out)

        # 通过批量归一化层
        out = self.bn2(out)

        # 添加残差连接
        out += residual
        out = self.activation(out)

        return out


class SingleSchNetPlusPlus(MessagePassing):
    """Single SchNet++
    这个模块实现了单个SchNet++模型，用于分子能量预测。
    """

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

        # 原子嵌入层，将原子类型编码为固定长度的向量
        self.embedding = nn.Embedding(100, hidden_channels)

        # 距离展开层，将距离转换为高斯展开形式
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # 特征提取层，通过两个线性层和LeakyReLU激活函数
        self.feature_nn = nn.Sequential(
            nn.Linear(hidden_channels, num_filters),
            nn.LeakyReLU(0.01),
            nn.Linear(num_filters, num_filters),
            nn.LeakyReLU(0.01)
        )

        # 交互模块，包含多个InteractionBlock
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # 输出层，通过两个线性层和LeakyReLU激活函数
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

        # 计算边索引，构建k近邻图
        edge_index = knn_graph(pos, batch)
        row, col = edge_index

        # 计算距离向量和距离长度
        edge_attr = pos[row] - pos[col]
        edge_length = torch.norm(edge_attr, dim=1)
        edge_attr = self.distance_expansion(edge_length)

        # 通过多个交互模块
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr, edge_length)

        # 聚合特征，使用全局加和池化
        h = global_add_pool(h, batch)

        # 通过输出层
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        
        # 确保输出是[batch_size]而不是[batch_size, 1]
        return h.view(-1)


class InteractionBlock(MessagePassing):
    """Interaction block
    这个模块实现了SchNet++中的交互块，用于更新节点特征。
    """

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.LeakyReLU(0.01),
            nn.Linear(num_filters, num_filters),
            nn.LeakyReLU(0.01),
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
    """Radius basis function
    这个模块实现了高斯展开函数，用于将距离转换为高斯展开形式。
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SchNetPlusPlus(nn.Module):
    """Main body of SchNet++
    这个模块是SchNet++模型的主模块，用于分子能量预测。
    """

    def __init__(
        self,
        batch_size=32,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
    ):
        super(SchNetPlusPlus, self).__init__()
        self.batch_size = batch_size

        # 初始化SchNet++模块
        self.schnet_plus_plus = SingleSchNetPlusPlus(
            hidden_channels, num_filters, num_interactions, num_gaussians, cutoff
        )

    def forward(self, z, pos, batch):
        # 直接传入整个批次
        energy = self.schnet_plus_plus(z, pos, batch)
        return energy
    

class LossWithPhysicalConstraint(nn.Module):
    """Loss function
    这个模块实现了带有物理约束的损失函数，用于训练模型。
    """

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

# 添加自定义的kNN函数
def knn_graph(x, batch, k=12):
    """构建k近邻图
    这个函数根据给定的原子坐标和批次信息构建k近邻图。
    """
    # 计算每个样本的起始索引
    batch_size = int(batch.max()) + 1
    ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=batch.device)
    for i in range(batch_size):
        ptr[i + 1] = (batch == i).sum() + ptr[i]
    
    # 构建边
    edge_index = []
    for i in range(batch_size):
        # 获取当前批次的原子
        start, end = ptr[i], ptr[i+1]
        if end > start:  # 确保有原子
            # 当前批次的原子坐标
            pos_i = x[start:end]
            # 计算原子间距离
            dist = torch.cdist(pos_i, pos_i)
            # 对每个原子找出k个最近的原子(不包括自己)
            _, topk_idx = dist.topk(k=min(k+1, dist.size(1)), dim=1, largest=False)
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