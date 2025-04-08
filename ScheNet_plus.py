from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch.nn import Embedding, Sequential, Linear, ModuleList


class Emb(torch.nn.Module):
    """Distance embedding module"""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(Emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class UpdateE(torch.nn.Module):
    """Module to update edge features with deeper MLP and ResNet optimization"""

    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(UpdateE, self).__init__()
        self.cutoff = cutoff
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            torch.nn.LeakyReLU(0.015),
            Linear(num_filters, num_filters),
            torch.nn.LeakyReLU(0.015),
            Linear(num_filters, num_filters),
        )
        self.lin_out = Linear(num_filters, num_filters)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)
        self.lin_out.bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        v = self.lin(v)
        e = v[j] * W
        return self.lin_out(e) + e  # Residual connection


class UpdateV(torch.nn.Module):
    """Simplified module to update node features with a single-layer perceptron"""

    def __init__(self, hidden_channels, num_filters):
        super(UpdateV, self).__init__()
        self.act = torch.nn.LeakyReLU(0.015)
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):
        _, i = edge_index
        out = torch.zeros_like(v).scatter_add_(
            dim=0, index=i.unsqueeze(-1).expand_as(e), src=e
        )
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return v + out  # External residual connection


class UpdateU(torch.nn.Module):
    """Module to update global features"""

    def __init__(self, hidden_channels, out_channels):
        super(UpdateU, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = torch.nn.LeakyReLU(0.015)
        self.lin2 = Linear(hidden_channels // 2, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)

        output = torch.zeros(batch.max().item() + 1, v.size(1), device=v.device)
        for i in range(output.size(0)):
            mask = batch == i
            if mask.any():
                output[i] = v[mask].mean(dim=0)
        return output


class SchNetPlus(torch.nn.Module):
    """Complete SchNetPlus model implementation with deep MLP and ResNet optimization"""

    def __init__(
        self,
        cutoff=5.0,
        num_layers=3,
        hidden_channels=128,
        out_channels=1,
        num_filters=128,
        num_gaussians=50,
    ):
        super(SchNetPlus, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = Emb(0.0, cutoff, num_gaussians)

        self.update_vs = ModuleList(
            [UpdateV(hidden_channels, num_filters) for _ in range(num_layers)]
        )
        self.update_es = ModuleList(
            [
                UpdateE(hidden_channels, num_filters, num_gaussians, cutoff)
                for _ in range(num_layers)
            ]
        )
        self.update_u = UpdateU(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        edge_index = compute_radius_graph(pos, batch, self.cutoff)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v = self.init_v(z)
        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v, e, edge_index)

        u = self.update_u(v, batch)
        return u


# ========================= Helper Functions =========================
def compute_radius_graph(pos, batch, cutoff):
    """Manually implement radius-based adjacency relationships"""
    edge_index = []
    for b in torch.unique(batch):
        mask = batch == b
        pos_b = pos[mask]
        dist = torch.cdist(pos_b, pos_b)
        src, dst = torch.where((dist <= cutoff) & (dist > 0))
        edge_index.append(torch.stack([src, dst], dim=0))
    return torch.cat(edge_index, dim=1)


def compute_force_loss(pred_energy, pos, true_force):
    """Compute force field loss"""
    pred_force = -torch.autograd.grad(
        outputs=pred_energy.sum(),
        inputs=pos,
        create_graph=True,
        retain_graph=True,
    )[0]
    force_loss = F.mse_loss(pred_force, true_force)
    return force_loss


def compute_energy_conservation_loss(pred_energy, batch):
    """Compute energy conservation constraint loss"""
    # Ensure pred_energy has the correct shape
    batch_size = batch.max().item() + 1

    # If pred_energy is of shape [batch_size, out_channels], use it directly
    if pred_energy.size(0) == batch_size:
        return F.mse_loss(pred_energy, pred_energy.detach())

    # Otherwise, aggregate energy based on batch indices
    total_energy = torch.zeros(
        batch_size, pred_energy.size(1), device=pred_energy.device
    )
    batch_idx = torch.unique(batch)
    for i in batch_idx:
        mask = batch == i
        if mask.any():
            # Sum only the matching parts
            total_energy[i] = pred_energy[mask].mean(dim=0)

    energy_conservation_loss = F.mse_loss(total_energy, total_energy.detach())
    return energy_conservation_loss
