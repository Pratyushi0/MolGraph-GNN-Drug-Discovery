# ============================================================
# src/model.py
# MolGraph Neural Network Architecture
#
# Architecture: GAT (Graph Attention Network) with:
#   - Multi-head attention message passing
#   - Residual connections
#   - Batch normalization
#   - Global attention pooling
#   - Multi-layer MLP readout
#
# This is a state-of-the-art molecular property predictor.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    BatchNorm,
    LayerNorm,
)
from torch_geometric.nn import aggr


# ─────────────────────────────────────────
# RESIDUAL GNN BLOCK
# ─────────────────────────────────────────

class GATResidualBlock(nn.Module):
    """
    A single GATv2 layer with residual connection and normalization.
    
    GATv2 is an improved version of GAT that fixes expressiveness issues
    of the original GAT by computing dynamic attention scores.
    
    Paper: "How Attentive are Graph Attention Networks?" (ICLR 2022)
    """

    def __init__(self, in_channels, out_channels, heads=4,
                 dropout=0.2, edge_dim=10):
        super(GATResidualBlock, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads

        # Main GATv2 convolution
        self.conv = GATv2Conv(
            in_channels  = in_channels,
            out_channels = out_channels // heads,
            heads        = heads,
            dropout      = dropout,
            edge_dim     = edge_dim,
            concat       = True,    # Concat head outputs
            add_self_loops = True,
        )

        # Batch normalization
        self.norm = BatchNorm(out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Residual projection (if dimensions differ)
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr=None):
        # Save input for residual
        residual = x

        # GATv2 message passing
        out = self.conv(x, edge_index, edge_attr=edge_attr)
        out = self.norm(out)
        out = F.elu(out)
        out = self.dropout(out)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return out + residual


# ─────────────────────────────────────────
# GIN BLOCK (Graph Isomorphism Network)
# ─────────────────────────────────────────

class GINBlock(nn.Module):
    """
    GIN layer — the most expressive GNN in the Weisfeiler-Lehman hierarchy.
    Paper: "How Powerful are Graph Neural Networks?" (ICLR 2019)
    """

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(GINBlock, self).__init__()

        # MLP for GIN aggregation
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        self.conv    = GINConv(mlp, train_eps=True)
        self.dropout = nn.Dropout(dropout)

        # Residual projection
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr=None):
        residual = x
        out = self.conv(x, edge_index)
        out = self.dropout(out)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return F.relu(out + residual)


# ─────────────────────────────────────────
# MAIN MolGraph MODEL
# ─────────────────────────────────────────

class MolGraphNet(nn.Module):
    """
    MolGraph: Full Graph Attention Network for Molecular Property Prediction.
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  INPUT: Molecule as Graph                           │
    │  x: [N, 75] atom features                          │
    │  edge_index: [2, E] bond connectivity              │
    │  edge_attr: [E, 10] bond features                  │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │  INPUT PROJECTION                                   │
    │  Linear(75 → hidden_dim)                            │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │  GATv2 RESIDUAL BLOCKS × num_layers                 │
    │  Each: GATv2 → BatchNorm → ELU → Dropout → Add      │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │  GLOBAL ATTENTION POOLING                           │
    │  Learns which atoms are most important              │
    │  [N, hidden_dim] → [batch_size, hidden_dim]         │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │  READOUT MLP                                        │
    │  hidden_dim → 256 → 128 → 64 → num_classes          │
    └──────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────┐
    │  OUTPUT: Property Prediction Score                  │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        in_channels     = 75,
        hidden_channels = 256,
        out_channels    = 128,
        num_layers      = 4,
        num_heads       = 8,
        edge_dim        = 10,
        num_classes     = 1,
        dropout         = 0.2,
        task_type       = "classification",
    ):
        super(MolGraphNet, self).__init__()

        self.task_type       = task_type
        self.num_layers      = num_layers
        self.hidden_channels = hidden_channels

        # ── Input Projection ──
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
        )

        # ── GATv2 Residual Blocks ──
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GATResidualBlock(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels,
                    heads        = num_heads,
                    dropout      = dropout,
                    edge_dim     = edge_dim,
                )
            )

        # ── Global Attention Pooling ──
        # Learns a "gate" to weight each atom's importance
        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

        # ── Readout MLP ──
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass through MolGraphNet.
        
        Args:
            x:          Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr:  Edge features [num_edges, edge_dim]
            batch:      Batch assignment vector [num_nodes]
            
        Returns:
            Prediction tensor [batch_size, num_classes]
        """
        # Input projection
        x = self.input_proj(x)

        # Message passing through GATv2 layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr=edge_attr)

        # Global pooling → molecule-level representation
        x = self.pool(x, batch)   # [batch_size, hidden_channels]

        # Readout
        out = self.readout(x)     # [batch_size, num_classes]

        return out

    def get_molecular_embedding(self, x, edge_index, edge_attr, batch):
        """
        Extract molecular embeddings (before readout MLP).
        Useful for visualizing chemical space / clustering.
        
        Returns:
            embeddings: [batch_size, hidden_channels]
        """
        x = self.input_proj(x)
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr=edge_attr)
        embeddings = self.pool(x, batch)
        return embeddings

    def count_parameters(self):
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────
# ALTERNATIVE: DUAL-STREAM MODEL
# (GNN + Morgan Fingerprint combined)
# ─────────────────────────────────────────

class DualStreamMolNet(nn.Module):
    """
    Advanced model combining:
    1. Graph-based features (GNN branch)
    2. Morgan fingerprint features (MLP branch)
    
    This mimics how chemists think: both 3D structure AND
    substructure patterns matter for drug activity.
    """

    def __init__(
        self,
        in_channels      = 75,
        hidden_channels  = 256,
        fingerprint_dim  = 2048,
        num_layers       = 4,
        num_heads        = 8,
        edge_dim         = 10,
        num_classes      = 1,
        dropout          = 0.2,
        task_type        = "classification",
    ):
        super(DualStreamMolNet, self).__init__()

        self.task_type = task_type

        # ── GNN Stream ──
        self.gnn_input = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
        )

        self.gnn_layers = nn.ModuleList([
            GATResidualBlock(hidden_channels, hidden_channels, num_heads, dropout, edge_dim)
            for _ in range(num_layers)
        ])

        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

        # ── Fingerprint Stream ──
        self.fp_encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Fusion Layer ──
        fusion_dim = hidden_channels + 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch, fingerprints):
        # GNN branch
        gnn_x = self.gnn_input(x)
        for layer in self.gnn_layers:
            gnn_x = layer(gnn_x, edge_index, edge_attr=edge_attr)
        gnn_out = self.pool(gnn_x, batch)  # [B, hidden_channels]

        # Fingerprint branch
        fp_out = self.fp_encoder(fingerprints)  # [B, 256]

        # Fuse both streams
        combined = torch.cat([gnn_out, fp_out], dim=1)  # [B, fusion_dim]
        out = self.fusion(combined)

        return out


# ─────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────

def build_model(model_type="GAT", in_channels=75, hidden_channels=256,
                out_channels=128, num_layers=4, num_heads=8,
                edge_dim=10, num_classes=1, dropout=0.2,
                task_type="classification"):
    """Factory function to build models by name."""

    if model_type == "GAT":
        model = MolGraphNet(
            in_channels     = in_channels,
            hidden_channels = hidden_channels,
            out_channels    = out_channels,
            num_layers      = num_layers,
            num_heads       = num_heads,
            edge_dim        = edge_dim,
            num_classes     = num_classes,
            dropout         = dropout,
            task_type       = task_type,
        )
    elif model_type == "DualStream":
        model = DualStreamMolNet(
            in_channels     = in_channels,
            hidden_channels = hidden_channels,
            fingerprint_dim = 2048,
            num_layers      = num_layers,
            num_heads       = num_heads,
            edge_dim        = edge_dim,
            num_classes     = num_classes,
            dropout         = dropout,
            task_type       = task_type,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'GAT' or 'DualStream'")

    return model


if __name__ == "__main__":
    # Architecture test
    print("🏗️  Building MolGraphNet...")
    model = build_model(
        model_type      = "GAT",
        in_channels     = 75,
        hidden_channels = 256,
        num_layers      = 4,
        num_heads       = 8,
        edge_dim        = 10,
        num_classes     = 1,
        dropout         = 0.2,
        task_type       = "classification",
    )

    print(f"\n📐 Model Architecture:")
    print(model)
    print(f"\n📊 Total Parameters: {model.count_parameters():,}")

    # Dummy forward pass
    import torch
    from torch_geometric.data import Batch, Data

    # Simulate a batch of 4 molecules
    data_list = []
    for _ in range(4):
        n = torch.randint(5, 30, (1,)).item()  # Random number of atoms
        e = n * 2  # Approximate number of edges
        data = Data(
            x          = torch.randn(n, 75),
            edge_index = torch.randint(0, n, (2, e)),
            edge_attr  = torch.randn(e, 10),
            y          = torch.tensor([[1.0]]),
        )
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    print(f"\n✅ Forward pass successful!")
    print(f"   Input: 4 molecules (variable sizes)")
    print(f"   Output shape: {out.shape}  (4 predictions)")
    print(f"   Output values: {out.squeeze().tolist()}")
