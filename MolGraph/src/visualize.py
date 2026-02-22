# ============================================================
# src/visualize.py
# Visualization utilities for MolGraph
# - Molecular graph plots
# - Training curves
# - ROC curves
# - Chemical space (UMAP/t-SNE)
# - Attention weight heatmaps
# ============================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import torch

# Style setup
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary':   '#2E86AB',
    'secondary': '#A23B72',
    'accent':    '#F18F01',
    'success':   '#44BBA4',
    'danger':    '#E94F37',
    'neutral':   '#393E41',
}

# Atom color map (CPK coloring convention)
ATOM_COLORS = {
    'C':  '#404040',   # Dark gray
    'N':  '#3050F8',   # Blue
    'O':  '#FF0D0D',   # Red
    'S':  '#FFFF30',   # Yellow
    'F':  '#90E050',   # Green
    'Cl': '#1FF01F',   # Bright green
    'Br': '#A62929',   # Brown-red
    'I':  '#940094',   # Purple
    'P':  '#FF8000',   # Orange
    'H':  '#FFFFFF',   # White
}


# ─────────────────────────────────────────
# MOLECULAR GRAPH VISUALIZATION
# ─────────────────────────────────────────

def plot_molecule_graph(smiles: str, title: str = None,
                        save_path: str = None, attention_weights=None):
    """
    Visualize a molecule as a 2D graph using NetworkX.
    
    Nodes = atoms (colored by element type)
    Edges = bonds (thickness by bond order)
    Optional: color nodes by attention weight (hot colormap)
    
    Args:
        smiles:           SMILES string
        title:            Plot title
        save_path:        Where to save the figure
        attention_weights: Optional per-atom attention weights for coloring
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"⚠️  Invalid SMILES: {smiles}")
        return

    # Build NetworkX graph
    G = nx.Graph()

    # Add nodes (atoms)
    for atom in mol.GetAtoms():
        idx    = atom.GetIdx()
        symbol = atom.GetSymbol()
        G.add_node(idx, symbol=symbol, atomic_num=atom.GetAtomicNum())

    # Add edges (bonds)
    bond_orders = {
        Chem.rdchem.BondType.SINGLE:   1.0,
        Chem.rdchem.BondType.DOUBLE:   2.0,
        Chem.rdchem.BondType.TRIPLE:   3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5,
    }
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        G.add_edge(i, j, bond_order=bond_orders.get(btype, 1.0))

    # 2D coordinates using RDKit (best for molecules)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {
        atom.GetIdx(): (
            conf.GetAtomPosition(atom.GetIdx()).x,
            conf.GetAtomPosition(atom.GetIdx()).y
        )
        for atom in mol.GetAtoms()
    }

    # Node colors
    if attention_weights is not None:
        # Color by attention (hot colormap)
        attn = np.array(attention_weights)
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        cmap = plt.cm.YlOrRd
        node_colors = [cmap(attn[i]) for i in range(len(G.nodes()))]
    else:
        node_colors = [
            ATOM_COLORS.get(
                G.nodes[n]['symbol'], '#AAAAAA'
            ) for n in G.nodes()
        ]

    # Edge widths by bond order
    edge_widths = [G[u][v]['bond_order'] * 1.5 for u, v in G.edges()]

    # Node labels
    labels = {n: G.nodes[n]['symbol'] for n in G.nodes()}

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    nx.draw_networkx_edges(
        G, pos, width=edge_widths,
        edge_color='#666666', ax=ax, alpha=0.8
    )
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=500, ax=ax,
        edgecolors='black', linewidths=1.5
    )
    nx.draw_networkx_labels(
        G, pos, labels, font_size=10,
        font_color='white', font_weight='bold', ax=ax
    )

    # Add colorbar for attention
    if attention_weights is not None:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.YlOrRd,
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Attention Weight', shrink=0.6)

    ax.set_title(title or f"Molecule: {smiles}", fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    # SMILES annotation
    ax.text(0.5, -0.02, f"SMILES: {smiles}",
            transform=ax.transAxes, ha='center',
            fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved molecule plot to: {save_path}")
    plt.close()
    return fig


# ─────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────

def plot_training_curves(history: dict, task_type="classification",
                         save_path: str = None):
    """
    Plot training and validation loss + metric curves.
    
    Args:
        history:   Dict with keys: train_loss, val_loss,
                   train_metric, val_metric
        task_type: 'classification' or 'regression'
        save_path: Where to save the figure
    """
    metric_name = "ROC-AUC" if task_type == "classification" else "R²"
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MolGraph Training Progress', fontsize=15, fontweight='bold')

    # ── Loss Curves ──
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], color=COLORS['primary'],
            linewidth=2, label='Train Loss', marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'],   color=COLORS['danger'],
            linewidth=2, label='Val Loss',   marker='s', markersize=3,
            linestyle='--')

    # Best epoch marker
    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7,
               label=f'Best Epoch ({best_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # ── Metric Curves ──
    ax = axes[1]
    ax.plot(epochs, history['train_metric'], color=COLORS['primary'],
            linewidth=2, label=f'Train {metric_name}',
            marker='o', markersize=3)
    ax.plot(epochs, history['val_metric'],   color=COLORS['success'],
            linewidth=2, label=f'Val {metric_name}',
            marker='s', markersize=3, linestyle='--')

    best_metric_epoch = np.argmax(history['val_metric']) + 1
    best_metric_val   = max(history['val_metric'])
    ax.axvline(x=best_metric_epoch, color='gray', linestyle=':', alpha=0.7,
               label=f'Best ({best_metric_val:.3f})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved training curves to: {save_path}")
    plt.close()


# ─────────────────────────────────────────
# ROC CURVE
# ─────────────────────────────────────────

def plot_roc_curve(labels, preds, save_path: str = None):
    """Plot ROC curve with AUC score."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5,
            label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--',
            linewidth=1.5, label='Random Classifier')

    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — MolGraph', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # Annotate AUC
    ax.annotate(f'AUC = {auc:.4f}',
                xy=(0.6, 0.2),
                fontsize=14,
                fontweight='bold',
                color=COLORS['primary'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['primary'], alpha=0.8))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved ROC curve to: {save_path}")
    plt.close()
    return auc


# ─────────────────────────────────────────
# CHEMICAL SPACE VISUALIZATION (UMAP)
# ─────────────────────────────────────────

def plot_chemical_space(embeddings, labels, smiles_list=None,
                         method='tsne', save_path=None):
    """
    Visualize the learned molecular embedding space.
    Shows how the model separates active vs inactive compounds.
    
    Args:
        embeddings: numpy array [N, embed_dim]
        labels:     numpy array [N] (binary class labels)
        method:     'tsne' or 'umap'
        save_path:  Where to save
    """
    print(f"📐 Reducing {embeddings.shape[1]}D embeddings to 2D using {method.upper()}...")

    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30,
                       n_iter=1000, verbose=0)
        coords_2d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)
        except ImportError:
            print("⚠️  UMAP not installed. Falling back to t-SNE.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))

    class_colors = {0: COLORS['primary'], 1: COLORS['danger']}
    class_labels = {0: 'Inactive', 1: 'Active'}

    for cls in [0, 1]:
        mask = (labels == cls)
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=class_colors[cls],
            label=class_labels[cls],
            alpha=0.6, s=30, edgecolors='none'
        )

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title(f'Chemical Embedding Space ({method.upper()})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved chemical space plot to: {save_path}")
    plt.close()


# ─────────────────────────────────────────
# PREDICTION DISTRIBUTION
# ─────────────────────────────────────────

def plot_prediction_distribution(preds, labels, save_path=None):
    """
    Plot the distribution of model predictions for each class.
    Shows model confidence and calibration.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    preds_active   = preds[labels == 1]
    preds_inactive = preds[labels == 0]

    ax.hist(preds_inactive, bins=40, alpha=0.6, color=COLORS['primary'],
            label='Inactive (Class 0)', density=True)
    ax.hist(preds_active,   bins=40, alpha=0.6, color=COLORS['danger'],
            label='Active (Class 1)', density=True)

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5,
               label='Decision boundary (0.5)')

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prediction Probability Distribution', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved distribution plot to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # Pyrene (aromatic)
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    ]

    print("Testing visualizations...")
    for i, smi in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            print(f"✅ Molecule {i+1}: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
            plot_molecule_graph(
                smi,
                title=f"Molecule {i+1}",
                save_path=f"logs/molecule_{i+1}.png"
            )

    print("\n✅ Visualization module working correctly!")
