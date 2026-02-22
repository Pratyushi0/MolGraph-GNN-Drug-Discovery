# ============================================================
# src/dataset.py
# MoleculeNet Dataset Loader with custom preprocessing
# Supports: BBBP, HIV, ESOL, FreeSolv datasets
# ============================================================

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.molecular_features import smiles_to_graph


# ─────────────────────────────────────────
# DATASET REGISTRY — Download URLs
# ─────────────────────────────────────────

DATASET_REGISTRY = {
    "BBBP": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smiles_col": "smiles",
        "label_col": "p_np",
        "task": "classification",
        "description": "Blood-Brain Barrier Penetration (2,039 molecules)",
    },
    "HIV": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "smiles_col": "smiles",
        "label_col": "HIV_active",
        "task": "classification",
        "description": "HIV Inhibition Activity (41,127 molecules)",
    },
    "ESOL": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smiles_col": "smiles",
        "label_col": "measured log solubility in mols per litre",
        "task": "regression",
        "description": "Aqueous Solubility (1,128 molecules)",
    },
    "FreeSolv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/FreeSolv.csv",
        "smiles_col": "smiles",
        "label_col": "expt",
        "task": "regression",
        "description": "Free Solvation Energy (642 molecules)",
    },
}


# ─────────────────────────────────────────
# CORE DATASET CLASS
# ─────────────────────────────────────────

class MolGraphDataset(InMemoryDataset):
    """
    Custom PyTorch Geometric Dataset for molecular graphs.
    
    Automatically:
    1. Downloads the dataset from MoleculeNet
    2. Converts each SMILES to a graph (atoms=nodes, bonds=edges)
    3. Extracts rich atomic and bond features
    4. Caches processed graphs to disk for fast re-loading
    
    Usage:
        dataset = MolGraphDataset(name="BBBP", root="data/")
        data = dataset[0]   # First molecule as a graph
        print(data.x.shape) # [num_atoms, 75]
    """

    def __init__(self, name="BBBP", root="data/", transform=None,
                 pre_transform=None, force_reload=False):
        self.name = name
        self.config = DATASET_REGISTRY[name]
        self.force_reload = force_reload

        super(MolGraphDataset, self).__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [f"{self.name}.csv"]

    @property
    def processed_file_names(self):
        return [f"{self.name}_processed.pt"]

    def download(self):
        """Download dataset CSV from MoleculeNet S3."""
        import urllib.request
        url = self.config["url"]
        save_path = os.path.join(self.raw_dir, f"{self.name}.csv")

        if not os.path.exists(save_path) or self.force_reload:
            print(f"📥 Downloading {self.name} dataset...")
            print(f"   URL: {url}")
            urllib.request.urlretrieve(url, save_path)
            print(f"✅ Downloaded to {save_path}")
        else:
            print(f"✅ Dataset already exists at {save_path}")

    def process(self):
        """Convert all SMILES to graphs and save as .pt file."""
        csv_path = os.path.join(self.raw_dir, f"{self.name}.csv")
        df = pd.read_csv(csv_path)

        smiles_col = self.config["smiles_col"]
        label_col  = self.config["label_col"]
        task_type  = self.config["task"]

        print(f"\n🔬 Processing {self.name} dataset...")
        print(f"   Total molecules: {len(df)}")
        print(f"   Task type: {task_type}")

        data_list = []
        skipped = 0

        for idx, row in tqdm(df.iterrows(), total=len(df),
                             desc="Converting molecules to graphs"):
            smiles = row[smiles_col]
            label  = row[label_col]

            # Skip invalid SMILES or NaN labels
            if pd.isna(smiles) or pd.isna(label):
                skipped += 1
                continue

            # Convert SMILES to graph
            graph_dict = smiles_to_graph(str(smiles))
            if graph_dict is None:
                skipped += 1
                continue

            # Create PyG Data object
            y = torch.tensor([[float(label)]], dtype=torch.float)

            data = Data(
                x          = graph_dict['x'],
                edge_index = graph_dict['edge_index'],
                edge_attr  = graph_dict['edge_attr'],
                y          = y,
                smiles     = smiles,
                num_nodes  = graph_dict['num_atoms'],
            )
            data_list.append(data)

        print(f"✅ Successfully processed: {len(data_list)} molecules")
        print(f"⚠️  Skipped (invalid):      {skipped} molecules")

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"💾 Saved processed dataset to {self.processed_paths[0]}")


# ─────────────────────────────────────────
# DATASET SPLITTER
# ─────────────────────────────────────────

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1,
                  test_ratio=0.1, seed=42):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: MolGraphDataset instance
        train_ratio: fraction for training (default 0.8)
        val_ratio: fraction for validation (default 0.1)
        test_ratio: fraction for testing (default 0.1)
        seed: random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    print(f"\n📊 Dataset Split Summary:")
    print(f"   Total:      {n:,}")
    print(f"   Train:      {n_train:,} ({train_ratio*100:.0f}%)")
    print(f"   Validation: {n_val:,} ({val_ratio*100:.0f}%)")
    print(f"   Test:       {n_test:,} ({test_ratio*100:.0f}%)")

    return train_set, val_set, test_set


# ─────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────

def get_dataloaders(dataset_name="BBBP", root="data/",
                    batch_size=64, num_workers=0,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                    seed=42):
    """
    Complete pipeline: download → process → split → create dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info (dict)
    """
    # Load dataset
    dataset = MolGraphDataset(name=dataset_name, root=root)

    # Print dataset info
    sample = dataset[0]
    dataset_info = {
        'name':              dataset_name,
        'num_molecules':     len(dataset),
        'num_node_features': sample.x.shape[1],
        'num_edge_features': sample.edge_attr.shape[1] if sample.edge_attr is not None else 0,
        'task_type':         DATASET_REGISTRY[dataset_name]['task'],
        'description':       DATASET_REGISTRY[dataset_name]['description'],
    }

    print(f"\n📋 Dataset Info:")
    for k, v in dataset_info.items():
        print(f"   {k}: {v}")

    # Split
    train_set, val_set, test_set = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, dataset_info


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Testing dataset loading pipeline...")
    
    train_loader, val_loader, test_loader, info = get_dataloaders(
        dataset_name="BBBP",
        root="data/",
        batch_size=32
    )
    
    # Inspect first batch
    batch = next(iter(train_loader))
    print(f"\n🔍 First Batch:")
    print(f"   Batch type:   {type(batch)}")
    print(f"   Node features x: {batch.x.shape}")
    print(f"   Edge index:      {batch.edge_index.shape}")
    print(f"   Edge features:   {batch.edge_attr.shape}")
    print(f"   Labels y:        {batch.y.shape}")
    print(f"   Batch vector:    {batch.batch.shape}")
    print(f"\n✅ Dataset pipeline working correctly!")
