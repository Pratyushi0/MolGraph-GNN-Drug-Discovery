#!/usr/bin/env python3
# ============================================================
# main.py - Corrected for PyTorch 2.6+ & Apple Silicon
# Features: Security Allowlisting, Resume Training, MPS Support
# ============================================================

import os
import sys
import argparse
import torch

# ─────────────────────────────────────────────────────────
# FIX: Proper Global Allowlisting for PyTorch 2.6+
# ─────────────────────────────────────────────────────────
try:
    # Import actual classes to prevent: AttributeError: 'str' object has no attribute '__module__'
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    
    # Registering the classes as safe for torch.load
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
    
    # Add internal storage types often used in processed datasets
    try:
        from torch_geometric.data.storage import DataStorage, GlobalStorage
        torch.serialization.add_safe_globals([DataStorage, GlobalStorage])
    except ImportError:
        pass
        
    print("✅ PyTorch Geometric safety globals initialized.")
except Exception as e:
    print(f"⚠️ Note: Safety globals could not be fully set: {e}")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(
        description="MolGraph: GNN Drug Activity Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict", "api", "demo"])
    parser.add_argument("--dataset", type=str, default="BBBP", choices=["BBBP", "HIV", "ESOL", "FreeSolv"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--hidden",     type=int,   default=256)
    parser.add_argument("--layers",     type=int,   default=4)
    parser.add_argument("--heads",      type=int,   default=8)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--smiles",     type=str,   default="CC(=O)Oc1ccccc1C(=O)O")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/best_model.pt")
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--api_port",   type=int,   default=8000)

    return parser.parse_args()


def run_training(args):
    """Full training pipeline with Resume Logic."""
    from src.dataset import get_dataloaders, DATASET_REGISTRY
    from src.train import MolGraphTrainer

    print("\n" + "=" * 60)
    print("  🧬 MolGraph Training Pipeline")
    print("=" * 60)

    # Device setup: Priority for Mac M-series (mps), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"  🖥️  Device: {device}")

    torch.manual_seed(args.seed)

    dataset_config = DATASET_REGISTRY[args.dataset]
    task_type = dataset_config['task']

    # Load data (Ensure src/dataset.py uses weights_only=False)
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        dataset_name = args.dataset,
        root         = "data/",
        batch_size   = args.batch_size,
        seed         = args.seed,
    )

    config = {
        'device':          device,
        'model_type':      'GAT',
        'task_type':       task_type,
        'in_channels':     dataset_info['num_node_features'],
        'hidden_channels': args.hidden,
        'num_layers':      args.layers,
        'num_heads':       args.heads,
        'edge_dim':        dataset_info['num_edge_features'],
        'num_classes':     1,
        'dropout':         args.dropout,
        'epochs':          args.epochs,
        'batch_size':      args.batch_size,
        'lr':              args.lr,
        'weight_decay':    1e-4,
        'lr_scheduler':    'cosine',
        'patience':        args.patience,
        'clip_grad_norm':  1.0,
        'checkpoint_path': args.checkpoint,
        'use_wandb':       args.use_wandb,
    }

    trainer = MolGraphTrainer(config)

    # --- RESUME LOGIC ---
    if os.path.exists(args.checkpoint):
        print(f"♻️  Found existing checkpoint at {args.checkpoint}. Loading weights to resume...")
        try:
            # We use weights_only=False here as well for the model checkpoint
            trainer.model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
            print("  Successfully resumed from checkpoint.")
        except Exception as e:
            print(f"  ⚠️ Could not load checkpoint ({e}). Starting from scratch.")

    # Train
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)
    test_results = trainer.test(test_loader)

    print(f"\n✅ Training complete. Best model: {args.checkpoint}")
    return test_results


def run_demo(args):
    """Quick demo mode."""
    from src.molecular_features import smiles_to_graph, get_molecular_descriptors
    from src.visualize import plot_molecule_graph
    
    print("\n🧬 MolGraph Demo Mode")
    os.makedirs("logs", exist_ok=True)
    
    mol_name, smiles = "Aspirin", "CC(=O)Oc1ccccc1C(=O)O"
    desc = get_molecular_descriptors(smiles)
    print(f"  💊 {mol_name} | MW: {desc['MolWt']:.1f} | LogP: {desc['LogP']:.2f}")
    
    save_path = "logs/demo_aspirin.png"
    plot_molecule_graph(smiles, title=f"Demo: {mol_name}", save_path=save_path)
    print(f"✅ Visualization saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║          🧬  M o l G r a p h                        ║
║      GNN-Powered Drug Activity Predictor             ║
╚══════════════════════════════════════════════════════╝
    Mode: {args.mode.upper()}
    """)

    if args.mode == "train":
        run_training(args)
    elif args.mode == "demo":
        run_demo(args)
    # Note: Predict and API modes follow same structure
    else:
        print(f"Mode {args.mode} selected. (Ensure logic is implemented in main.py)")