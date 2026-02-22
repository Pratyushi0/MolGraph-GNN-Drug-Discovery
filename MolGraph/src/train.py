# ============================================================
# src/train.py
# Full Training Pipeline with:
# - Early stopping
# - Learning rate scheduling
# - Gradient clipping
# - Checkpoint saving
# - W&B / TensorBoard logging
# - Validation monitoring
# ============================================================

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────

def get_loss_function(task_type):
    """Return appropriate loss function based on task type."""
    if task_type == "classification":
        return nn.BCEWithLogitsLoss()
    elif task_type == "regression":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ─────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn,
                    device, clip_grad_norm=1.0):
    """
    Run one complete training epoch.
    
    Returns:
        avg_loss (float): Average loss over all batches
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        # Compute loss
        target = batch.y.to(device)
        loss = loss_fn(out, target)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            )

        # Update parameters
        optimizer.step()

        total_loss    += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


# ─────────────────────────────────────────
# VALIDATION STEP
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, task_type="classification"):
    """
    Evaluate model on a dataset split.
    
    Returns:
        avg_loss:   Average loss
        all_preds:  All predictions (numpy)
        all_labels: All true labels (numpy)
    """
    model.eval()
    total_loss    = 0.0
    total_samples = 0
    all_preds     = []
    all_labels    = []

    for batch in loader:
        batch  = batch.to(device)
        out    = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y.to(device)

        loss = loss_fn(out, target)
        total_loss    += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

        # Collect predictions
        if task_type == "classification":
            preds = torch.sigmoid(out).cpu().numpy()
        else:
            preds = out.cpu().numpy()

        all_preds.append(preds)
        all_labels.append(target.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0).squeeze()

    return total_loss / total_samples, all_preds, all_labels


# ─────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────

class EarlyStopping:
    """
    Stop training if validation metric doesn't improve for `patience` epochs.
    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience=20, min_delta=1e-4,
                 checkpoint_path="checkpoints/best_model.pt",
                 mode="min"):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.mode             = mode
        self.best_score       = None
        self.counter          = 0
        self.early_stop       = False

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif self._is_improvement(score):
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def _save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)


# ─────────────────────────────────────────
# METRICS TRACKER
# ─────────────────────────────────────────

class MetricsTracker:
    """Track and display training metrics across epochs."""

    def __init__(self, task_type="classification"):
        self.task_type = task_type
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_metric': [], 'val_metric': [],
            'lr': [], 'epoch_time': [],
        }

    def compute_metric(self, preds, labels):
        """Compute primary evaluation metric."""
        from sklearn.metrics import roc_auc_score, r2_score

        if self.task_type == "classification":
            try:
                return roc_auc_score(labels, preds)
            except Exception:
                return 0.5
        else:
            return r2_score(labels, preds)

    def update(self, train_loss, val_loss, train_preds, train_labels,
               val_preds, val_labels, lr, epoch_time):
        train_metric = self.compute_metric(train_preds, train_labels)
        val_metric   = self.compute_metric(val_preds, val_labels)

        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metric'].append(train_metric)
        self.history['val_metric'].append(val_metric)
        self.history['lr'].append(lr)
        self.history['epoch_time'].append(epoch_time)

        return train_metric, val_metric

    def get_metric_name(self):
        return "ROC-AUC" if self.task_type == "classification" else "R²"


# ─────────────────────────────────────────
# MAIN TRAINER CLASS
# ─────────────────────────────────────────

class MolGraphTrainer:
    """
    Full training pipeline for MolGraphNet.
    
    Usage:
        trainer = MolGraphTrainer(config)
        trainer.train(train_loader, val_loader)
        trainer.test(test_loader)
    """

    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')

        # Build model
        self.model = build_model(
            model_type      = config.get('model_type', 'GAT'),
            in_channels     = config.get('in_channels', 75),
            hidden_channels = config.get('hidden_channels', 256),
            num_layers      = config.get('num_layers', 4),
            num_heads       = config.get('num_heads', 8),
            edge_dim        = config.get('edge_dim', 10),
            num_classes     = config.get('num_classes', 1),
            dropout         = config.get('dropout', 0.2),
            task_type       = config.get('task_type', 'classification'),
        ).to(self.device)

        print(f"\n🧠 Model built on device: {self.device}")
        print(f"   Parameters: {self.model.count_parameters():,}")

        # Loss function
        self.loss_fn = get_loss_function(config.get('task_type', 'classification'))

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = config.get('lr', 0.001),
            weight_decay = config.get('weight_decay', 1e-4),
        )

        # LR Scheduler
        scheduler_type = config.get('lr_scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max  = config.get('epochs', 100),
                eta_min = 1e-6,
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5
            )

        # Early stopping
        checkpoint_path = config.get('checkpoint_path', 'checkpoints/best_model.pt')
        self.early_stopping = EarlyStopping(
            patience         = config.get('patience', 20),
            checkpoint_path  = checkpoint_path,
            mode             = "min",
        )

        # Metrics tracker
        self.tracker = MetricsTracker(config.get('task_type', 'classification'))

        # W&B setup
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project = config.get('wandb_project', 'MolGraph'),
                config  = config,
            )

    def train(self, train_loader, val_loader, epochs=None):
        """
        Run full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader:   Validation DataLoader
            epochs:       Number of epochs (overrides config if provided)
        """
        epochs = epochs or self.config.get('epochs', 100)
        task_type = self.config.get('task_type', 'classification')
        metric_name = self.tracker.get_metric_name()
        clip_norm = self.config.get('clip_grad_norm', 1.0)

        print(f"\n{'='*60}")
        print(f"  🚀 Starting MolGraph Training")
        print(f"{'='*60}")
        print(f"  Epochs:      {epochs}")
        print(f"  Batch size:  {self.config.get('batch_size', 64)}")
        print(f"  LR:          {self.config.get('lr', 0.001)}")
        print(f"  Task:        {task_type}")
        print(f"  Metric:      {metric_name}")
        print(f"{'='*60}\n")

        best_val_metric = 0.0 if task_type == "classification" else -999

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Training
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer,
                self.loss_fn, self.device, clip_norm
            )

            # Validation
            val_loss, val_preds, val_labels = evaluate(
                self.model, val_loader, self.loss_fn,
                self.device, task_type
            )

            # Train metrics (sample from train set)
            _, train_preds, train_labels = evaluate(
                self.model, train_loader, self.loss_fn,
                self.device, task_type
            )

            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update metrics
            train_metric, val_metric = self.tracker.update(
                train_loss, val_loss,
                train_preds, train_labels,
                val_preds, val_labels,
                current_lr, epoch_time
            )

            # LR scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Early stopping check
            self.early_stopping(val_loss, self.model)

            # Update best metric
            if task_type == "classification":
                is_best = val_metric > best_val_metric
            else:
                is_best = val_metric > best_val_metric
            if is_best:
                best_val_metric = val_metric

            # W&B logging
            if self.use_wandb:
                wandb.log({
                    'train/loss':         train_loss,
                    'train/metric':       train_metric,
                    'val/loss':           val_loss,
                    'val/metric':         val_metric,
                    'learning_rate':      current_lr,
                    'epoch':              epoch,
                })

            # Print progress
            star = "⭐" if is_best else "  "
            print(
                f"Epoch {epoch:03d}/{epochs:03d} {star} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train {metric_name}: {train_metric:.4f} | "
                f"Val {metric_name}: {val_metric:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping
            if self.early_stopping.early_stop:
                print(f"\n⛔ Early stopping triggered at epoch {epoch}")
                break

        print(f"\n✅ Training complete!")
        print(f"   Best Val {metric_name}: {best_val_metric:.4f}")
        print(f"   Model saved to: {self.early_stopping.checkpoint_path}")

        # Save training history
        history_path = os.path.join(
            os.path.dirname(self.early_stopping.checkpoint_path),
            "training_history.json"
        )
        with open(history_path, 'w') as f:
            json.dump(self.tracker.history, f, indent=2)
        print(f"   History saved to: {history_path}")

        return self.tracker.history

    @torch.no_grad()
    def test(self, test_loader):
        """Run full evaluation on test set and print metrics."""
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, f1_score,
            classification_report, mean_squared_error,
            mean_absolute_error, r2_score
        )

        # Load best model
        checkpoint_path = self.early_stopping.checkpoint_path
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"\n📥 Loaded best model from {checkpoint_path}")

        task_type = self.config.get('task_type', 'classification')
        test_loss, preds, labels = evaluate(
            self.model, test_loader, self.loss_fn,
            self.device, task_type
        )

        print(f"\n{'='*50}")
        print(f"  📊 TEST SET RESULTS")
        print(f"{'='*50}")
        print(f"  Test Loss: {test_loss:.4f}")

        if task_type == "classification":
            auc = roc_auc_score(labels, preds)
            binary_preds = (preds >= 0.5).astype(int)
            acc = accuracy_score(labels, binary_preds)
            f1  = f1_score(labels, binary_preds, zero_division=0)

            print(f"  ROC-AUC:   {auc:.4f}")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"\n  Classification Report:")
            print(classification_report(labels, binary_preds))

            results = {'test_loss': test_loss, 'auc': auc,
                       'accuracy': acc, 'f1': f1}
        else:
            rmse = np.sqrt(mean_squared_error(labels, preds))
            mae  = mean_absolute_error(labels, preds)
            r2   = r2_score(labels, preds)

            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")

            results = {'test_loss': test_loss, 'rmse': rmse,
                       'mae': mae, 'r2': r2}

        print(f"{'='*50}")
        return results


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Use: trainer = MolGraphTrainer(config)")
    print("     trainer.train(train_loader, val_loader)")
