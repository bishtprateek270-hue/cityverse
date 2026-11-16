"""
cityverse_ai.trainers.trainer
=============================
Training engine for the Urban World Model.
Handles mini-batch optimization, validation loops, learning rate scheduling,
and best-model checkpointing.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel
from cityverse_ai.losses.physics_losses import PhysicsInformedFloodLoss


class WorldModelTrainer:
    """
    Manages end-to-end training of the Urban World Model.
    """

    def __init__(
        self,
        model: UrbanWorldModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        checkpoint_dir: str = "models/checkpoints",
        loss_fn: Optional[PhysicsInformedFloodLoss] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = loss_fn or PhysicsInformedFloodLoss(weight_iou=2.5, weight_mass=3.0, weight_scalars=0.5)
        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch across all batches."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_iou = 0.0
        total_mass = 0.0
        n_batches = len(dataloader)
        
        for batch in dataloader:
            grid_state = batch["grid_state_t"].to(self.device) # (B, 2, H, W)
            scalars = batch["scalars_t"].to(self.device)       # (B, 11)
            actions = batch["actions"][:, 0, :].to(self.device)# (B, 4)
            
            target_h = batch["target_grid_h"][:, 0:1, :, :].to(self.device) # (B, 1, H, W)
            target_scalars = batch["target_scalars"][:, 0, :].to(self.device) # (B, 11)
            target_roads = batch["target_road_depth"][:, 0, :].to(self.device) # (B, E)
            curr_h = grid_state[:, 0:1, :, :]
            
            self.optimizer.zero_grad()
            
            out = self.model(grid_state, scalars, actions)
            
            loss, metrics = self.loss_fn(
                pred_h=out["pred_grid_h"],
                target_h=target_h,
                curr_h=curr_h,
                actions=actions,
                pred_scalars=out["pred_scalars"],
                target_scalars=target_scalars,
                pred_roads=out["pred_road_depth"],
                target_roads=target_roads,
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            total_loss += metrics["loss_total"]
            total_mae += metrics["loss_mae"]
            total_iou += metrics["mean_iou"]
            total_mass += metrics["loss_mass"]
            
        return {
            "train_loss": total_loss / max(1, n_batches),
            "train_mae": total_mae / max(1, n_batches),
            "train_iou": total_iou / max(1, n_batches),
            "train_mass": total_mass / max(1, n_batches),
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs validation pass without gradient tracking."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_iou = 0.0
        n_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                grid_state = batch["grid_state_t"].to(self.device)
                scalars = batch["scalars_t"].to(self.device)
                actions = batch["actions"][:, 0, :].to(self.device)
                
                target_h = batch["target_grid_h"][:, 0:1, :, :].to(self.device)
                target_scalars = batch["target_scalars"][:, 0, :].to(self.device)
                target_roads = batch["target_road_depth"][:, 0, :].to(self.device)
                curr_h = grid_state[:, 0:1, :, :]
                
                out = self.model(grid_state, scalars, actions)
                
                loss, metrics = self.loss_fn(
                    pred_h=out["pred_grid_h"],
                    target_h=target_h,
                    curr_h=curr_h,
                    actions=actions,
                    pred_scalars=out["pred_scalars"],
                    target_scalars=target_scalars,
                    pred_roads=out["pred_road_depth"],
                    target_roads=target_roads,
                )
                
                total_loss += metrics["loss_total"]
                total_mae += metrics["loss_mae"]
                total_iou += metrics["mean_iou"]
                
        val_loss = total_loss / max(1, n_batches)
        val_mae = total_mae / max(1, n_batches)
        val_iou = total_iou / max(1, n_batches)
        
        return {
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_iou": val_iou,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        save_best: bool = True
    ) -> List[Dict[str, float]]:
        """Executes full multi-epoch training loop with Cosine Annealing learning rate schedule."""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            scheduler.step()
            elapsed = time.time() - t0
            
            combined = {
                "epoch": epoch,
                "elapsed_s": elapsed,
                "lr": float(scheduler.get_last_lr()[0]),
                **train_metrics,
                **val_metrics,
            }
            self.history.append(combined)
            
            print(f"Epoch {epoch:02d}/{epochs}: Train Loss={train_metrics['train_loss']:.4f} (MAE: {train_metrics['train_mae']:.4f}, IoU: {train_metrics['train_iou']:.3f}) | Val Loss={val_metrics['val_loss']:.4f} (MAE: {val_metrics['val_mae']:.4f}, IoU: {val_metrics['val_iou']:.3f}) | LR={combined['lr']:.2e} | Time={elapsed:.1f}s")
            
            if save_best and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(str(self.checkpoint_dir / "world_model_best.pt"))
                print(f"  --> Saved new best model checkpoint! (Val Loss: {self.best_val_loss:.4f})")
                
        return self.history

    def save_checkpoint(self, filepath: str):
        """Saves model weights."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, filepath)

    def load_checkpoint(self, filepath: str):
        """Loads model weights."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

