"""
Training script for the learned Urban World Model on MetroVerdia trajectories.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel
from cityverse_ai.dataset import UrbanTrajectoryDataset
from cityverse_ai.trainers.trainer import WorldModelTrainer
from cityverse_ai.trainers.evaluator import WorldModelEvaluator


def main():
    print("Initializing Urban World Model training pipeline...")
    
    # 1. Datasets
    train_dataset = UrbanTrajectoryDataset(
        trajectory_dir="data/trajectories/train",
        rollout_horizon=1,
        dem_path="data/raw_city_gis/dem_elevation.npy",
    )
    val_dataset = UrbanTrajectoryDataset(
        trajectory_dir="data/trajectories/val",
        rollout_horizon=1,
        dem_path="data/raw_city_gis/dem_elevation.npy",
    )
    
    print(f"Loaded {len(train_dataset)} training transitions, {len(val_dataset)} validation transitions.")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 2. Model
    model = UrbanWorldModel(
        in_grid_channels=6,
        base_channels=24,
        scalar_dim=11,
        action_dim=4,
        num_road_edges=464,
    )
    
    # 3. Trainer (Runs on CPU for rock-solid stability and zero-driver issues)
    trainer = WorldModelTrainer(
        model=model,
        lr=1e-3,
        weight_decay=1e-4,
        device="cpu",
        checkpoint_dir="models/checkpoints",
    )
    
    # 4. Train 30 epochs with Cosine Annealing learning rate schedule
    epochs = 30
    print(f"Training model across {epochs} epochs on expanded dataset ({len(train_dataset)} transitions)...")
    history = trainer.fit(train_loader, val_loader, epochs=epochs, save_best=True)
    
    print("\n--- Training Convergence Summary ---")
    for h in history:
        print(f"Epoch {h['epoch']:02d}/{epochs}: Train Loss={h['train_loss']:.4f} (MAE: {h['train_mae']:.4f}, IoU: {h['train_iou']:.3f}) | Val Loss={h['val_loss']:.4f} (MAE: {h['val_mae']:.4f}, IoU: {h['val_iou']:.3f}) | LR={h['lr']:.2e} | Time={h['elapsed_s']:.1f}s")
        
    # 5. Evaluate Latency & Best Checkpoint Accuracy
    trainer.load_checkpoint("models/checkpoints/world_model_best.pt")
    final_val = trainer.validate(val_loader)
    evaluator = WorldModelEvaluator(trainer.model)
    bench = evaluator.benchmark_inference_latency(n_trials=30)
    
    print("\n========================================================")
    print("WORLD MODEL RETRAINING COMPLETE")
    print("========================================================")
    print(f"  Best Val Loss: {final_val['val_loss']:.4f}")
    print(f"  Best Val MAE:  {final_val['val_mae']:.4f} m ({final_val['val_mae']*100:.2f} cm depth error)")
    print(f"  Best Val IoU:  {final_val['val_iou']:.4f} ({final_val['val_iou']*100:.1f}% Critical Success Index)")
    print(f"  Inference Latency: {bench['mean_latency_ms']:.2f} ms per step (Target <30ms: {bench['meets_target_30ms']})")
    print(f"  Checkpoint saved: models/checkpoints/world_model_best.pt")
    print("========================================================")


if __name__ == "__main__":
    main()

