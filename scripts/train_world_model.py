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
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 2. Model
    model = UrbanWorldModel(
        in_grid_channels=6,
        base_channels=24,
        scalar_dim=11,
        action_dim=4,
        num_road_edges=464,
    )
    
    # 3. Trainer
    trainer = WorldModelTrainer(
        model=model,
        lr=1e-3,
        checkpoint_dir="models/checkpoints",
    )
    
    # 4. Train 25 epochs for accurate hydrological convergence
    print("Training model across 25 epochs...")
    history = trainer.fit(train_loader, val_loader, epochs=25, save_best=True)
    
    for h in history:
        print(f"Epoch {h['epoch']}: Train Loss = {h['train_loss']:.4f} (MAE: {h['train_mae']:.4f}) | Val Loss = {h['val_loss']:.4f} (MAE: {h['val_mae']:.4f}) | Time = {h['elapsed_s']:.1f}s")
        
    # 5. Evaluate Latency
    evaluator = WorldModelEvaluator(model)
    bench = evaluator.benchmark_inference_latency(n_trials=30)
    print(f"\nFinal Inference Latency: {bench['mean_latency_ms']:.2f} ms per step (Target <30ms: {bench['meets_target_30ms']})")
    print(f"Saved trained checkpoint to models/checkpoints/world_model_best.pt")


if __name__ == "__main__":
    main()

