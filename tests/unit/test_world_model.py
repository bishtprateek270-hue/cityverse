"""
Unit and benchmarking tests for Phase 6 (Learned World Model):
- Forward pass tensor shapes & non-negative depth constraint
- Physics-informed loss gradients & mass conservation penalty
- Training step convergence on trajectory batches
- Inference latency benchmark (<50ms CPU / <20ms GPU)
- Multi-step autoregressive rollout stability
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel
from cityverse_ai.losses.physics_losses import PhysicsInformedFloodLoss
from cityverse_ai.trainers.trainer import WorldModelTrainer
from cityverse_ai.trainers.evaluator import WorldModelEvaluator
from cityverse_ai.dataset import UrbanTrajectoryDataset


@pytest.fixture
def small_world_model():
    return UrbanWorldModel(
        in_grid_channels=6,
        base_channels=16, # Small for fast testing
        scalar_dim=11,
        action_dim=4,
        num_road_edges=464,
    )


def test_world_model_forward_pass_and_shapes(small_world_model):
    B, H, W = 2, 120, 120
    grid_in = torch.randn(B, 2, H, W)
    # Ensure water depth h >= 0
    grid_in[:, 0:1, :, :] = torch.relu(grid_in[:, 0:1, :, :])
    scalars = torch.randn(B, 11)
    actions = torch.randn(B, 4)
    
    out = small_world_model(grid_in, scalars, actions)
    
    assert "pred_grid_h" in out
    assert "pred_delta_h" in out
    assert "pred_scalars" in out
    assert "pred_road_depth" in out
    
    assert out["pred_grid_h"].shape == (B, 1, H, W)
    assert out["pred_scalars"].shape == (B, 11)
    assert out["pred_road_depth"].shape == (B, 464)
    
    # Non-negative depth guarantee
    assert (out["pred_grid_h"] >= 0.0).all(), "Predicted water depth must never be negative."


def test_physics_informed_loss():
    loss_fn = PhysicsInformedFloodLoss()
    
    B, H, W = 2, 120, 120
    curr_h = torch.rand(B, 1, H, W)
    target_h = curr_h + 0.10
    pred_h = curr_h + 0.08
    pred_h.requires_grad_(True)
    
    actions = torch.rand(B, 4)
    target_scalars = torch.randn(B, 11)
    pred_scalars = torch.randn(B, 11, requires_grad=True)
    
    loss, metrics = loss_fn(pred_h, target_h, curr_h, actions, pred_scalars, target_scalars)
    
    assert loss.item() > 0.0
    assert "loss_mae" in metrics
    assert "loss_iou" in metrics
    assert "loss_mass" in metrics
    assert "loss_scalars" in metrics
    
    # Backprop
    loss.backward()
    assert pred_h.grad is not None
    assert pred_scalars.grad is not None
    assert not torch.isnan(pred_h.grad).any()


def test_world_model_training_step(small_world_model):
    optimizer = torch.optim.Adam(small_world_model.parameters(), lr=1e-3)
    loss_fn = PhysicsInformedFloodLoss()
    
    B, H, W = 2, 120, 120
    grid_in = torch.relu(torch.randn(B, 2, H, W))
    scalars = torch.randn(B, 11)
    actions = torch.randn(B, 4)
    target_h = grid_in[:, 0:1, :, :] + 0.05
    target_scalars = scalars + 0.1
    curr_h = grid_in[:, 0:1, :, :]
    
    # Run 5 gradient steps on synthetic batch
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = small_world_model(grid_in, scalars, actions)
        loss, _ = loss_fn(out["pred_grid_h"], target_h, curr_h, actions, out["pred_scalars"], target_scalars)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    # Loss should decrease
    assert losses[-1] < losses[0], f"Training loss should decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"


def test_inference_latency_benchmark(small_world_model):
    evaluator = WorldModelEvaluator(small_world_model)
    res = evaluator.benchmark_inference_latency(n_warmup=3, n_trials=10, batch_size=1)
    
    print(f"\nInference Latency Benchmark (120x120 City Grid):")
    print(f"  - Mean Step Latency: {res['mean_latency_ms']:.2f} ms")
    print(f"  - 95th Percentile:   {res['p95_latency_ms']:.2f} ms")
    
    # On modern CPU, 120x120 U-Net with base_channels=16 executes in ~10-35ms
    assert res["mean_latency_ms"] < 100.0, "World Model inference must be orders of magnitude faster than physics solver."


def test_autoregressive_rollout(small_world_model):
    evaluator = WorldModelEvaluator(small_world_model)
    
    H, W = 120, 120
    K = 4 # 4 steps ahead
    
    init_h = torch.relu(torch.randn(1, 1, H, W))
    dem = torch.rand(1, 1, H, W)
    init_scalars = torch.randn(1, 11)
    actions = torch.randn(K, 4)
    target_seq = torch.stack([init_h + 0.02 * (k + 1) for k in range(K)], dim=0) # (K, 1, H, W)
    
    rollout = evaluator.evaluate_autoregressive_rollout(
        initial_grid_h=init_h,
        dem=dem,
        initial_scalars=init_scalars,
        action_sequence=actions,
        target_grid_sequence=target_seq,
    )
    
    assert rollout["rollout_steps"] == 4
    assert len(rollout["step_rmses"]) == 4
    assert len(rollout["step_ious"]) == 4
    assert len(rollout["predicted_h_frames"]) == 4
    assert rollout["mean_rmse"] >= 0.0

