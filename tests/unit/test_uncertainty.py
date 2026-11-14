"""
Unit and benchmark tests for Phase 9 (Uncertainty Quantification & Multi-Trajectory Ensembles):
- Monte Carlo dropout epistemic variance estimation
- Confidence interval bounds (lower <= mean <= upper)
- Probabilistic road closures and multi-member trajectory fan-outs
- Comprehensive OOD test split benchmark & report generation
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel
from cityverse_ai.uncertainty.estimator import UncertaintyEstimator
from cityverse_ai.uncertainty.benchmark import ComprehensiveBenchmark


@pytest.fixture
def trained_model():
    model = UrbanWorldModel(
        in_grid_channels=6,
        base_channels=24,
        scalar_dim=11,
        action_dim=4,
        num_road_edges=464,
    )
    ckpt_path = Path("models/checkpoints/world_model_best.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def test_mc_dropout_uncertainty_quantification(trained_model):
    estimator = UncertaintyEstimator(trained_model, dropout_rate=0.20)
    
    H, W = 120, 120
    grid_in = torch.relu(torch.randn(1, 2, H, W))
    scalars = torch.randn(1, 11)
    actions = torch.randn(1, 4)
    
    res = estimator.predict_with_uncertainty(grid_in, scalars, actions, n_samples=8, confidence_level=0.90)
    
    assert "mean_depth" in res
    assert "std_depth" in res
    assert "lower_bound_depth" in res
    assert "upper_bound_depth" in res
    assert "road_closure_probabilities" in res
    
    # Check shapes
    assert res["mean_depth"].shape == (H, W)
    assert res["std_depth"].shape == (H, W)
    assert res["lower_bound_depth"].shape == (H, W)
    assert res["upper_bound_depth"].shape == (H, W)
    assert res["road_closure_probabilities"].shape == (464,)
    
    # Check confidence bound ordering: lower <= mean <= upper
    mean_d = res["mean_depth"]
    lower_d = res["lower_bound_depth"]
    upper_d = res["upper_bound_depth"]
    
    assert (lower_d <= mean_d + 1e-5).all(), "Lower bound must be <= mean."
    assert (mean_d <= upper_d + 1e-5).all(), "Mean must be <= upper bound."
    assert (lower_d >= 0.0).all(), "Lower bound depth must be non-negative."
    
    # Road closure probabilities in [0, 1]
    prob = res["road_closure_probabilities"]
    assert (prob >= 0.0).all() and (prob <= 1.0).all()


def test_multi_trajectory_ensemble(trained_model):
    estimator = UncertaintyEstimator(trained_model, dropout_rate=0.15)
    
    H, W = 120, 120
    K = 4 # 4 steps ahead
    
    init_h = torch.relu(torch.randn(1, 1, H, W))
    dem = torch.rand(1, 1, H, W)
    init_scalars = torch.randn(1, 11)
    actions = torch.randn(K, 4)
    
    ensemble = estimator.generate_multi_trajectory_ensemble(
        initial_grid_h=init_h,
        dem=dem,
        initial_scalars=init_scalars,
        action_sequence=actions,
        n_ensemble_members=4
    )
    
    assert ensemble["n_members"] == 4
    assert ensemble["rollout_steps"] == 4
    assert len(ensemble["flooded_area_p10"]) == 4
    assert len(ensemble["flooded_area_median"]) == 4
    assert len(ensemble["flooded_area_p90"]) == 4
    
    # Percentile ordering: p10 <= median <= p90
    for k in range(K):
        assert ensemble["flooded_area_p10"][k] <= ensemble["flooded_area_median"][k] + 1e-4
        assert ensemble["flooded_area_median"][k] <= ensemble["flooded_area_p90"][k] + 1e-4


def test_comprehensive_benchmark_test_split(trained_model, tmp_path):
    test_dir = Path("data/trajectories/test")
    if not test_dir.exists() or not list(test_dir.glob("*.npz")):
        pytest.skip("Test trajectories not generated.")
        
    benchmarker = ComprehensiveBenchmark(trained_model, test_dir=str(test_dir))
    report = benchmarker.evaluate_test_split(flood_thresh=0.15, max_rollout_steps=3)
    
    print("\nComprehensive Benchmark Evaluation (Out-of-Distribution Test Set):")
    print(f"  - Test Samples Evaluated:      {report['test_sample_count']}")
    print(f"  - Flooded Area IoU (CSI):      {report['critical_success_index_iou']:.4f}")
    print(f"  - Depth MAE:                   {report['depth_mae_m']:.4f} m")
    print(f"  - Depth RMSE:                  {report['depth_rmse_m']:.4f} m")
    print(f"  - Road Accessibility Accuracy: {report['road_accessibility_accuracy']*100:.1f}%")
    print(f"  - 90% CI Empirical Coverage:   {report['ci_90_empirical_coverage']*100:.1f}%")
    print(f"  - Multi-Step Rollout RMSEs:    {report['multi_step_rollout_rmse']}")
    
    assert report["critical_success_index_iou"] >= 0.0
    assert report["depth_mae_m"] < 0.25, "Depth MAE should remain bounded on test set."
    assert report["road_accessibility_accuracy"] > 0.65, "Road accessibility classification should exceed baseline."
    assert report["ci_90_empirical_coverage"] > 0.50, "Uncertainty intervals should cover majority of test observations."
    
    # Save report
    out_json = tmp_path / "benchmark_report.json"
    benchmarker.save_benchmark_report(report, str(out_json))
    assert out_json.exists()
