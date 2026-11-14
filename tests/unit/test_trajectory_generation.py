"""
Unit and integration tests for Phase 5:
- Chicago design storm hyetograph generator
- Trajectory recording harness (S_0, A_0, ..., S_T)
- Dataset generation and PyTorch Dataset / DataLoader ingestion
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.environment.rainfall_generator import RainfallHyetograph
from cityverse_sim.environment.trajectory_recorder import TrajectoryRecorder
from cityverse_sim.environment.dataset_generator import DatasetGenerator
from cityverse_ai.dataset import UrbanTrajectoryDataset


@pytest.fixture
def metro_city():
    city = MetroVerdiaCity("configs/city_spec.yaml")
    city.build(seed=42)
    return city


def test_chicago_hyetograph_properties():
    # 2-hour storm, peak 100 mm/hr at r = 0.40 (t = 48 minutes = 2880s)
    storm = RainfallHyetograph(
        peak_intensity_mm_hr=100.0,
        duration_hours=2.0,
        peak_ratio_r=0.40,
        profile_type="chicago"
    )
    
    # Peak intensity check
    peak_val = storm.get_intensity(storm.t_peak_s)
    assert abs(peak_val - 100.0) < 5.0
    
    # Intensity before and after peak should be lower
    i_early = storm.get_intensity(600.0) # 10 mins
    i_late = storm.get_intensity(6000.0) # 100 mins
    assert i_early < peak_val
    assert i_late < peak_val
    
    # Total depth check
    total_depth_mm = storm.get_total_depth_mm(dt_s=30.0)
    assert 30.0 < total_depth_mm < 150.0


def test_trajectory_recorder_single_run(metro_city, tmp_path):
    recorder = TrajectoryRecorder(metro_city)
    
    # Short 15-minute storm (900s) recorded every 3 minutes (180s) -> 6 frames
    storm = RainfallHyetograph(
        peak_intensity_mm_hr=60.0,
        duration_hours=0.25, # 15 minutes
        profile_type="block"
    )
    
    traj = recorder.run_trajectory(
        hyetograph=storm,
        drainage_capacity_factor=0.8,
        dt_sim_s=60.0,
        record_interval_s=180.0
    )
    
    # Check frame counts
    n_frames = len(traj["time_steps_s"])
    assert n_frames == 6
    
    # Check array shapes
    assert traj["grid_depths"].shape == (6, 120, 120)
    assert traj["road_depths"].shape == (6, len(metro_city.roads.graph.edges()))
    assert traj["system_scalars"].shape == (6, 11)
    assert traj["actions"].shape == (6, 4)
    
    # Check no NaNs or Infs
    assert not np.isnan(traj["grid_depths"]).any()
    assert not np.isnan(traj["system_scalars"]).any()
    assert not np.isnan(traj["actions"]).any()
    
    # Test saving and loading from disk
    out_file = tmp_path / "test_run.npz"
    recorder.save_trajectory(traj, str(out_file))
    assert out_file.exists()
    
    loaded = np.load(out_file)
    assert loaded["grid_depths"].shape == (6, 120, 120)


def test_dataset_generator_and_pytorch_loading(metro_city, tmp_path):
    gen = DatasetGenerator(city=metro_city, output_base_dir=str(tmp_path))
    
    # Generate miniature dataset: 2 train, 1 val, 1 test
    # Short duration (12 minutes, record every 3 minutes -> 5 frames)
    spec = {
        "train": [
            {"id": "train_0001", "peak_intensity_mm_hr": 50.0, "duration_hours": 0.2, "peak_ratio_r": 0.4, "profile_type": "block", "drainage_capacity_factor": 1.0},
            {"id": "train_0002", "peak_intensity_mm_hr": 70.0, "duration_hours": 0.2, "peak_ratio_r": 0.4, "profile_type": "block", "drainage_capacity_factor": 0.5},
        ],
        "val": [
            {"id": "val_0001", "peak_intensity_mm_hr": 60.0, "duration_hours": 0.2, "peak_ratio_r": 0.4, "profile_type": "block", "drainage_capacity_factor": 0.8},
        ],
        "test": [
            {"id": "test_0001", "peak_intensity_mm_hr": 100.0, "duration_hours": 0.2, "peak_ratio_r": 0.4, "profile_type": "block", "drainage_capacity_factor": 0.3},
        ]
    }
    
    manifest = gen.generate_and_save_all(
        scenario_spec=spec,
        dt_sim_s=60.0,
        record_interval_s=180.0
    )
    
    assert (tmp_path / "train" / "train_0001.npz").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    
    # Ingest into PyTorch Dataset
    train_dataset = UrbanTrajectoryDataset(
        trajectory_dir=str(tmp_path / "train"),
        rollout_horizon=1,
        dem_path="data/raw_city_gis/dem_elevation.npy"
    )
    
    assert len(train_dataset) > 0
    sample = train_dataset[0]
    
    # Validate tensor keys and dimensions
    assert sample["grid_state_t"].shape == (2, 120, 120) # [surface_h, dem]
    assert sample["target_grid_h"].shape == (1, 120, 120)
    assert sample["actions"].shape == (1, 4)
    assert sample["scalars_t"].shape == (11,)
    
    # Test DataLoader batching
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    assert batch["grid_state_t"].shape == (2, 2, 120, 120)
    assert batch["target_grid_h"].shape == (2, 1, 120, 120)
    assert batch["actions"].shape == (2, 1, 4)

