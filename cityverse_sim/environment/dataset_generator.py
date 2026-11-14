"""
cityverse_sim.environment.dataset_generator
===========================================
Batch trajectory dataset generator using stratified parameter sampling.
Produces structured train/val/test splits for the learned World Model.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import numpy as np

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.environment.rainfall_generator import RainfallHyetograph
from cityverse_sim.environment.trajectory_recorder import TrajectoryRecorder


class DatasetGenerator:
    """
    Orchestrates batch generation of multi-scenario trajectories.
    """

    def __init__(
        self,
        city: Optional[MetroVerdiaCity] = None,
        output_base_dir: str = "data/trajectories"
    ):
        self.city = city or MetroVerdiaCity("configs/city_spec.yaml")
        if not hasattr(self.city.grid, "dem") or self.city.grid.dem.max() == 0:
            self.city.build()
            
        self.recorder = TrajectoryRecorder(self.city)
        self.base_dir = Path(output_base_dir)

    def generate_scenario_list(
        self,
        n_train: int = 10,
        n_val: int = 2,
        n_test: int = 3,
        seed: int = 42
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Samples stratified parameters across precipitation, drainage, and pump status.
        """
        np.random.seed(seed)
        
        def sample_params(split: str, count: int) -> List[Dict[str, Any]]:
            scenarios = []
            for i in range(count):
                if split == "test":
                    # OOD / Stress conditions: severe 130-160 mm/hr storms, heavy blockages
                    peak_i = float(np.random.uniform(125.0, 160.0))
                    duration = float(np.random.uniform(2.0, 4.0))
                    alpha_drain = float(np.random.uniform(0.15, 0.45))
                    pump_override = {"PUMP-1": float(np.random.choice([0.0, 0.5]))}
                else:
                    # Normal training/validation domain
                    peak_i = float(np.random.uniform(35.0, 115.0))
                    duration = float(np.random.uniform(1.5, 3.5))
                    alpha_drain = float(np.random.uniform(0.40, 1.0))
                    pump_override = None
                    
                r_skew = float(np.random.uniform(0.25, 0.65))
                profile = str(np.random.choice(["chicago", "triangular"]))
                
                scenarios.append({
                    "id": f"{split}_run_{i:04d}",
                    "peak_intensity_mm_hr": peak_i,
                    "duration_hours": duration,
                    "peak_ratio_r": r_skew,
                    "profile_type": profile,
                    "drainage_capacity_factor": alpha_drain,
                    "pump_override": pump_override,
                })
            return scenarios

        return {
            "train": sample_params("train", n_train),
            "val": sample_params("val", n_val),
            "test": sample_params("test", n_test),
        }

    def generate_and_save_all(
        self,
        scenario_spec: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        dt_sim_s: float = 60.0,
        record_interval_s: float = 300.0, # 5 min frames
    ) -> Dict[str, Any]:
        """
        Executes simulations and saves .npz files for all splits.
        """
        if scenario_spec is None:
            scenario_spec = self.generate_scenario_list(n_train=4, n_val=1, n_test=2)
            
        manifest = {"splits": {}}
        
        for split, scenario_list in scenario_spec.items():
            split_dir = self.base_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            
            for sc in scenario_list:
                run_id = sc["id"]
                hyetograph = RainfallHyetograph(
                    peak_intensity_mm_hr=sc["peak_intensity_mm_hr"],
                    duration_hours=sc["duration_hours"],
                    peak_ratio_r=sc["peak_ratio_r"],
                    profile_type=sc["profile_type"],
                )
                
                traj_data = self.recorder.run_trajectory(
                    hyetograph=hyetograph,
                    drainage_capacity_factor=sc["drainage_capacity_factor"],
                    pump_override=sc.get("pump_override"),
                    dt_sim_s=dt_sim_s,
                    record_interval_s=record_interval_s,
                )
                
                file_path = split_dir / f"{run_id}.npz"
                self.recorder.save_trajectory(traj_data, str(file_path))
                saved_paths.append(str(file_path))
                
            manifest["splits"][split] = {
                "count": len(saved_paths),
                "files": saved_paths
            }
            
        with open(self.base_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        return manifest

