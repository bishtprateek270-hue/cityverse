"""
Script to generate pilot benchmark trajectory dataset for World Model training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cityverse_sim.environment.dataset_generator import DatasetGenerator

def main():
    print("Initializing DatasetGenerator...")
    gen = DatasetGenerator()
    # Sample a representative set of storm trajectories
    spec = gen.generate_scenario_list(n_train=6, n_val=2, n_test=2, seed=42)
    print("Generating simulation trajectories across train, val, and test splits...")
    manifest = gen.generate_and_save_all(scenario_spec=spec, dt_sim_s=60.0, record_interval_s=300.0)
    print("Dataset generation complete!")
    for split, info in manifest["splits"].items():
        print(f"  - Split '{split}': {info['count']} trajectories recorded.")

if __name__ == "__main__":
    main()
