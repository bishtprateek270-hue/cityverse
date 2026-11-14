# CityVerse: Interconnected Urban Dynamics World Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Change one condition. Predict how the entire interconnected city responds."**

**CityVerse** is a scientifically grounded, geography-based urban digital twin and learned neural World Model $S_{t+1} = f_\theta(S_t, A_t)$. It simulates coupled multi-scale urban disaster dynamics—routing 2D shallow water flooding, dual-drainage pipe networks, electrical substation failure cascades, transportation gridlocks, and emergency medical services (EMS) routing at sub-3ms inference latency.

---

## Key Capabilities

1. **Top-Down Geographic Urban Digital Twin**:
   - Multi-directional analytical hillshaded topography with hypsometric elevation tinting.
   - Natural river channel, gravel riverbanks, drainage canals, and outfall basins.
   - Directed road hierarchy (Divided 4-lane expressways, arterials, local streets, elevated bridges).
   - 43 urban zoning blocks and 275 architectural building footprints with realistic setbacks.

2. **Coupled 2D/1D Hydrodynamics**:
   - 2D Diffusive wave shallow water overland routing with Manning friction slope and mass-conserving flux limiters ($<0.0001\%$ error).
   - Horton soil infiltration with land-use imperviousness partitioning ($20\% - 85\%$).
   - Dual-drainage curb inlet weir/orifice coupling and stormwater lift pump operations ($25\text{ m}^3/\text{s}$).

3. **Interconnected Infrastructure Cascades**:
   - Substation yard flood trip ($>0.20\text{m}$) de-energizing stormwater pumps.
   - Spatial road water sampling at 25m intervals with Pregnolato empirical speed curves ($v_{\max} = 0$ at $0.30\text{m}$).
   - BPR volume-delay traffic congestion and multi-base Dijkstra emergency hospital dispatch.

4. **Neural World Model ($f_\theta$)**:
   - Multi-scale residual U-Net predicting state deltas $\Delta h(t)$ with strictly enforced non-negative physics $\text{ReLU}(h_t + \Delta h) \ge 0$.
   - Evaluates a 5-minute urban state transition in **2.12 ms** ($>400\times$ faster than numerical solvers).

5. **Interactive Simulation Timeline & Scenario Branching**:
   - Bottom timeline scrubber for exploring 3-hour storm trajectories forward and backward.
   - Natural language scenario parser: e.g., *"Simulate 120 mm/hour rainfall for 3 hours with 40% drainage blockage"*.
   - Counterfactual timeline forking: simulate interventions (Branch B) and compute spatial A/B deltas.

---

## Repository Structure

```
cityverse/
├── backend/                  # FastAPI web server, WebSocket streamer, & timeline cache
│   ├── app.py                # REST endpoints and WebSocket orchestration
│   ├── nlp_parser.py         # Natural language prompt extractor
│   ├── timeline_cache.py     # Trajectory buffer and multi-step scenario runner
│   └── static/               # Geographical web dashboard (index.html, style.css, app.js)
├── cityverse_ai/             # PyTorch Neural World Model & Uncertainty Estimation
│   ├── models/               # SpatioTemporalWorldModel architecture
│   ├── losses/               # Physics-informed loss functions (Mass conservation, IoU)
│   ├── trainers/             # Model training pipelines
│   └── uncertainty/          # Monte Carlo Dropout & multi-trajectory ensemble estimators
├── cityverse_sim/            # Ground-truth numerical physics & infrastructure simulation
│   ├── city/                 # Grid generation, road/drainage networks, & geo-fabric
│   ├── hydraulics/           # 2D overland diffusive wave & 1D dual drainage coupled solver
│   ├── infrastructure/       # Electrical grid manager & substation failure models
│   ├── transportation/       # Pregnolato speed curves, BPR congestion, & routing
│   ├── emergency/            # Ambulance dispatch & trauma hospital routing
│   ├── real_world/           # Real DEM & OpenStreetMap ingestion protocols
│   └── interventions/        # State manager & counterfactual branching engine
├── configs/                  # City YAML specifications
├── data/                     # Raw GIS layers, elevation arrays, & training trajectories
├── models/                   # Trained PyTorch checkpoint weights (world_model_best.pt)
└── tests/                    # 46 automated unit, physics, and integration test suites
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/<your-username>/cityverse.git
cd cityverse
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

### 3. Run Test Suite

```bash
python -m pytest tests/unit/ -v
```
All 46 unit, hydrologic mass balance, and integration test suites will execute.

---

## License

MIT License. Developed for urban resilience, flood mitigation planning, and interconnected disaster response modeling.

