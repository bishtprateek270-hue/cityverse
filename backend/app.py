"""
backend.app
===========
FastAPI REST and WebSocket Application Server for CityVerse.
Orchestrates high-speed World Model rollouts, ground-truth physics simulations,
counterfactual scenario forks, and live UI streaming.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import asyncio
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cityverse_sim.city.city_builder import MetroVerdiaCity
from cityverse_sim.interventions.state_manager import CityStateManager
from cityverse_sim.interventions.branch_engine import CounterfactualBranchEngine
from cityverse_ai.models.spatio_temporal_world_model import UrbanWorldModel
from cityverse_ai.uncertainty.estimator import UncertaintyEstimator
from cityverse_sim.real_world.calibration import ModelCalibrationProtocol, ModelProvenanceTier
from backend.nlp_parser import NaturalLanguageScenarioParser
from backend.timeline_cache import TimelineTrajectoryCache


app = FastAPI(
    title="CityVerse: Urban World Model API",
    description="Real-time interconnected urban flood and disaster dynamics simulation engine.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
city_instance: Optional[MetroVerdiaCity] = None
world_model_instance: Optional[UrbanWorldModel] = None
branch_engine: Optional[CounterfactualBranchEngine] = None
timeline_cache: TimelineTrajectoryCache = TimelineTrajectoryCache()


def init_engine():
    global city_instance, world_model_instance, branch_engine
    if city_instance is None:
        city_instance = MetroVerdiaCity("configs/city_spec.yaml")
        city_instance.build()
        
    # Attempt loading PyTorch World Model
    ckpt_path = Path("models/checkpoints/world_model_best.pt")
    if ckpt_path.exists() and world_model_instance is None:
        try:
            model = UrbanWorldModel(in_grid_channels=6, base_channels=24, scalar_dim=11, action_dim=4, num_road_edges=464)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            world_model_instance = model
        except Exception as e:
            print(f"Notice: Could not load trained world model: {e}")
            
    root_state = CityStateManager(city=city_instance, world_model=world_model_instance)
    branch_engine = CounterfactualBranchEngine(root_state)
    
    # Pre-populate baseline 3-hour storm trajectory so UI loads instantly with zero wait
    if not timeline_cache.branches.get("baseline"):
        try:
            timeline_cache.run_full_scenario(
                state_mgr=root_state,
                rainfall_mm_hr=120.0,
                duration_hours=3.0,
                drainage_capacity_factor=0.6,
                pump_overrides={"PUMP-1": 1.0, "PUMP-2": 1.0},
                engine_mode="physics",
                branch_id="baseline",
                step_dt_s=300.0
            )
        except Exception as e:
            print(f"Notice: Baseline pre-rollout deferred: {e}")


@app.on_event("startup")
def startup_event():
    init_engine()


# --- Pydantic Request Schemas ---

class ScenarioStartRequest(BaseModel):
    rainfall_mm_hr: float = Field(80.0, ge=0.0, le=250.0)
    duration_hours: float = Field(2.0, ge=0.5, le=12.0)
    drainage_capacity_factor: float = Field(1.0, ge=0.0, le=1.0)
    pump_overrides: Optional[Dict[str, float]] = None
    engine_mode: str = Field("world_model", description="'world_model' or 'physics'")


class StepRequest(BaseModel):
    branch_id: str = "baseline"
    engine_mode: str = "world_model" # "world_model" or "physics"
    dt_s: float = 300.0


class InterventionRequest(BaseModel):
    branch_id: str = "baseline"
    rainfall_mm_hr: Optional[float] = None
    drainage_capacity_factor: Optional[float] = None
    pump_overrides: Optional[Dict[str, float]] = None
    block_road: Optional[List[str]] = None # [u, v]


class ForkRequest(BaseModel):
    source_branch_id: str = "baseline"
    new_branch_id: str
    intervention: Optional[Dict[str, Any]] = None


class CompareRequest(BaseModel):
    branch_a_id: str
    branch_b_id: str


class NLPParseRequest(BaseModel):
    prompt: str


# --- REST API Endpoints ---

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "city": "MetroVerdia",
        "has_world_model": world_model_instance is not None,
    }


@app.get("/api/real_world/provenance")
def get_provenance():
    protocol = ModelCalibrationProtocol(city_name="MetroVerdia")
    return protocol.get_provenance_manifest()


@app.get("/api/city/info")
def get_city_info():
    init_engine()
    summary_path = Path("data/raw_city_gis/city_summary.json")
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"city_name": "MetroVerdia", "grid_dimensions": [120, 120]}


@app.get("/api/city/terrain")
def get_city_terrain():
    init_engine()
    # Return downsampled or full DEM and river mask
    dem = city_instance.grid.dem
    river = city_instance.grid.river_mask
    return {
        "rows": dem.shape[0],
        "cols": dem.shape[1],
        "min_elevation_m": float(dem.min()),
        "max_elevation_m": float(dem.max()),
        "elevation_grid": dem.tolist(),
        "river_mask": river.tolist(),
    }


@app.get("/api/city/roads")
def get_city_roads():
    road_path = Path("data/raw_city_gis/road_network.geojson")
    if road_path.exists():
        with open(road_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Road network GeoJSON not found.")


@app.get("/api/city/facilities")
def get_city_facilities():
    fac_path = Path("data/raw_city_gis/critical_facilities.geojson")
    if fac_path.exists():
        with open(fac_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Facilities GeoJSON not found.")


@app.get("/api/city/buildings")
def get_city_buildings():
    bldg_path = Path("data/raw_city_gis/urban_buildings.geojson")
    if bldg_path.exists():
        with open(bldg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Buildings GeoJSON not found.")


@app.get("/api/city/districts")
def get_city_districts():
    dist_path = Path("data/raw_city_gis/urban_districts.geojson")
    if dist_path.exists():
        with open(dist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Districts GeoJSON not found.")


@app.get("/api/city/hillshade")
def get_city_hillshade():
    hs_path = Path("data/raw_city_gis/dem_hillshade.npy")
    if hs_path.exists():
        hs = np.load(hs_path)
        return {"rows": hs.shape[0], "cols": hs.shape[1], "hillshade_grid": hs.tolist()}
    raise HTTPException(status_code=404, detail="Hillshade layer not found.")


@app.get("/api/city/blocks")
def get_city_blocks():
    blocks_path = Path("data/raw_city_gis/urban_blocks.geojson")
    if blocks_path.exists():
        with open(blocks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Urban blocks GeoJSON not found.")


@app.get("/api/city/footprints")
def get_city_footprints():
    fp_path = Path("data/raw_city_gis/building_footprints.geojson")
    if fp_path.exists():
        with open(fp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Building footprints GeoJSON not found.")


# --- Geographic WGS84 GIS Endpoints for Leaflet Digital Twin ---

@app.get("/api/city/geo/river")
def get_city_geo_river():
    path = Path("data/raw_city_gis/river_wgs84.geojson")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="River GeoJSON not found.")


@app.get("/api/city/geo/roads")
def get_city_geo_roads():
    path = Path("data/raw_city_gis/road_network_wgs84.geojson")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Road network GeoJSON not found.")


@app.get("/api/city/geo/facilities")
def get_city_geo_facilities():
    path = Path("data/raw_city_gis/critical_facilities_wgs84.geojson")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Facilities GeoJSON not found.")


@app.get("/api/city/geo/blocks")
def get_city_geo_blocks():
    path = Path("data/raw_city_gis/urban_blocks_wgs84.geojson")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Urban blocks GeoJSON not found.")


@app.get("/api/city/geo/footprints")
def get_city_geo_footprints():
    path = Path("data/raw_city_gis/building_footprints_wgs84.geojson")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Footprints GeoJSON not found.")


@app.get("/api/city/geo/contours")
def get_city_geo_contours():
    path = Path("data/raw_city_gis/elevation_contours_wgs84.geojson")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Contours GeoJSON not found.")


@app.get("/api/city/geo/bounds")
def get_city_geo_bounds():
    return {
        "lat_min": 29.7200,
        "lat_max": 29.77412,
        "lon_min": -95.4000,
        "lon_max": -95.33792,
        "center": [29.74706, -95.36896],
        "domain_name": "MetroVerdia",
        "extent_km": [6.0, 6.0]
    }


@app.post("/api/simulation/run_scenario")
def run_scenario(req: ScenarioStartRequest):
    init_engine()
    root_state = branch_engine.branches.get("baseline")
    summary = timeline_cache.run_full_scenario(
        state_mgr=root_state,
        rainfall_mm_hr=req.rainfall_mm_hr,
        duration_hours=req.duration_hours,
        drainage_capacity_factor=req.drainage_capacity_factor,
        pump_overrides=req.pump_overrides,
        engine_mode=req.engine_mode,
        branch_id="baseline",
        step_dt_s=300.0
    )
    return summary


@app.get("/api/simulation/frame")
def get_simulation_frame(step_idx: int = 0, branch_id: str = "baseline"):
    frame = timeline_cache.get_frame(frame_idx=step_idx, branch_id=branch_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found in timeline cache.")
    return frame


@app.get("/api/simulation/frames_all")
def get_all_simulation_frames(branch_id: str = "baseline"):
    frames = timeline_cache.get_all_frames(branch_id=branch_id)
    if not frames:
        raise HTTPException(status_code=404, detail="No frames cached for branch.")
    return {
        "branch_id": branch_id,
        "total_frames": len(frames),
        "frames": frames,
        "summary": timeline_cache.get_timeline_summary(branch_id)
    }


@app.get("/api/simulation/timeline")
def get_timeline(branch_id: str = "baseline"):
    return timeline_cache.get_timeline_summary(branch_id=branch_id)


@app.post("/api/scenario/parse")
def parse_nlp_prompt(req: NLPParseRequest):
    structured_params = NaturalLanguageScenarioParser.parse_prompt(req.prompt)
    return structured_params


@app.post("/api/simulation/start")
def start_simulation(req: ScenarioStartRequest):
    init_engine()
    root_state = branch_engine.branches["baseline"]
    root_state.current_time_s = 0.0
    root_state.step_idx = 0
    root_state.hydro.reset()
    root_state.active_rainfall_mm_hr = req.rainfall_mm_hr
    root_state.drainage_capacity_factor = req.drainage_capacity_factor
    root_state.pump_overrides = req.pump_overrides or {}
    root_state.manual_roadblocks.clear()
    root_state.history_scalars.clear()
    
    # Reset branch tree
    branch_engine.branches = {"baseline": root_state}
    branch_engine.branch_lineage = {"baseline": {"parent": None, "fork_time_s": 0.0, "interventions": {}}}
    
    return {"status": "started", "branch": "baseline", "params": req.model_dump() if hasattr(req, "model_dump") else req.dict()}


@app.post("/api/simulation/step")
def step_simulation(req: StepRequest):
    if req.branch_id not in branch_engine.branches:
        raise HTTPException(status_code=404, detail=f"Branch '{req.branch_id}' not found.")
        
    state = branch_engine.branches[req.branch_id]
    
    if req.engine_mode == "world_model" and state.world_model is not None:
        snapshot = state.step_world_model(dt_s=req.dt_s)
    else:
        snapshot = state.step_physics(dt_s=min(req.dt_s, 120.0))
        
    # Return water grid and road states for client map visualization
    h_surface = state.hydro.overland.h
    return {
        "snapshot": snapshot,
        "surface_depth_grid": h_surface.tolist(),
        "time_s": state.current_time_s,
    }


@app.post("/api/simulation/intervention")
def inject_intervention(req: InterventionRequest):
    if req.branch_id not in branch_engine.branches:
        raise HTTPException(status_code=404, detail=f"Branch '{req.branch_id}' not found.")
        
    state = branch_engine.branches[req.branch_id]
    interv = {}
    if req.rainfall_mm_hr is not None:
        interv["rainfall_mm_hr"] = req.rainfall_mm_hr
    if req.drainage_capacity_factor is not None:
        interv["drainage_capacity_factor"] = req.drainage_capacity_factor
    if req.pump_overrides is not None:
        interv["pump_overrides"] = req.pump_overrides
    if req.block_road and len(req.block_road) == 2:
        interv["block_road"] = (req.block_road[0], req.block_road[1])
        
    state.inject_intervention(interv)
    return {"status": "injected", "branch_id": req.branch_id, "interventions": interv}


@app.post("/api/simulation/predict_uncertainty")
def predict_uncertainty(req: StepRequest):
    if req.branch_id not in branch_engine.branches:
        raise HTTPException(status_code=404, detail=f"Branch '{req.branch_id}' not found.")
        
    state = branch_engine.branches[req.branch_id]
    if state.world_model is None:
        raise HTTPException(status_code=400, detail="World Model checkpoint not loaded.")
        
    estimator = UncertaintyEstimator(state.world_model, device=state.device)
    
    h_t = torch.from_numpy(state.hydro.overland.h).unsqueeze(0).unsqueeze(0).float().to(state.device)
    grid_in = torch.cat([h_t, state.dem_normalized], dim=1)
    
    last_scalar = state.history_scalars[-1] if state.history_scalars else {}
    scalar_vec = torch.tensor([
        state.active_rainfall_mm_hr,
        state.drainage_capacity_factor,
        last_scalar.get("flooded_area_km2_gt15cm", 0.0),
        last_scalar.get("flooded_area_km2_gt30cm", 0.0),
        float(np.sum(state.hydro.overland.h) * state.city.grid.dx * state.city.grid.dy),
        float(last_scalar.get("closed_roads_passenger", 0)),
        float(last_scalar.get("closed_roads_emergency", 0)),
        float(last_scalar.get("operational_substations", 3)),
        float(last_scalar.get("disabled_pumps", 0)),
        float(last_scalar.get("ems_coverage_pct", 100.0)),
        float(last_scalar.get("avg_ems_response_min", 5.0)),
    ], dtype=torch.float32, device=state.device).unsqueeze(0)
    
    p1 = state.pump_overrides.get("PUMP-1", 1.0)
    p2 = state.pump_overrides.get("PUMP-2", 1.0)
    action_vec = torch.tensor([
        state.active_rainfall_mm_hr,
        state.drainage_capacity_factor,
        p1,
        p2
    ], dtype=torch.float32, device=state.device).unsqueeze(0)
    
    res = estimator.predict_with_uncertainty(grid_in, scalar_vec, action_vec, n_samples=10, confidence_level=0.90)
    
    return {
        "mean_depth_grid": res["mean_depth"].tolist(),
        "std_depth_grid": res["std_depth"].tolist(),
        "lower_bound_grid": res["lower_bound_depth"].tolist(),
        "upper_bound_grid": res["upper_bound_depth"].tolist(),
        "road_closure_probabilities": res["road_closure_probabilities"].tolist(),
        "confidence_level": res["confidence_level"],
    }


@app.post("/api/branch/fork")
def fork_branch(req: ForkRequest):
    try:
        new_state = branch_engine.fork_branch(
            new_branch_id=req.new_branch_id,
            source_branch_id=req.source_branch_id,
            intervention=req.intervention
        )
        return {
            "status": "forked",
            "new_branch_id": req.new_branch_id,
            "source_branch_id": req.source_branch_id,
            "fork_time_s": new_state.current_time_s,
            "active_branches": list(branch_engine.branches.keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/branch/compare")
def compare_branches(req: CompareRequest):
    try:
        diff = branch_engine.compare_branches(req.branch_a_id, req.branch_b_id)
        # Convert numpy array to list for JSON serialization
        diff_payload = dict(diff)
        diff_payload["spatial_depth_diff_grid"] = diff["spatial_depth_diff_grid"].tolist()
        return diff_payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- WebSocket Streaming ---

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    init_engine()
    state = branch_engine.branches.get("baseline")
    
    try:
        while True:
            # Receive client command if any
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                cmd = data.get("cmd")
                if cmd == "set_rain":
                    state.inject_intervention({"rainfall_mm_hr": float(data.get("val", 0.0))})
                elif cmd == "set_drainage":
                    state.inject_intervention({"drainage_capacity_factor": float(data.get("val", 1.0))})
                elif cmd == "set_pump":
                    state.inject_intervention({"pump_overrides": {data.get("pump_id"): float(data.get("val", 1.0))}})
            except asyncio.TimeoutError:
                pass
                
            # Step model forward
            if state.world_model is not None:
                snap = state.step_world_model(dt_s=300.0)
            else:
                snap = state.step_physics(dt_s=60.0)
                
            payload = {
                "type": "frame",
                "snapshot": snap,
                "surface_depth_grid": state.hydro.overland.h.tolist(),
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.2) # ~5 FPS playback
    except WebSocketDisconnect:
        pass


# Static directory setup
static_dir = Path("backend/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="backend/static"), name="static_dir")
app.mount("/", StaticFiles(directory="backend/static", html=True), name="static")
