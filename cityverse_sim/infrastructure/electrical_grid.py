"""
cityverse_sim.infrastructure.electrical_grid
============================================
Electrical power grid topology and cascading failure simulator.
Models flood-induced substation trips and downstream asset power losses
(e.g., stormwater pump station shutdowns).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from cityverse_sim.city.grid import CityGrid
from cityverse_sim.city.network import FacilityRegistry


class ElectricalGridManager:
    """
    Manages electrical substations and outage propagation to dependent infrastructure.
    """

    def __init__(
        self,
        grid: CityGrid,
        facilities: FacilityRegistry,
    ):
        self.grid = grid
        self.facilities = facilities
        
        # Precompute substation cell locations
        self.substation_cells: Dict[str, Tuple[int, int]] = {}
        self._init_substations()

    def _init_substations(self):
        subs = self.facilities.get_by_category("electrical_substations")
        for sub in subs:
            sid = sub["id"]
            r, c = self.grid.coord_to_cell(sub["x"], sub["y"])
            self.substation_cells[sid] = (r, c)
            sub["is_operational"] = True
            sub["flood_depth_m"] = 0.0

    def update_grid_status(self, surface_water_m: np.ndarray) -> Dict[str, Any]:
        """
        Samples surface water at substation yards and trips breakers if threshold is exceeded.
        Propagates power loss to dependent stormwater pumps and facilities.
        
        Args:
            surface_water_m: (H, W) surface water depth [m]
            
        Returns:
            Dict containing substation statuses and tripped assets.
        """
        subs = self.facilities.get_by_category("electrical_substations")
        tripped_substations = []
        
        for sub in subs:
            sid = sub["id"]
            r, c = self.substation_cells[sid]
            # Sample max depth in immediate yard vicinity (3x3 cell neighborhood)
            r_min, r_max = max(0, r - 1), min(self.grid.rows, r + 2)
            c_min, c_max = max(0, c - 1), min(self.grid.cols, c + 2)
            local_depth = float(np.max(surface_water_m[r_min:r_max, c_min:c_max]))
            
            sub["flood_depth_m"] = local_depth
            thresh = float(sub.get("flood_vulnerability_threshold_m", 0.25))
            
            if local_depth >= thresh:
                sub["is_operational"] = False
                tripped_substations.append(sid)
            else:
                sub["is_operational"] = True

        # Propagate power outage to pumping stations
        pumps = self.facilities.get_by_category("pumping_stations")
        disabled_pumps = []
        
        for pump in pumps:
            pid = pump["id"]
            power_source = pump.get("powered_by")
            if power_source and power_source in self.facilities.facilities:
                is_powered = self.facilities.facilities[power_source]["is_operational"]
                pump["is_operational"] = is_powered
                if not is_powered:
                    disabled_pumps.append(pid)
            else:
                pump["is_operational"] = True
                
        # Update hospitals (hospitals remain operational via backup generators, but log emergency power)
        hospitals = self.facilities.get_by_category("hospitals")
        for hosp in hospitals:
            hosp["on_backup_generator"] = bool(tripped_substations)
            
        return {
            "total_substations": len(subs),
            "operational_substations": len(subs) - len(tripped_substations),
            "tripped_substations": tripped_substations,
            "disabled_pumps": disabled_pumps,
            "grid_reliability_pct": ((len(subs) - len(tripped_substations)) / max(1, len(subs))) * 100.0,
        }
