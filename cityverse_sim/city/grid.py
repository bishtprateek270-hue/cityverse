"""
cityverse_sim.city.grid
=======================
2D Spatially distributed grid representation for MetroVerdia.
Handles Digital Elevation Model (DEM), terrain slopes, river channel incision,
roughness, soil infiltration parameters, and coordinate transformations.
"""

from typing import Tuple, Dict, Any
import numpy as np


class CityGrid:
    """
    Represents the 2D spatial surface domain of MetroVerdia discretized into cells.
    Coordinates:
      - Continuous (x, y) in meters: x in [0, extent_x], y in [0, extent_y]
      - Discrete (row, col): row in [0, height-1] (corresponding to y from top to bottom),
                             col in [0, width-1] (corresponding to x from left to right)
    """

    def __init__(
        self,
        extent_x_m: float = 6000.0,
        extent_y_m: float = 6000.0,
        resolution_m: float = 50.0,
    ):
        self.extent_x = extent_x_m
        self.extent_y = extent_y_m
        self.dx = resolution_m
        self.dy = resolution_m
        
        self.cols = int(np.round(self.extent_x / self.dx))
        self.rows = int(np.round(self.extent_y / self.dy))
        
        # 1D coordinate vectors (cell centers)
        self.x_coords = np.linspace(self.dx / 2.0, self.extent_x - self.dx / 2.0, self.cols)
        # Note: row 0 is top (y_max), row rows-1 is bottom (y_min)
        self.y_coords = np.linspace(self.extent_y - self.dy / 2.0, self.dy / 2.0, self.rows)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
        # Core layers
        self.dem = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.river_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.manning_n = np.full((self.rows, self.cols), 0.035, dtype=np.float32)
        self.impervious_fraction = np.full((self.rows, self.cols), 0.30, dtype=np.float32)
        
        # Hydrological infiltration parameters (Horton model)
        self.horton_f0 = np.full((self.rows, self.cols), 30.0, dtype=np.float32) # mm/hr
        self.horton_fc = np.full((self.rows, self.cols), 5.0, dtype=np.float32)  # mm/hr
        self.horton_decay = np.full((self.rows, self.cols), 2.5, dtype=np.float32) # 1/hr

    def coord_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous (x, y) meters to (row, col) grid indices."""
        col = int(np.clip(np.floor(x / self.dx), 0, self.cols - 1))
        row = int(np.clip(np.floor((self.extent_y - y) / self.dy), 0, self.rows - 1))
        return row, col

    def cell_to_coord(self, row: int, col: int) -> Tuple[float, float]:
        """Convert (row, col) grid indices to center (x, y) coordinates in meters."""
        x = (col + 0.5) * self.dx
        y = self.extent_y - (row + 0.5) * self.dy
        return x, y

    def generate_terrain(
        self,
        min_elev: float = 12.0,
        max_elev: float = 68.0,
        seed: int = 42
    ):
        """
        Synthesizes a realistic geomorphology for MetroVerdia:
        1. Macro-scale regional slope from NW (highland) to SE (alluvial basin).
        2. Rolling hill ridges in the North.
        3. Meandering river valley incised across the city.
        """
        np.random.seed(seed)
        
        # 1. Macro slope: high in NW (x=0, y=6000), low in SE (x=6000, y=0)
        norm_x = self.X / self.extent_x
        norm_y = self.Y / self.extent_y
        
        # Distance from bottom-right corner (SE basin)
        dist_from_se = np.sqrt((1.0 - norm_x)**2 + norm_y**2)
        base_slope = min_elev + (max_elev - min_elev) * 0.70 * (dist_from_se / np.sqrt(2.0))
        
        # 2. Add rolling hills / ridges in the Northern quadrant
        ridge_1 = 12.0 * np.sin(norm_x * 4.5 * np.pi) * np.cos(norm_y * 3.0 * np.pi) * norm_y**1.5
        ridge_2 = 6.0 * np.cos(norm_x * 7.0 + norm_y * 5.0) * (norm_y > 0.5)
        
        # 3. Combine macro slope and ridges
        elevation = base_slope + ridge_1 + ridge_2
        
        # 4. Define and incise the Central River
        # River path parameterized by bezier/smooth curve
        river_pts = self._generate_river_spline()
        
        # Incise channel
        river_width = 50.0 # meters
        incision_depth = 3.5 # meters
        
        for pt in river_pts:
            rx, ry = pt
            dist_sq = (self.X - rx)**2 + (self.Y - ry)**2
            in_channel = dist_sq <= (river_width / 2.0)**2
            in_bank = (dist_sq > (river_width / 2.0)**2) & (dist_sq <= (river_width * 1.5)**2)
            
            # Carve channel bed
            elevation[in_channel] = np.minimum(elevation[in_channel], 14.0 + (5000.0 - ry) * 0.001)
            self.river_mask[in_channel] = True
            
            # Smooth banks
            bank_dist = np.sqrt(dist_sq[in_bank]) - (river_width / 2.0)
            elevation[in_bank] -= incision_depth * (1.0 - bank_dist / river_width)
            
        # Clip to defined bounds
        self.dem = np.clip(elevation, min_elev, max_elev).astype(np.float32)

    def _generate_river_spline(self, n_points: int = 400) -> np.ndarray:
        """Generates a smooth meandering spline from NW to SE."""
        t = np.linspace(0, 1, n_points)
        # Quadratic/Cubic Bezier curve with meanders
        p0 = np.array([400.0, 5800.0])
        p1 = np.array([2200.0, 4200.0])
        p2 = np.array([3400.0, 2600.0])
        p3 = np.array([5700.0, 300.0])
        
        # Cubic Bezier curve
        pts = (
            (1 - t)[:, None]**3 * p0
            + 3 * (1 - t)[:, None]**2 * t[:, None] * p1
            + 3 * (1 - t)[:, None] * t[:, None]**2 * p2
            + t[:, None]**3 * p3
        )
        
        # Add natural sinuous meanders
        meander = 120.0 * np.sin(t * 7.0 * np.pi)[:, None] * np.array([1.0, -0.8])
        return pts + meander

    def assign_zones(self, config_zones: Dict[str, Any]):
        """Assigns imperviousness, roughness, and Horton infiltration across zones."""
        # Default: Suburban
        sub = config_zones.get("suburban_residential", {})
        self.impervious_fraction.fill(sub.get("impervious_fraction", 0.45))
        self.manning_n.fill(sub.get("manning_n", 0.035))
        self.horton_f0.fill(sub.get("horton_f0_mm_hr", 35.0))
        self.horton_fc.fill(sub.get("horton_fc_mm_hr", 5.0))
        self.horton_decay.fill(sub.get("horton_decay_hr", 2.5))
        
        # Greenbelt (elevations > 40m or river corridor)
        gb = config_zones.get("greenbelt", {})
        gb_mask = (self.dem > 46.0) | (self.Y < 800.0)
        self.impervious_fraction[gb_mask] = gb.get("impervious_fraction", 0.10)
        self.manning_n[gb_mask] = gb.get("manning_n", 0.060)
        self.horton_f0[gb_mask] = gb.get("horton_f0_mm_hr", 75.0)
        self.horton_fc[gb_mask] = gb.get("horton_fc_mm_hr", 12.0)
        self.horton_decay[gb_mask] = gb.get("horton_decay_hr", 1.8)
        
        # Urban Core (bounded central rectangle)
        uc = config_zones.get("urban_core", {})
        bbox = uc.get("area_bbox", [2200, 1800, 4200, 3800])
        core_mask = (
            (self.X >= bbox[0]) & (self.X <= bbox[2]) &
            (self.Y >= bbox[1]) & (self.Y <= bbox[3])
        )
        self.impervious_fraction[core_mask] = uc.get("impervious_fraction", 0.85)
        self.manning_n[core_mask] = uc.get("manning_n", 0.015)
        self.horton_f0[core_mask] = uc.get("horton_f0_mm_hr", 10.0)
        self.horton_fc[core_mask] = uc.get("horton_fc_mm_hr", 2.0)
        self.horton_decay[core_mask] = uc.get("horton_decay_hr", 4.0)
        
        # River cells are 100% water conveyance
        self.impervious_fraction[self.river_mask] = 1.0
        self.manning_n[self.river_mask] = 0.025
        self.horton_f0[self.river_mask] = 0.0
        self.horton_fc[self.river_mask] = 0.0

    def calculate_slopes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates terrain gradient:
        dz/dx, dz/dy and total slope magnitude S_0 = sqrt((dz/dx)^2 + (dz/dy)^2).
        Uses central differences with one-sided differences at boundaries.
        """
        # dz/dx along columns
        dz_dx = np.zeros_like(self.dem)
        dz_dx[:, 1:-1] = (self.dem[:, 2:] - self.dem[:, :-2]) / (2.0 * self.dx)
        dz_dx[:, 0] = (self.dem[:, 1] - self.dem[:, 0]) / self.dx
        dz_dx[:, -1] = (self.dem[:, -1] - self.dem[:, -2]) / self.dx
        
        # dz/dy along rows (note: row index decreases as y increases)
        dz_dy = np.zeros_like(self.dem)
        dz_dy[1:-1, :] = (self.dem[:-2, :] - self.dem[2:, :]) / (2.0 * self.dy)
        dz_dy[0, :] = (self.dem[0, :] - self.dem[1, :]) / self.dy
        dz_dy[-1, :] = (self.dem[-2, :] - self.dem[-1, :]) / self.dy
        
        slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
        return dz_dx, dz_dy, slope_magnitude

