"""
cityverse_sim.hydraulics.overland_2d
===================================
Vectorized 2D Diffusive Wave / Shallow Inundation Solver.
Solves the 2D non-linear diffusion equation for overland urban flood routing:
    dh/dt + d(uh)/dx + d(vh)/dy = R(x,y,t) - Q_inlet + Q_surcharge

Includes flux limiters to strictly prevent negative water depths and ensure
exact conservation of mass.
"""

from typing import Tuple, Optional
import numpy as np


class Overland2DSolver:
    """
    Vectorized 2D hydrodynamic overland flood routing solver on regular Cartesian grids.
    Uses Manning-based diffusive wave friction slope with sub-cycling CFL checks.
    """

    def __init__(
        self,
        dem: np.ndarray,
        manning_n: np.ndarray,
        dx: float = 50.0,
        dy: float = 50.0,
        river_mask: Optional[np.ndarray] = None,
    ):
        self.dem = dem.astype(np.float32)
        self.manning_n = np.maximum(manning_n.astype(np.float32), 0.010)
        self.dx = float(dx)
        self.dy = float(dy)
        self.cell_area = self.dx * self.dy
        self.shape = dem.shape
        self.rows, self.cols = self.shape
        
        self.river_mask = river_mask if river_mask is not None else np.zeros(self.shape, dtype=bool)
        
        # State arrays
        self.h = np.zeros(self.shape, dtype=np.float32) # Surface water depth [m]
        self.u = np.zeros(self.shape, dtype=np.float32) # Velocity X [m/s]
        self.v = np.zeros(self.shape, dtype=np.float32) # Velocity Y [m/s]
        
        # Free-draining boundary mask (domain boundaries on the river channel)
        self.outfall_boundary_mask = np.zeros(self.shape, dtype=bool)
        self.outfall_boundary_mask[0, :] = self.river_mask[0, :]
        self.outfall_boundary_mask[-1, :] = self.river_mask[-1, :]
        self.outfall_boundary_mask[:, 0] = self.river_mask[:, 0]
        self.outfall_boundary_mask[:, -1] = self.river_mask[:, -1]
        
        # Cumulative mass tracking (in cubic meters)
        self.cumulative_inflow_vol_m3 = 0.0
        self.cumulative_boundary_outflow_vol_m3 = 0.0
        self.cumulative_drainage_exchange_vol_m3 = 0.0

    def reset(self):
        """Clears water depth and mass balance tracking."""
        self.h.fill(0.0)
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.cumulative_inflow_vol_m3 = 0.0
        self.cumulative_boundary_outflow_vol_m3 = 0.0
        self.cumulative_drainage_exchange_vol_m3 = 0.0

    def compute_max_stable_dt(self, cfl_factor: float = 0.5) -> float:
        """
        Estimates the maximum stable time step based on the diffusion stability condition:
            dt <= cfl * dx / sqrt(g * h_max)
        """
        g = 9.81
        h_max = float(np.max(self.h))
        if h_max < 0.005:
            return 30.0 # High stable step when mostly dry
            
        c_wave = np.sqrt(g * h_max)
        v_max = max(float(np.max(np.abs(self.u))), float(np.max(np.abs(self.v))), 0.1)
        dt_cfl = cfl_factor * min(self.dx, self.dy) / (c_wave + v_max)
        return float(np.clip(dt_cfl, 2.0, 30.0))

    def step(
        self,
        dt_s: float,
        effective_runoff_rate_m_s: np.ndarray,
        net_drainage_exchange_rate_m_s: Optional[np.ndarray] = None
    ) -> float:
        """
        Advances the 2D overland flow field by dt_s using sub-cycling.
        
        Args:
            dt_s: Outer simulation step in seconds
            effective_runoff_rate_m_s: (H, W) net water rate added from rainfall [m/s]
            net_drainage_exchange_rate_m_s: (H, W) rate transferred to subsurface drainage [m/s]
                                            
        Returns:
            mass_balance_error_m3: Current instantaneous mass conservation residual [m3]
        """
        remaining_time = dt_s
        
        while remaining_time > 1e-4:
            sub_dt = min(remaining_time, self.compute_max_stable_dt())
            self._sub_step(sub_dt, effective_runoff_rate_m_s, net_drainage_exchange_rate_m_s)
            remaining_time -= sub_dt
            
        return self.compute_mass_balance_residual()

    def _sub_step(
        self,
        dt: float,
        runoff_rate: np.ndarray,
        drainage_rate: Optional[np.ndarray]
    ):
        """Single explicit finite-volume update with flux limiting."""
        # 1. Source / Sink term update
        source_depth = runoff_rate * dt
        self.cumulative_inflow_vol_m3 += float(np.sum(source_depth) * self.cell_area)
        self.h += source_depth
        
        if drainage_rate is not None:
            drain_depth = drainage_rate * dt
            # Positive drain_depth removes water; negative adds water (surcharge)
            actual_drain = np.where(
                drain_depth > 0,
                np.minimum(drain_depth, self.h),
                drain_depth
            )
            self.h -= actual_drain
            self.cumulative_drainage_exchange_vol_m3 += float(np.sum(actual_drain) * self.cell_area)
            
        self.h = np.maximum(0.0, self.h)
        
        # 2. Water Surface Elevation (Hydraulic Head)
        eta = self.dem + self.h # [m]
        
        # 3. Inter-cell flux calculations along X (horizontal / East-West)
        # Interface between (r, c) and (r, c+1)
        eta_left = eta[:, :-1]
        eta_right = eta[:, 1:]
        z_left = self.dem[:, :-1]
        z_right = self.dem[:, 1:]
        h_left = self.h[:, :-1]
        h_right = self.h[:, 1:]
        
        # Hydraulic gradient
        dh_dx = (eta_left - eta_right) / self.dx
        
        # Flow depth: effective depth above highest bed level
        z_crest = np.maximum(z_left, z_right)
        eta_high = np.maximum(eta_left, eta_right)
        h_flow_x = np.maximum(0.0, eta_high - z_crest)
        
        # Threshold: flow occurs only when upstream cell has at least 0.1 mm
        flow_threshold = 1e-4 # 0.1 mm
        valid_x = np.where(
            dh_dx > 0,
            h_left > flow_threshold,
            np.where(dh_dx < 0, h_right > flow_threshold, False)
        ) & (h_flow_x > flow_threshold)
        
        n_x = 0.5 * (self.manning_n[:, :-1] + self.manning_n[:, 1:])
        slope_term_x = np.sign(dh_dx) * np.sqrt(np.abs(dh_dx))
        q_x = np.zeros_like(dh_dx)
        q_x[valid_x] = (1.0 / n_x[valid_x]) * (h_flow_x[valid_x]**(5.0/3.0)) * slope_term_x[valid_x]
        Q_x = q_x * self.dy # [m3/s]
        
        # 4. Inter-cell flux calculations along Y (vertical / North-South)
        eta_up = eta[:-1, :]
        eta_down = eta[1:, :]
        z_up = self.dem[:-1, :]
        z_down = self.dem[1:, :]
        h_up = self.h[:-1, :]
        h_down = self.h[1:, :]
        
        dh_dy = (eta_up - eta_down) / self.dy
        z_crest_y = np.maximum(z_up, z_down)
        eta_high_y = np.maximum(eta_up, eta_down)
        h_flow_y = np.maximum(0.0, eta_high_y - z_crest_y)
        
        valid_y = np.where(
            dh_dy > 0,
            h_up > flow_threshold,
            np.where(dh_dy < 0, h_down > flow_threshold, False)
        ) & (h_flow_y > flow_threshold)
        
        n_y = 0.5 * (self.manning_n[:-1, :] + self.manning_n[1:, :])
        slope_term_y = np.sign(dh_dy) * np.sqrt(np.abs(dh_dy))
        q_y = np.zeros_like(dh_dy)
        q_y[valid_y] = (1.0 / n_y[valid_y]) * (h_flow_y[valid_y]**(5.0/3.0)) * slope_term_y[valid_y]
        Q_y = q_y * self.dx # [m3/s]
        
        # 5. Flux Limiter to ensure volume out <= available volume
        vol_out = np.zeros(self.shape, dtype=np.float32)
        
        # Outflow east: Q_x > 0 from (:, :-1)
        vol_out[:, :-1] += np.maximum(0.0, Q_x) * dt
        # Outflow west: Q_x < 0 from (:, 1:)
        vol_out[:, 1:] += np.maximum(0.0, -Q_x) * dt
        # Outflow south: Q_y > 0 from (:-1, :)
        vol_out[:-1, :] += np.maximum(0.0, Q_y) * dt
        # Outflow north: Q_y < 0 from (1:, :)
        vol_out[1:, :] += np.maximum(0.0, -Q_y) * dt
        
        # Free-draining outfall boundaries (river mouth)
        boundary_vol_out = np.zeros(self.shape, dtype=np.float32)
        outfall_cells = self.outfall_boundary_mask & (self.h > flow_threshold)
        if np.any(outfall_cells):
            c_weir = 1.4
            q_boundary = c_weir * (self.h[outfall_cells]**1.5) * self.dx
            boundary_vol_out[outfall_cells] = q_boundary * dt
            vol_out[outfall_cells] += boundary_vol_out[outfall_cells]
            
        cell_vol = self.h * self.cell_area
        alpha = np.ones(self.shape, dtype=np.float32)
        excess = vol_out > (cell_vol * 0.98)
        alpha[excess] = (cell_vol[excess] * 0.98) / np.maximum(vol_out[excess], 1e-7)
        
        # Apply alpha limiting
        Q_x_lim = np.where(
            Q_x > 0,
            Q_x * alpha[:, :-1],
            Q_x * alpha[:, 1:]
        )
        Q_y_lim = np.where(
            Q_y > 0,
            Q_y * alpha[:-1, :],
            Q_y * alpha[1:, :]
        )
        
        actual_boundary_vol = boundary_vol_out * alpha
        self.cumulative_boundary_outflow_vol_m3 += float(np.sum(actual_boundary_vol))
        
        # 6. Apply Net Fluxes to Update Water Depth
        delta_vol = np.zeros(self.shape, dtype=np.float32)
        
        delta_vol[:, :-1] -= Q_x_lim * dt
        delta_vol[:, 1:] += Q_x_lim * dt
        
        delta_vol[:-1, :] -= Q_y_lim * dt
        delta_vol[1:, :] += Q_y_lim * dt
        
        delta_vol -= actual_boundary_vol
        
        self.h += delta_vol / self.cell_area
        self.h = np.maximum(0.0, self.h)
        
        # 7. Velocity diagnostics
        depth_safe = np.maximum(self.h, 0.01)
        flux_x_center = np.zeros(self.shape, dtype=np.float32)
        flux_x_center[:, 1:-1] = 0.5 * (Q_x_lim[:, :-1] + Q_x_lim[:, 1:]) / self.dy
        self.u = flux_x_center / depth_safe
        
        flux_y_center = np.zeros(self.shape, dtype=np.float32)
        flux_y_center[1:-1, :] = 0.5 * (Q_y_lim[:-1, :] + Q_y_lim[1:, :]) / self.dx
        self.v = flux_y_center / depth_safe

    def compute_mass_balance_residual(self) -> float:
        """
        Mass balance residual:
            Residual = V_inflow - V_boundary_outflow - V_drainage - V_stored
        """
        current_stored_vol = float(np.sum(self.h) * self.cell_area)
        net_conserved = (
            self.cumulative_inflow_vol_m3
            - self.cumulative_boundary_outflow_vol_m3
            - self.cumulative_drainage_exchange_vol_m3
        )
        return float(abs(current_stored_vol - net_conserved))
