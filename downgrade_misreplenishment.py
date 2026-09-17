from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DowngradeMisreplenishmentResult:
    ab_forecast: float
    c_forecast: float
    ab_replenishment: float
    c_replenishment: float
    avg_months_1_3: float
    avg_months_4_6: float
    avg_months_7_12: float
    is_issue: bool


class DowngradeMisreplenishmentAnalyzer:
    """Detect replenishment caused by expanding the sales window after downgrade."""

    @staticmethod
    def _weighted_forecast(sales: np.ndarray, is_c_level: bool) -> float:
        near_n = 6 if is_c_level else 3
        far_n = 6 if is_c_level else 3
        near = sales[:near_n]
        far = sales[near_n: near_n + far_n]
        if len(near) < near_n or len(far) < far_n or np.isnan(near).any() or np.isnan(far).any():
            return np.nan
        return float(0.6 * np.mean(near) + 0.4 * np.mean(far))

    def analyze(
        self,
        sales_recent_first: np.ndarray,
        current_level: str,
        previous_level: str,
        current_stock: float,
    ) -> DowngradeMisreplenishmentResult:
        ab_forecast = self._weighted_forecast(sales_recent_first, False)
        c_forecast = self._weighted_forecast(sales_recent_first, True)
        ab_replenishment = (
            max(0.0, 6.0 * ab_forecast - current_stock)
            if np.isfinite(ab_forecast) and np.isfinite(current_stock) else np.nan
        )
        c_replenishment = (
            max(0.0, 3.0 * c_forecast - current_stock)
            if np.isfinite(c_forecast) and np.isfinite(current_stock) else np.nan
        )
        avg_1_3 = self._average(sales_recent_first[:3])
        avg_4_6 = self._average(sales_recent_first[3:6])
        avg_7_12 = self._average(sales_recent_first[6:12])
        is_issue = bool(
            current_level in {"PC", "RC"}
            and previous_level in {"PA", "RA", "PB", "RB"}
            and np.isfinite(ab_forecast)
            and np.isfinite(c_forecast)
            and 6.0 * ab_forecast < current_stock < 3.0 * c_forecast
            and avg_7_12 > 2.25 * avg_1_3 + 1.25 * avg_4_6
        )
        return DowngradeMisreplenishmentResult(
            ab_forecast=ab_forecast,
            c_forecast=c_forecast,
            ab_replenishment=ab_replenishment,
            c_replenishment=c_replenishment,
            avg_months_1_3=avg_1_3,
            avg_months_4_6=avg_4_6,
            avg_months_7_12=avg_7_12,
            is_issue=is_issue,
        )

    @staticmethod
    def _average(values: np.ndarray) -> float:
        return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan

