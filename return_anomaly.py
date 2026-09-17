from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReturnAnomalyResult:
    last_month_sales: float
    baseline_monthly_sales: float
    observed_forecast: float
    estimated_distortion: float
    real_mos: float
    theoretical_mos: float
    false_safety_gap: float
    effect_duration_months: float
    stockout_months: float
    is_risk: bool
    status: str
    reason: str
    recommended_action: str


class ReturnAnomalyAnalyzer:
    """Evaluate whether a negative last month distorts stock coverage enough to matter."""

    def __init__(self, minimum_history_months: int = 3) -> None:
        self.minimum_history_months = minimum_history_months

    @staticmethod
    def _weighted_forecast(sales: np.ndarray, is_c_level: bool) -> float:
        near_n = 6 if is_c_level else 3
        far_n = 6 if is_c_level else 3
        near = sales[:near_n]
        far = sales[near_n: near_n + far_n]
        if len(near) < near_n or len(far) < far_n or np.isnan(near).any() or np.isnan(far).any():
            return np.nan
        return float(0.6 * np.mean(near) + 0.4 * np.mean(far))

    @staticmethod
    def _baseline_without_negative_sales(sales: np.ndarray, is_c_level: bool) -> tuple[float, str]:
        near_n = 6 if is_c_level else 3
        far_n = 6 if is_c_level else 3
        near = sales[:near_n]
        far = sales[near_n: near_n + far_n]
        near = near[np.isfinite(near) & (near >= 0)]
        far = far[np.isfinite(far) & (far >= 0)]
        if len(near) == 0 or len(far) == 0:
            return np.nan, "insufficient history in weighted forecast window"
        return float(0.6 * np.mean(near) + 0.4 * np.mean(far)), ""

    def analyze(
        self,
        sales_recent_first: np.ndarray,
        current_level: str,
        current_stock: float,
    ) -> ReturnAnomalyResult:
        is_c_level = current_level.endswith("C")
        observed = self._weighted_forecast(sales_recent_first, is_c_level)
        last_sales = sales_recent_first[0] if len(sales_recent_first) else np.nan
        effect_months = 12.0 if is_c_level else 6.0
        baseline = distortion = real_mos = theory_mos = gap = stockout_months = np.nan
        is_risk = False
        status = "not triggered"
        reason = "Last month sales are not negative"
        action = "Keep original historical sales"

        if np.isfinite(last_sales) and last_sales < 0:
            baseline, baseline_note = self._baseline_without_negative_sales(sales_recent_first, is_c_level)
            usable_count = int(np.sum(np.isfinite(sales_recent_first[1:]) & (sales_recent_first[1:] >= 0)))
            if usable_count < self.minimum_history_months or not np.isfinite(baseline):
                status = "insufficient history"
                reason = baseline_note or f"Only {usable_count} usable historical months"
                action = "Manual review: insufficient history"
            elif baseline <= 0:
                status = "invalid baseline"
                reason = "Baseline monthly sales are zero or negative"
                action = "Manual review: invalid baseline"
            elif not np.isfinite(current_stock):
                distortion = baseline - last_sales
                status = "invalid stock"
                reason = "Current stock is missing or non-numeric"
                action = "Manual review: invalid stock"
            elif current_stock <= 0:
                distortion = baseline - last_sales
                real_mos = current_stock / baseline
                status = "stock already depleted"
                reason = "Current stock is zero or negative"
                action = "Immediate stock review"
            else:
                distortion = baseline - last_sales
                real_mos = current_stock / baseline
                if np.isfinite(observed) and observed > 0:
                    theory_mos = current_stock / observed
                    gap = theory_mos - real_mos
                    status = "evaluated"
                else:
                    theory_mos = np.inf
                    gap = np.inf
                    status = "severe forecast distortion"
                    reason = "Observed forecast is zero or negative"
                is_risk = bool(real_mos < effect_months and real_mos < theory_mos)
                if is_risk:
                    stockout_months = max(0.0, min(effect_months, theory_mos) - real_mos)
                    reason = "Actual stock coverage is shorter than both the return-effect period and theoretical stock coverage"
                    action = "Correct/remove return distortion from historical sales"
                elif status == "evaluated":
                    stockout_months = 0.0
                    reason = "The revised stock-coverage conditions are not both satisfied"

        return ReturnAnomalyResult(
            last_month_sales=last_sales,
            baseline_monthly_sales=baseline,
            observed_forecast=observed,
            estimated_distortion=distortion,
            real_mos=real_mos,
            theoretical_mos=theory_mos,
            false_safety_gap=gap,
            effect_duration_months=effect_months,
            stockout_months=stockout_months,
            is_risk=is_risk,
            status=status,
            reason=reason,
            recommended_action=action,
        )

