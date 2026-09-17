from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OneOffOrderResult:
    status: str
    reference_mean: float
    excess_sales: float
    ratio_to_mean: float
    robust_upper: float
    ratio_upper: float
    suggested_replacement: float
    is_issue: bool


class OneOffOrderAnalyzer:
    """Detect an unusually large most-recent completed month using robust history."""

    def __init__(self, minimum_positive_months: int = 3) -> None:
        self.minimum_positive_months = minimum_positive_months

    def analyze(self, sales_recent_first: np.ndarray) -> OneOffOrderResult:
        last_sales = sales_recent_first[0] if len(sales_recent_first) else np.nan
        history = sales_recent_first[1:12]
        positive_history = history[np.isfinite(history) & (history > 0)]
        status = "正常销量"
        reference_mean = excess_sales = ratio_to_mean = np.nan
        robust_upper = ratio_upper = replacement = np.nan
        is_issue = False

        if not np.isfinite(last_sales):
            status = ""
        elif last_sales < 0:
            status = "转问题2：上月负销量"
        elif len(positive_history) < self.minimum_positive_months:
            status = ""
        else:
            logs = np.log1p(positive_history)
            median_log = float(np.median(logs))
            mad = float(np.median(np.abs(logs - median_log)))
            robust_scale = max(1.4826 * mad, 0.2)
            robust_upper = float(np.expm1(median_log + 3.5 * robust_scale))
            ratio_upper = 2.0 * float(np.median(positive_history))
            reference_mean = float(np.mean(positive_history))
            excess_sales = float(last_sales - reference_mean)
            ratio_to_mean = float(last_sales / reference_mean) if reference_mean > 0 else np.nan
            replacement = (
                float(reference_mean + np.std(positive_history, ddof=1))
                if len(positive_history) > 1 else np.nan
            )
            is_issue = bool(last_sales > max(robust_upper, ratio_upper))
            if is_issue:
                status = "疑似偶发性大单"

        return OneOffOrderResult(
            status=status,
            reference_mean=reference_mean,
            excess_sales=excess_sales,
            ratio_to_mean=ratio_to_mean,
            robust_upper=robust_upper,
            ratio_upper=ratio_upper,
            suggested_replacement=replacement,
            is_issue=is_issue,
        )
