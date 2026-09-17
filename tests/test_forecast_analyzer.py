import unittest

import numpy as np
import pandas as pd

from forecast_analyzer import AnalysisSettings, analyze, build_views


def make_frame(rows):
    months = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep.1"]
    columns = [
        ("物料", "物料"),
        ("物料", "物料描述"),
        ("产品基本信息", "产品组"),
        ("产品基本信息", "Current Stock level"),
        ("产品基本信息", "Q-1 Stock level"),
        ("产品基本信息", "库存月数"),
        *[("实际销量", month) for month in months],
        ("库存信息", "非限制库存"),
        ("库存信息", "当前可用库存"),
        ("在途在产信息", "在途+在产"),
    ]
    return pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))


class AnalyzerTests(unittest.TestCase):
    def test_detects_13_months_and_excludes_current(self):
        row = ["1", "A", "G", "RA", "RA", 6] + list(range(1, 14)) + [100, 90, 10]
        result = analyze(make_frame([row]))
        self.assertEqual(result["summary"]["monthly_columns"][-1], "Sep")
        self.assertEqual(result["full"].iloc[0]["Last Month Sales"], 12)

    def test_return_risk_uses_stock_coverage_and_effect_duration(self):
        # Oldest -> newest completed. The last completed month is -200; current month is ignored.
        sales = [100] * 11 + [-200] + [9999]
        row = ["2", "Return", "G", "RA", "RA", 6] + sales + [250, 250, 0]
        result = analyze(make_frame([row]), AnalysisSettings(positive_history_min=6))
        rec = result["full"].iloc[0]
        self.assertAlmostEqual(rec["Baseline Monthly Sales (P)"], 100.0)
        self.assertTrue(rec["Return_Risk"])
        self.assertEqual(rec["Last Month Sales"], -200)
        self.assertAlmostEqual(rec["Return Effect Duration (months)"], 6.0)
        self.assertAlmostEqual(rec["Stockout Months"], 3.5)

    def test_return_is_not_risk_when_stock_outlasts_return_effect(self):
        sales = [100] * 11 + [-200] + [0]
        row = ["7", "Covered", "G", "RA", "RA", 6] + sales + [700, 700, 0]
        rec = analyze(make_frame([row]))["full"].iloc[0]
        self.assertAlmostEqual(rec["Real MOS = S / P"], 7.0)
        self.assertFalse(rec["Return_Risk"])
        self.assertEqual(rec["Stockout Months"], 0.0)

    def test_oneoff_order_and_current_month_ignored(self):
        sales = [10] * 11 + [500] + [100000]
        row = ["3", "Order", "G", "RB", "RB", 5] + sales + [1000, 1000, 0]
        result = analyze(make_frame([row]))
        rec = result["full"].iloc[0]
        self.assertTrue(rec["Issue 3"])
        self.assertEqual(rec["Last Month Sales"], 500)
        self.assertAlmostEqual(rec["reference mean"], 10.0)
        self.assertAlmostEqual(rec["Excess sales"], 490.0)
        self.assertAlmostEqual(rec["ratio to mean"], 50.0)
        self.assertAlmostEqual(rec["suggested replacement (mean+std)"], 10.0)

    def test_issue1_downgrade(self):
        # Recent 6 months are low; months 7-12 are high enough to satisfy the explicit rule.
        completed_recent_first = [10, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100]
        sales_oldest_first = list(reversed(completed_recent_first)) + [0]
        # AB forecast 10 => 6*10=60; C forecast 46 => 3*46=138. Stock 100.
        row = ["4", "Downgrade", "G", "PC", "PA", 2] + sales_oldest_first + [100, 100, 0]
        result = analyze(make_frame([row]))
        rec = result["full"].iloc[0]
        self.assertTrue(rec["Issue 1"])
        self.assertAlmostEqual(rec["A/B Forecast Replenishment"], 0.0)
        self.assertAlmostEqual(rec["C Forecast Replenishment"], 38.0)
        issue1_view = build_views(result)["降级误补货问题"]
        self.assertIn("A/B级预测补货量", issue1_view.columns)
        self.assertIn("C级预测补货量", issue1_view.columns)

    def test_history_shortfall_is_not_flagged(self):
        sales = [0] * 9 + [10, 20, 30] + [0]
        row = ["5", "Sparse", "G", "RB", "RB", 5] + sales + [100, 100, 0]
        result = analyze(make_frame([row]))
        rec = result["full"].iloc[0]
        self.assertEqual(rec["One-off Order Status"], "")
        self.assertFalse(rec["Issue 3"])
        self.assertEqual(rec["Issue Type"], "")
        self.assertTrue(result["issues"].empty)

    def test_missing_stock_is_not_coerced_to_zero(self):
        sales = [100] * 11 + [-50] + [0]
        row = ["6", "Missing stock", "G", "RA", "RA", 6] + sales + [np.nan, np.nan, 0]
        rec = analyze(make_frame([row]))["full"].iloc[0]
        self.assertTrue(np.isnan(rec["Current Stock"]))
        self.assertEqual(rec["Return_Status"], "invalid stock")

    def test_reader_views_are_chinese_and_return_risk_is_first(self):
        risky = [100] * 11 + [-200] + [0]
        covered = [100] * 11 + [-200] + [0]
        rows = [
            ["8", "Risk", "G", "RA", "RA", 6] + risky + [250, 250, 0],
            ["9", "Covered", "G", "RA", "RA", 6] + covered + [700, 700, 0],
        ]
        views = build_views(analyze(make_frame(rows)))
        returns = views["退货异常问题"]
        self.assertEqual(returns.iloc[0]["是否会引起断货问题"], "是")
        self.assertEqual(returns.iloc[1]["是否会引起断货问题"], "否")
        self.assertNotIn("Excluded Current Month", returns.columns)
        self.assertIn("实际月销量（不含退货月）P", returns.columns)
        self.assertEqual(views["异常清单汇总"].columns[0], "问题类型")


if __name__ == "__main__":
    unittest.main()
