from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

from downgrade_misreplenishment import DowngradeMisreplenishmentAnalyzer
from oneoff_order_analysis import OneOffOrderAnalyzer
from return_anomaly import ReturnAnomalyAnalyzer


MONTH_NAMES = {
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep",
    "sept", "september", "oct", "october", "nov", "november", "dec",
    "december",
}


@dataclass(frozen=True)
class AnalysisSettings:
    positive_history_min: int = 3
    stock_method: str = "on_hand_plus_pipeline"


@dataclass(frozen=True)
class WorkbookSchema:
    sku: tuple[str, str]
    description: tuple[str, str] | None
    product_group: tuple[str, str] | None
    product_group_description: tuple[str, str] | None
    current_level: tuple[str, str]
    prior_level: tuple[str, str]
    target_mos: tuple[str, str] | None
    on_hand: tuple[str, str] | None
    available_stock: tuple[str, str] | None
    pipeline: tuple[str, str] | None
    in_transit: tuple[str, str] | None
    in_production: tuple[str, str] | None
    monthly_sales: tuple[tuple[str, str], ...]


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _base_header(value: object) -> str:
    return re.sub(r"\.\d+$", "", _clean(value))


def _is_month_header(value: object) -> bool:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return True
    text = _base_header(value).lower()
    if text in MONTH_NAMES:
        return True
    patterns = (
        r"^(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])$",
        r"^(?:19|20)\d{2}年(?:0?[1-9]|1[0-2])月$",
        r"^(?:0?[1-9]|1[0-2])月$",
    )
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def read_source_excel(source: str | BinaryIO | io.BytesIO, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read the supplied two-row grouped header without modifying the source workbook."""
    df = pd.read_excel(source, sheet_name=sheet_name, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(_clean(a), _clean(b)) for a, b in df.columns], names=["Section", "Field"]
    )
    return df.dropna(how="all").reset_index(drop=True)


def _find_column(
    columns: Iterable[tuple[str, str]],
    field_names: Iterable[str],
    section_names: Iterable[str] | None = None,
    required: bool = False,
) -> tuple[str, str] | None:
    fields = {_clean(x).lower() for x in field_names}
    sections = {_clean(x).lower() for x in section_names} if section_names else None
    for col in columns:
        section, field = _clean(col[0]).lower(), _base_header(col[1]).lower()
        if field in fields and (sections is None or section in sections):
            return col
    if required:
        raise ValueError(f"缺少必需字段：{', '.join(field_names)}")
    return None


def detect_schema(df: pd.DataFrame) -> WorkbookSchema:
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        raise ValueError("工作簿必须使用两行表头（第一行为分组，第二行为字段名）。")

    cols = list(df.columns)
    sales_cols = [
        col for col in cols
        if _clean(col[0]) == "实际销量" and _is_month_header(col[1])
    ]
    if len(sales_cols) < 13:
        raise ValueError(f"在“实际销量”分组中仅识别到 {len(sales_cols)} 个自然月列，需要至少 13 个。")
    # The template stores the rolling 13 months as one block. Keep the latest block
    # and exclude its last column later because it is the incomplete current month.
    sales_cols = sales_cols[-13:]

    prior_level = _find_column(cols, ["Q-1 Stock level", "上一季度 Stock level", "Previous Stock level"])
    if prior_level is None:
        for col in cols:
            field = _base_header(col[1])
            if re.fullmatch(r"Q(?:-?1|[1-4])\s+Stock\s+level", field, flags=re.IGNORECASE):
                prior_level = col
                break
    if prior_level is None:
        raise ValueError("缺少必需字段：上一季度 Stock level（例如 Q-1/Q1/Q2 Stock level）")

    return WorkbookSchema(
        sku=_find_column(cols, ["物料", "SAP Code", "SAP代码"], required=True),
        description=_find_column(cols, ["物料描述", "Material Description"]),
        product_group=_find_column(cols, ["产品组", "Product Group"]),
        product_group_description=_find_column(cols, ["产品组描述", "Product Group Description"]),
        current_level=_find_column(cols, ["Current Stock level", "当前库存等级"], required=True),
        prior_level=prior_level,
        target_mos=_find_column(cols, ["库存月数", "Target MOS"]),
        on_hand=_find_column(cols, ["非限制库存", "On Hand Stock"], ["库存信息"]),
        available_stock=_find_column(cols, ["当前可用库存", "Available Stock"], ["库存信息"]),
        pipeline=_find_column(cols, ["在途+在产", "在途在产"], ["在途在产信息"]),
        in_transit=_find_column(cols, ["在途数量", "In Transit"], ["在途在产信息"]),
        in_production=_find_column(cols, ["在产数量", "In Production"], ["在途在产信息"]),
        monthly_sales=tuple(sales_cols),
    )


def _num(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _series(df: pd.DataFrame, col: tuple[str, str] | None) -> pd.Series | None:
    return None if col is None else df[col]


def _stock_values(df: pd.DataFrame, schema: WorkbookSchema, method: str) -> pd.Series:
    idx = df.index
    pipeline = _num(_series(df, schema.pipeline), idx)
    if pipeline.isna().all():
        pipeline = _num(_series(df, schema.in_transit), idx).fillna(0) + _num(
            _series(df, schema.in_production), idx
        ).fillna(0)
    else:
        pipeline = pipeline.fillna(0)

    if method == "available_plus_pipeline":
        if schema.available_stock is None:
            raise ValueError("无法计算库存：未找到“当前可用库存”字段。")
        base = _num(_series(df, schema.available_stock), idx)
    elif method == "on_hand_only":
        if schema.on_hand is None:
            raise ValueError("无法计算库存：未找到“非限制库存”字段。")
        base = _num(_series(df, schema.on_hand), idx)
        pipeline = pd.Series(0.0, index=idx)
    else:
        if schema.on_hand is None:
            raise ValueError("无法计算库存：未找到“非限制库存”字段。")
        base = _num(_series(df, schema.on_hand), idx)

    return base + pipeline


def _target_mos(level: str, supplied: object) -> float:
    supplied_num = pd.to_numeric(pd.Series([supplied]), errors="coerce").iloc[0]
    if pd.notna(supplied_num):
        return float(supplied_num)
    level = _clean(level).upper()
    return {
        "RA": 6.0, "RB": 5.0, "RC": 4.0,
        "PA": 6.0, "PB": 3.0, "PC": 2.0,
    }.get(level, np.nan)


def analyze(df: pd.DataFrame, settings: AnalysisSettings | None = None) -> dict[str, object]:
    settings = settings or AnalysisSettings()
    schema = detect_schema(df)
    stock = _stock_values(df, schema, settings.stock_method)

    # Source runs oldest -> newest. The last month is current/incomplete and is excluded.
    month_cols = list(schema.monthly_sales)
    completed_cols = month_cols[:-1]
    completed_labels = [_base_header(c[1]) for c in completed_cols]
    current_month_label = _base_header(month_cols[-1][1])
    values = df.loc[:, completed_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values_recent = values[:, ::-1]

    downgrade_analyzer = DowngradeMisreplenishmentAnalyzer()
    return_analyzer = ReturnAnomalyAnalyzer(settings.positive_history_min)
    oneoff_analyzer = OneOffOrderAnalyzer(settings.positive_history_min)

    rows: list[dict[str, object]] = []
    for pos, (_, source_row) in enumerate(df.iterrows()):
        sales = values_recent[pos]
        level = _clean(source_row[schema.current_level]).upper()
        previous_level = _clean(source_row[schema.prior_level]).upper()
        target_mos = _target_mos(level, source_row[schema.target_mos] if schema.target_mos else np.nan)
        s = float(stock.iloc[pos])
        downgrade = downgrade_analyzer.analyze(sales, level, previous_level, s)
        return_analysis = return_analyzer.analyze(sales, level, s)
        oneoff = oneoff_analyzer.analyze(sales)

        sku = source_row[schema.sku]
        description = source_row[schema.description] if schema.description else ""
        product_group = source_row[schema.product_group] if schema.product_group else ""
        group_description = source_row[schema.product_group_description] if schema.product_group_description else ""
        issue_types = []
        if downgrade.is_issue:
            issue_types.append("降级误补货问题")
        if return_analysis.is_risk:
            issue_types.append("退货异常问题")
        if oneoff.is_issue:
            issue_types.append("偶发大单分析")
        rows.append({
            "SAP Code": sku,
            "Material Description": description,
            "Product Group": product_group,
            "Product Group Description": group_description,
            "Stock Level / Category": level,
            "Previous Quarter Stock Level": previous_level,
            "Target MOS": target_mos,
            "Current Stock": s,
            "Last Completed Month": completed_labels[-1],
            "Excluded Current Month": current_month_label,
            "Last Month Sales": return_analysis.last_month_sales,
            "A/B Forecast": downgrade.ab_forecast,
            "C Forecast": downgrade.c_forecast,
            "A/B Forecast Replenishment": downgrade.ab_replenishment,
            "C Forecast Replenishment": downgrade.c_replenishment,
            "Avg Months 1-3": downgrade.avg_months_1_3,
            "Avg Months 4-6": downgrade.avg_months_4_6,
            "Avg Months 7-12": downgrade.avg_months_7_12,
            "Issue 1": downgrade.is_issue,
            "Baseline Monthly Sales (P)": return_analysis.baseline_monthly_sales,
            "Observed Forecast (P')": return_analysis.observed_forecast,
            "Estimated Sales Distortion (R = P - LastMonthSales)": return_analysis.estimated_distortion,
            "Real MOS = S / P": return_analysis.real_mos,
            "Theoretical MOS = S / P'": return_analysis.theoretical_mos,
            "False Safety Gap": return_analysis.false_safety_gap,
            "Return Effect Duration (months)": return_analysis.effect_duration_months,
            "Stockout Months": return_analysis.stockout_months,
            "Return_Risk": return_analysis.is_risk,
            "Return_Status": return_analysis.status,
            "Risk_Reason": return_analysis.reason,
            "Recommended_Action": return_analysis.recommended_action,
            "One-off Order Status": oneoff.status,
            "reference mean": oneoff.reference_mean,
            "Excess sales": oneoff.excess_sales,
            "ratio to mean": oneoff.ratio_to_mean,
            "Robust Statistical Upper": oneoff.robust_upper,
            "Ratio Upper": oneoff.ratio_upper,
            "suggested replacement (mean+std)": oneoff.suggested_replacement,
            "Issue 3": oneoff.is_issue,
            "Issue Type": "；".join(issue_types),
        })

    full = pd.DataFrame(rows)
    issues = full.loc[full["Issue Type"].ne("")].copy()
    returns = full.loc[full["Last Month Sales"].lt(0)].copy()
    summary = {
        "records": len(full),
        "issues": len(issues),
        "issue1": int(full["Issue 1"].sum()),
        "return_risk": int(full["Return_Risk"].sum()),
        "oneoff": int(full["Issue 3"].sum()),
        "negative_last_month": len(returns),
        "monthly_columns": [_base_header(c[1]) for c in month_cols],
        "completed_columns": completed_labels,
    }
    return {"full": full, "issues": issues, "returns": returns, "summary": summary, "schema": schema}


CHINESE_LABELS = {
    "SAP Code": "SAP编码",
    "Material Description": "物料描述",
    "Product Group": "产品组",
    "Product Group Description": "产品组描述",
    "Stock Level / Category": "当前库存等级",
    "Previous Quarter Stock Level": "上一季度库存等级",
    "Target MOS": "目标库存月数",
    "Current Stock": "当前库存",
    "Last Completed Month": "最近完整月份",
    "Last Month Sales": "上月销量",
    "A/B Forecast": "A/B级预测月销量",
    "C Forecast": "C级预测月销量",
    "A/B Forecast Replenishment": "A/B级预测补货量",
    "C Forecast Replenishment": "C级预测补货量",
    "Avg Months 1-3": "最近1–3月平均销量",
    "Avg Months 4-6": "最近4–6月平均销量",
    "Avg Months 7-12": "最近7–12月平均销量",
    "Baseline Monthly Sales (P)": "实际月销量（不含退货月）P",
    "Observed Forecast (P')": "理论月销量（含退货月）P'",
    "Real MOS = S / P": "实际库存可用月数",
    "Theoretical MOS = S / P'": "理论库存可用月数",
    "Stockout Months": "断货月数",
    "Return_Risk": "是否会引起断货问题",
    "reference mean": "历史正销量平均值",
    "Excess sales": "超出历史平均的销量",
    "ratio to mean": "相对历史平均倍数",
    "suggested replacement (mean+std)": "建议替换值（平均值+标准差）",
}


def _localized(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns].rename(columns=CHINESE_LABELS).reset_index(drop=True)


def build_views(result: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Build the four Chinese reader-facing tables used by the app and export."""
    full: pd.DataFrame = result["full"]  # type: ignore[assignment]

    issue1_columns = [
        "SAP Code", "Material Description", "Product Group", "Product Group Description",
        "Stock Level / Category", "Previous Quarter Stock Level", "Current Stock",
        "A/B Forecast", "A/B Forecast Replenishment", "C Forecast", "C Forecast Replenishment",
        "Avg Months 1-3", "Avg Months 4-6", "Avg Months 7-12",
    ]
    issue1 = _localized(full.loc[full["Issue 1"]], issue1_columns)

    return_columns = [
        "SAP Code", "Material Description", "Product Group", "Product Group Description",
        "Stock Level / Category", "Previous Quarter Stock Level", "Target MOS", "Current Stock",
        "Last Completed Month", "Return_Risk", "Baseline Monthly Sales (P)",
        "Observed Forecast (P')", "Real MOS = S / P", "Theoretical MOS = S / P'", "Stockout Months",
    ]
    returns_source = full.loc[full["Last Month Sales"].lt(0)].copy()
    returns_source = returns_source.sort_values("Return_Risk", ascending=False, kind="stable")
    returns = _localized(returns_source, return_columns)
    returns["是否会引起断货问题"] = returns["是否会引起断货问题"].map({True: "是", False: "否"})

    issue3_columns = [
        "SAP Code", "Material Description", "Product Group", "Product Group Description",
        "Stock Level / Category", "Current Stock", "Last Month Sales", "reference mean",
        "Excess sales", "ratio to mean", "suggested replacement (mean+std)",
    ]
    issue3 = _localized(full.loc[full["Issue 3"]], issue3_columns)

    typed_frames = []
    for problem_type, frame in (
        ("降级误补货问题", issue1),
        ("退货异常问题", returns),
        ("偶发大单分析", issue3),
    ):
        typed = frame.copy()
        typed.insert(0, "问题类型", problem_type)
        typed_frames.append(typed)
    summary = pd.concat(typed_frames, ignore_index=True, sort=False)
    if not summary.empty:
        summary = summary.sort_values("问题类型", kind="stable").reset_index(drop=True)

    return {
        "异常清单汇总": summary,
        "降级误补货问题": issue1,
        "退货异常问题": returns,
        "偶发大单分析": issue3,
    }


def to_excel_bytes(result: dict[str, object]) -> bytes:
    output = io.BytesIO()
    views = build_views(result)

    descriptions = {
        "异常清单汇总": "本表汇总降级误补货、退货异常和偶发大单三个细分分析的全部记录。",
        "降级误补货问题": "该问题适用于上一季度为A/B级但本季度为C级的产品。降级后计算预测月销量的历史数据范围由6个月增加为12个月，可能导致预测月销量过高，从而误补货。",
        "退货异常问题": "检查上个月是否发生大额退货/冲销导致预测销量远低于实际月销，从而引起缺货问题。",
        "偶发大单分析": "判定最近一个月销量是否为偶发性大单。",
    }
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, data in views.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=3)
        workbook = writer.book
        title = workbook.add_format({"bold": True, "font_size": 14, "font_color": "#17365D"})
        description = workbook.add_format({"font_color": "#404040"})
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#17365D", "align": "center", "valign": "vcenter"})
        decimal = workbook.add_format({"num_format": "#,##0.00"})
        risk = workbook.add_format({"bg_color": "#F4CCCC", "font_color": "#9C0006"})
        all_sheets = [(name, data, 3) for name, data in views.items()]
        for sheet_name, data, startrow in all_sheets:
            ws = writer.sheets[sheet_name]
            ws.hide_gridlines(2)
            ws.write(0, 0, sheet_name, title)
            ws.write(1, 0, descriptions[sheet_name], description)
            ws.set_tab_color("#17365D" if sheet_name == "异常清单汇总" else "#9EADBF")
            ws.freeze_panes(startrow + 1, 1)
            ws.autofilter(startrow, 0, startrow + max(len(data), 1), max(len(data.columns) - 1, 0))
            ws.set_row(startrow, 28, header)
            for c, col in enumerate(data.columns):
                width = min(36, max(12, len(str(col)) * 2 + 2))
                fmt = decimal if any(key in str(col) for key in ("销量", "月数", "库存", "倍数", "数值")) else None
                ws.set_column(c, c, width, fmt)
            if "是否会引起断货问题" in data.columns and len(data):
                c = data.columns.get_loc("是否会引起断货问题")
                from xlsxwriter.utility import xl_col_to_name
                excel_row = startrow + 2
                formula = f'=${xl_col_to_name(c)}{excel_row}="是"'
                ws.conditional_format(startrow + 1, 0, startrow + len(data), len(data.columns) - 1, {"type": "formula", "criteria": formula, "format": risk})
    return output.getvalue()
