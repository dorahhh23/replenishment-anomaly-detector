from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from forecast_analyzer import AnalysisSettings, analyze, build_views, read_source_excel, to_excel_bytes


@st.cache_data(show_spinner="正在读取并分析工作簿…")
def run_analysis(file_bytes: bytes, stock_method: str):
    source = read_source_excel(io.BytesIO(file_bytes))
    return analyze(source, AnalysisSettings(stock_method=stock_method))


@st.cache_data(show_spinner="正在生成 Excel…")
def build_export(result):
    return to_excel_bytes(result)


def highlight_return_risk(row: pd.Series) -> list[str]:
    if row.get("是否会引起断货问题") == "是":
        return ["background-color: #f4cccc; color: #9c0006"] * len(row)
    return [""] * len(row)


def select_page(page_name: str) -> None:
    st.session_state["analysis_page"] = page_name


st.set_page_config(page_title="补货预测异常识别", page_icon="📦", layout="wide")

if "analysis_page" not in st.session_state:
    st.session_state["analysis_page"] = "异常清单汇总"

with st.sidebar:
    st.header("页面导航")
    st.markdown("#### 总览")
    st.button(
        "异常清单汇总",
        key="nav_summary",
        type="primary" if st.session_state["analysis_page"] == "异常清单汇总" else "secondary",
        use_container_width=True,
        on_click=select_page,
        args=("异常清单汇总",),
    )
    st.caption("包含下方三类细分分析的全部记录")
    st.markdown("#### 细分分析")
    for detail_page in ("降级误补货问题", "退货异常问题", "偶发大单分析"):
        st.button(
            detail_page,
            key=f"nav_{detail_page}",
            type="primary" if st.session_state["analysis_page"] == detail_page else "secondary",
            use_container_width=True,
            on_click=select_page,
            args=(detail_page,),
        )
    page = st.session_state["analysis_page"]
    st.divider()
    st.header("计算设置")
    stock_label = st.selectbox(
        "有效库存口径",
        ["非限制库存 + 在途 + 在产", "当前可用库存 + 在途 + 在产", "仅非限制库存"],
        help="默认口径包含在库、在途和在产，不会把退货扭曲量重复加回库存。",
    )
    stock_method = {
        "非限制库存 + 在途 + 在产": "on_hand_plus_pipeline",
        "当前可用库存 + 在途 + 在产": "available_plus_pipeline",
        "仅非限制库存": "on_hand_only",
    }[stock_label]

st.title("补货预测异常识别")
st.caption("上传补货预测 Excel，识别降级误补货、退货异常和疑似偶发性大单。")

uploaded = st.file_uploader("上传 .xlsx 文件", type=["xlsx"])

if uploaded is None:
    st.info("请上传工作簿。系统会自动读取两行表头，并排除13个月销量中的最后一个当前月。")
    st.stop()

try:
    result = run_analysis(uploaded.getvalue(), stock_method)
    views = build_views(result)
except Exception as exc:
    st.error(f"分析失败：{exc}")
    st.stop()

summary = result["summary"]
st.success(
    f"已读取 {summary['records']:,} 个物料。识别月份："
    f"{' → '.join(summary['monthly_columns'])}；已排除当前月 {summary['monthly_columns'][-1]}。"
)

cols = st.columns(4)
metrics = [
    ("汇总记录", len(views["异常清单汇总"])),
    ("降级误补货", summary["issue1"]),
    ("退货异常", summary["negative_last_month"]),
    ("偶发大单", summary["oneoff"]),
]
for col, (label, value) in zip(cols, metrics):
    col.metric(label, f"{value:,}")

excel = build_export(result)
st.download_button(
    "下载分析结果 (.xlsx)",
    data=excel,
    file_name="补货预测异常分析结果.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)

st.subheader(page)

if page == "异常清单汇总":
    st.info("本页汇总下方三个细分分析的全部记录。每条记录按问题类型归类，可进入对应细分页面查看统一字段和专项指标。")
    st.dataframe(views[page], use_container_width=True, hide_index=True)

elif page == "降级误补货问题":
    st.write("该问题适用于上一季度为A/B级但本季度为C级的产品。降级后计算预测月销量的历史数据范围由6个月增加为12个月，可能导致预测月销量过高，从而误补货。")
    st.dataframe(views[page], use_container_width=True, hide_index=True)

elif page == "退货异常问题":
    st.write("检查上个月是否发生大额退货/冲销导致预测销量远低于实际月销，从而引起缺货问题。")
    st.caption("红色强调行会引起断货问题，并优先显示；其余退货异常记录显示在下方。判定不考虑海运时间。")
    styled = views[page].style.apply(highlight_return_risk, axis=1).format(precision=2, na_rep="")
    st.dataframe(styled, use_container_width=True, hide_index=True)

elif page == "偶发大单分析":
    st.write("判定最近一个月销量是否为偶发性大单。")
    st.dataframe(views[page], use_container_width=True, hide_index=True)
