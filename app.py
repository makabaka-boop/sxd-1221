import os
import sys
import traceback
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.cleaning import clean_dataframe, REQUIRED_FIELDS
from modules.validation import run_all_validations, validate_raw_data, FIELD_LABELS
from modules.aggregation import (
    filter_dataframe, compute_overall_metrics,
    groupby_low_score_distribution, groupby_agent_load,
    groupby_channel_trend, get_pending_review_details
)
from modules.suggestions import generate_quality_suggestions
from modules.export import export_clean_data, export_full_report, generate_filename

st.set_page_config(
    page_title="客服响应质量分析看板",
    page_icon="📊",
    layout="wide"
)

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sample_data", "sample_customer_service.csv")

FIELD_ALIASES = {
    "record_date": ["日期", "date", "time", "record_time", "记录时间", "会话日期", "通话时间"],
    "channel_name": ["渠道", "channel", "来源", "platform", "服务渠道", "接入渠道"],
    "agent_name": ["坐席", "agent", "客服", "客服姓名", "员工", "处理人"],
    "response_seconds": ["响应时长", "响应时间", "response", "时长", "等待时长", "response_time", "等待时间"],
    "score": ["分数", "评分", "质量分", "质检分", "quality_score", "得分"],
    "issue_type": ["问题类型", "类型", "问题分类", "category", "工单类型", "咨询类型"],
    "solved_flag": ["是否解决", "解决状态", "resolved", "是否完结", "处理状态", "solved"],
    "note": ["备注", "说明", "note", "remark", "评语", "质检备注"]
}


def smart_field_mapping(uploaded_cols: list) -> dict:
    mapping = {f: "" for f in REQUIRED_FIELDS.keys()}
    uploaded_lower = {c.lower().strip(): c for c in uploaded_cols}

    for std_field in REQUIRED_FIELDS.keys():
        if std_field in uploaded_cols:
            mapping[std_field] = std_field
            continue
        if std_field in uploaded_lower:
            mapping[std_field] = uploaded_lower[std_field]
            continue
        for alias in FIELD_ALIASES.get(std_field, []):
            alias_lower = alias.lower()
            if alias in uploaded_cols:
                mapping[std_field] = alias
                break
            if alias_lower in uploaded_lower:
                mapping[std_field] = uploaded_lower[alias_lower]
                break
    return mapping


def safe_read_csv(uploaded_file):
    try:
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="gbk")
    except Exception as e:
        st.error(f"读取 CSV 文件失败：{str(e)}。请检查文件编码是否为 UTF-8 或 GBK。")
        return None


def _display_validation_banners(validations: dict):
    if validations["missing_fields"]:
        st.warning(
            "⚠️ **缺失字段提示**：以下关键字段未映射或全部为空 — "
            + "、".join([f"`{f}`（{REQUIRED_FIELDS.get(f, f)}）" for f in validations["missing_fields"]])
            + "，相关指标可能无法正常计算。"
        )

    dup = validations["duplicates"]
    if dup["duplicate_count"] > 0:
        st.warning(
            f"🔁 **重复记录检查**：发现 {dup['duplicate_count']} 条疑似重复记录，清洗阶段会自动去重。"
        )

    resp = validations["response_anomalies"]
    if resp["anomaly_count"] > 0:
        st.warning(
            f"⏱️ **响应时长异常**：{resp['anomaly_count']} 条记录超过 {resp['max_threshold']} 秒或为负数，建议复核。"
        )

    score = validations["score_format"]
    if score["invalid_count"] > 0 or score["out_of_range"] > 0:
        msgs = []
        if score["invalid_count"]:
            msgs.append(f"{score['invalid_count']} 条无法解析为数字")
        if score["out_of_range"]:
            msgs.append(f"{score['out_of_range']} 条超出 {score['score_min']}-{score['score_max']} 分区间")
        st.warning(f"🎯 **分数格式提醒**：共 {'、'.join(msgs)}，系统会尝试清洗并将无效值置空。")

    for field, err in validations["numeric_errors"].items():
        st.error(
            f"❌ **数值列解析异常** — 字段 `{field}`（{REQUIRED_FIELDS.get(field, field)}）：{err['hint']}。"
            f" 这些值将被置为缺失，不会导致页面崩溃。"
        )


def _render_metric_cards(metrics: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📋 总记录数", f"{metrics['total_records']:,}")
    avg_resp = metrics["avg_response_seconds"]
    c2.metric("⏱️ 平均响应时长", f"{avg_resp:.1f} 秒",
              delta="正常" if avg_resp <= 300 else "偏高",
              delta_color="inverse")
    c3.metric("✅ 一次解决率", f"{metrics['first_resolution_rate']:.2f} %")
    c4.metric("⭐ 平均质检分数", f"{metrics['avg_score']:.2f}",
              delta=f"低分{metrics['low_score_count']}条")
    c5.metric("📉 低分记录", f"{metrics['low_score_count']} 条")
    c6.metric("🔄 未解决", f"{metrics['unsolved_count']} 条")


def _plot_low_score_distribution(df_plot: pd.DataFrame):
    if df_plot.empty:
        st.info("近 60 分以下的低分记录。")
        return
    fig = px.bar(
        df_plot, x="issue_type", y="count",
        title="📊 低分问题类型分布（分数 < 60）",
        labels={"issue_type": "问题类型", "count": "低分数"},
        color="count", color_continuous_scale="Reds",
        text="count"
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


def _plot_agent_load(df_plot: pd.DataFrame):
    if df_plot.empty:
        st.info("无坐席负载数据。")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df_plot["agent_name"], y=df_plot["record_count"],
               name="服务记录数", marker_color="#4C78A8"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_plot["agent_name"], y=df_plot["avg_score"],
                   name="平均分数", mode="lines+markers",
                   line=dict(color="#E45756", width=2)),
        secondary_y=True
    )
    fig.update_layout(title="👥 坐席负载与平均分对比", showlegend=True,
                      xaxis_tickangle=-45)
    fig.update_yaxes(title_text="服务记录数", secondary_y=False)
    fig.update_yaxes(title_text="平均质检分数", secondary_y=True, range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)


def _plot_channel_trend(df_plot: pd.DataFrame):
    if df_plot.empty:
        st.info("无渠道趋势数据。")
        return
    fig = px.line(
        df_plot, x="record_date", y="count", color="channel_name",
        title="📈 渠道会话量趋势", markers=True,
        labels={"record_date": "日期", "count": "会话量", "channel_name": "渠道"}
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def _render_suggestions(suggestions: list):
    st.subheader("🎯 质检建议（基于当前筛选数据，近 30 天）")
    if not suggestions:
        st.success("✅ 近 30 天整体表现良好，暂无需要特别重点复核的对象。")
        return

    level_color = {"高": "🔴", "中": "🟡", "低": "🟢"}
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            with st.expander(f"{level_color.get(s['level'], '⚪')} "
                             f"[{s['category']}] {s['target']}（{s['level']}优先级）",
                             expanded=(s["level"] == "高")):
                st.markdown("**问题诊断：**")
                for reason in s["reasons"]:
                    st.markdown(f"- {reason}")
                st.markdown(f"**建议动作：** {s['action']}")


def main():
    st.title("📞 客服响应质量分析看板")
    st.caption("通话记录 / 在线会话记录 / 质检结果一站式分析，无需登录，即传即用")

    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = None
    if "raw_uploaded" not in st.session_state:
        st.session_state["raw_uploaded"] = None

    with st.sidebar:
        st.header("📂 数据上传")
        use_sample = st.button("🚀 一键加载示例数据", use_container_width=True, type="primary")
        if use_sample:
            try:
                with open(SAMPLE_PATH, "rb") as f:
                    st.session_state["raw_uploaded"] = pd.read_csv(f, encoding="utf-8-sig")
                st.success("✅ 示例数据已加载，共 {} 条记录".format(len(st.session_state["raw_uploaded"])))
            except Exception as e:
                st.error(f"加载示例数据失败：{e}")

        st.divider()
        uploaded_file = st.file_uploader("上传 CSV 文件（通话/会话/质检记录）", type=["csv"])
        if uploaded_file:
            try:
                st.session_state["raw_uploaded"] = safe_read_csv(uploaded_file)
                if st.session_state["raw_uploaded"] is not None:
                    st.success(f"✅ 已上传：{uploaded_file.name}，共 {len(st.session_state['raw_uploaded'])} 行")
            except Exception as e:
                st.error(f"文件读取异常：{str(e)}")
                st.session_state["raw_uploaded"] = None

        st.divider()
        st.caption("📌 支持字段：record_date / channel_name / agent_name / response_seconds / score / issue_type / solved_flag / note")

    raw_df = st.session_state.get("raw_uploaded")
    if raw_df is None:
        st.info("👈 请在左侧上传 CSV 文件，或点击「一键加载示例数据」开始体验。")

        with st.expander("📋 使用说明", expanded=True):
            st.markdown("""
            **支持字段说明：**

            | 标准字段 | 说明 | 示例 |
            |---------|------|------|
            | `record_date` | 记录日期时间 | 2026-06-20 14:30:00 |
            | `channel_name` | 服务渠道 | 电话客服 / 在线客服 / APP客服 |
            | `agent_name` | 坐席姓名 | 坐席01 / 张三 |
            | `response_seconds` | 响应时长（秒） | 45 / 120s / 60 |
            | `score` | 质检分数 | 85 / 90分 / 75 |
            | `issue_type` | 问题类型 | 账户问题 / 退款申请 |
            | `solved_flag` | 是否解决 | 是/否 / 1/0 / True/False |
            | `note` | 备注（可选） | 客户满意 / 需跟进 |

            **功能亮点：** 字段自动映射、缺失/重复/异常智能检查、数值列错误友好提示、多维筛选、Plotly 图表、自动生成质检建议。
            """)
        return

    with st.form("field_mapping_form"):
        st.subheader("🔗 字段映射")
        auto_map = smart_field_mapping(list(raw_df.columns))

        map_cols = st.columns(4)
        user_mapping = {}
        field_items = list(REQUIRED_FIELDS.items())
        for idx, (std_field, label) in enumerate(field_items):
            col = map_cols[idx % 4]
            default = auto_map.get(std_field, "")
            options = [""] + list(raw_df.columns)
            if default not in options:
                default = ""
            sel = col.selectbox(f"{label}", options,
                                index=options.index(default) if default in options else 0,
                                key=f"map_{std_field}")
            user_mapping[std_field] = sel

        submitted = st.form_submit_button("✅ 确认映射并开始清洗", type="primary", use_container_width=True)

    try:
        mapped_df = raw_df.rename(columns={v: k for k, v in user_mapping.items() if v}).copy()
        raw_validation = validate_raw_data(mapped_df)
    except Exception as e:
        st.error(f"原始数据预校验失败：{str(e)}")
        with st.expander("🔍 技术详情（仅供调试）"):
            st.code(traceback.format_exc())
        raw_validation = {"total_rows": len(raw_df), "issue_count": 0, "issues": []}

    with st.expander("🔍 原始数据预校验（清洗前，准确定位问题行）", expanded=True):
        if raw_validation["issue_count"] == 0:
            st.success("✅ 原始数据格式检查通过，未发现明显问题。")
        else:
            st.warning(f"⚠️ 发现 {raw_validation['issue_count']} 类数据质量问题，共 {raw_validation['total_rows']} 行记录：")
            for issue in raw_validation["issues"]:
                icon = "❌" if issue["level"] == "error" else "⚠️"
                with st.expander(f"{icon} {issue['title']}", expanded=(issue["level"] == "error")):
                    if not issue["detail_df"].empty:
                        st.dataframe(issue["detail_df"], use_container_width=True, hide_index=True)
                    st.caption(f"💡 {issue['tip']}")

    try:
        cleaned_df, clean_report = clean_dataframe(raw_df, user_mapping)
    except Exception as e:
        st.error(f"数据清洗过程出现异常：{str(e)}")
        with st.expander("🔍 技术详情（仅供调试）"):
            st.code(traceback.format_exc())
        return

    st.session_state["cleaned_df"] = cleaned_df

    try:
        validations = run_all_validations(cleaned_df)
    except Exception as e:
        st.warning(f"清洗后校验模块执行异常：{str(e)}，跳过校验提示。")
        validations = {"missing_fields": [], "duplicates": {"duplicate_count": 0},
                       "response_anomalies": {"anomaly_count": 0},
                       "score_format": {"invalid_count": 0, "out_of_range": 0},
                       "numeric_errors": {}}

    with st.expander("🧹 清洗后数据校验结果（点击展开/收起）", expanded=False):
        _display_validation_banners(validations)
        with st.expander("📝 清洗详细报告"):
            cr_cols = st.columns(len(clean_report))
            for col, (field, info) in zip(cr_cols, clean_report.items()):
                col.markdown(f"**{REQUIRED_FIELDS.get(field, field)}**")
                if isinstance(info, dict) and "failed_count" in info:
                    col.write(f"解析成功 {info.get('parsed_nonnull', 0)} / "
                              f"原始非空 {info.get('raw_nonnull', 0)}")
                    if info.get("failed_examples"):
                        col.caption("失败示例：" + "、".join(map(str, info["failed_examples"])))
                else:
                    col.write(f"有效 {info.get('valid', 0)} / "
                              f"无效 {info.get('invalid', 0)}")

    df = cleaned_df.copy()

    with st.expander("🧹 已清洗数据预览（前 50 行）", expanded=False):
        display_cols = [c for c in REQUIRED_FIELDS.keys() if c in df.columns]
        st.dataframe(df[display_cols].head(50), use_container_width=True)

    st.divider()
    st.subheader("🎛️ 多维筛选")
    fc = st.columns(6)
    date_range_val = fc[0].date_input("日期范围",
                                       value=(df["record_date"].min().date() if df["record_date"].notna().any() else None,
                                              df["record_date"].max().date() if df["record_date"].notna().any() else None),
                                       key="filter_date")
    channels = fc[1].multiselect("渠道", sorted(df["channel_name"].dropna().unique().tolist()), key="filter_chan")
    agents = fc[2].multiselect("坐席", sorted(df["agent_name"].dropna().unique().tolist()), key="filter_agent")
    issue_types = fc[3].multiselect("问题类型", sorted(df["issue_type"].dropna().unique().tolist()), key="filter_issue")
    score_vals = pd.to_numeric(df["score"], errors="coerce").dropna()
    s_min = int(score_vals.min()) if not score_vals.empty else 0
    s_max = int(score_vals.max()) if not score_vals.empty else 100
    score_range = fc[4].slider("分数区间", min_value=s_min, max_value=s_max, value=(s_min, s_max), key="filter_score")
    solved_status = fc[5].selectbox("解决状态", ["全部", "已解决", "未解决"], key="filter_solved")

    filters = {
        "date_range": date_range_val if all(date_range_val) else None,
        "channels": channels,
        "agents": agents,
        "issue_types": issue_types,
        "score_range": score_range,
        "solved_status": solved_status
    }

    filtered = filter_dataframe(df, filters)
    metrics = compute_overall_metrics(filtered)
    low_score_dist = groupby_low_score_distribution(filtered)
    agent_load = groupby_agent_load(filtered)
    channel_trend = groupby_channel_trend(filtered)
    pending_review = get_pending_review_details(filtered)
    suggestions = generate_quality_suggestions(filtered)

    st.divider()
    st.subheader("📊 核心指标概览")
    try:
        _render_metric_cards(metrics)
    except Exception as e:
        st.error(f"指标卡片渲染失败：{str(e)}")

    st.divider()
    row1 = st.columns(2)
    with row1[0]:
        try:
            _plot_low_score_distribution(low_score_dist)
        except Exception as e:
            st.error(f"低分分布图渲染失败：{str(e)}")
    with row1[1]:
        try:
            _plot_agent_load(agent_load)
        except Exception as e:
            st.error(f"坐席负载图渲染失败：{str(e)}")

    row2 = st.columns(1)
    with row2[0]:
        try:
            _plot_channel_trend(channel_trend)
        except Exception as e:
            st.error(f"渠道趋势图渲染失败：{str(e)}")

    st.divider()
    st.subheader("📋 待复核明细（低分 / 响应超时 / 未解决）")
    if pending_review.empty:
        st.success("🎉 当前筛选条件下没有需要复核的记录。")
    else:
        st.dataframe(pending_review, use_container_width=True, hide_index=True)
        st.caption(f"共 {len(pending_review)} 条待复核记录。")

    st.divider()
    _render_suggestions(suggestions)

    st.divider()
    st.subheader("📥 导出报告")
    exp_cols = st.columns(3)
    try:
        clean_bytes = export_clean_data(filtered)
        exp_cols[0].download_button(
            "下载清洗后数据 (Excel)",
            data=clean_bytes,
            file_name=generate_filename("cleaned_data"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        exp_cols[0].error(f"导出清洗数据失败：{str(e)}")

    try:
        report_bytes = export_full_report(filtered, metrics, agent_load,
                                          channel_trend, pending_review, suggestions)
        exp_cols[1].download_button(
            "下载完整分析报告 (Excel)",
            data=report_bytes,
            file_name=generate_filename("quality_report"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        exp_cols[1].error(f"导出完整报告失败：{str(e)}")

    try:
        csv_bytes = filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        exp_cols[2].download_button(
            "下载筛选结果 (CSV)",
            data=csv_bytes,
            file_name=generate_filename("filtered_data", suffix="csv"),
            mime="text/csv"
        )
    except Exception as e:
        exp_cols[2].error(f"导出 CSV 失败：{str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"页面出现未处理异常：{str(e)}")
        with st.expander("🔍 完整异常堆栈"):
            st.code(traceback.format_exc())
