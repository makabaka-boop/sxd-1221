import pandas as pd
import io
from datetime import datetime
from typing import Optional


def export_clean_data(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="清洗后数据", index=False)
    return output.getvalue()


def export_full_report(df: pd.DataFrame, metrics: dict,
                       agent_load: pd.DataFrame,
                       channel_trend: pd.DataFrame,
                       pending_review: pd.DataFrame,
                       suggestions: list,
                       review_tasks: Optional[pd.DataFrame] = None,
                       review_stats: Optional[dict] = None) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="原始数据", index=False)

        metrics_df = pd.DataFrame([
            {"指标": "总记录数", "数值": metrics.get("total_records", 0)},
            {"指标": "平均响应时长(秒)", "数值": metrics.get("avg_response_seconds", 0)},
            {"指标": "一次解决率(%)", "数值": metrics.get("first_resolution_rate", 0)},
            {"指标": "平均质检分数", "数值": metrics.get("avg_score", 0)},
            {"指标": "低分记录数", "数值": metrics.get("low_score_count", 0)},
            {"指标": "未解决记录数", "数值": metrics.get("unsolved_count", 0)},
        ])
        metrics_df.to_excel(writer, sheet_name="核心指标", index=False)

        agent_load.to_excel(writer, sheet_name="坐席负载", index=False)
        channel_trend.to_excel(writer, sheet_name="渠道趋势", index=False)
        pending_review.to_excel(writer, sheet_name="待复核明细", index=False)

        if suggestions:
            sug_rows = []
            for i, s in enumerate(suggestions, 1):
                sug_rows.append({
                    "序号": i,
                    "分类": s.get("category", ""),
                    "对象": s.get("target", ""),
                    "优先级": s.get("level", ""),
                    "问题": "；".join(s.get("reasons", [])),
                    "建议动作": s.get("action", "")
                })
            pd.DataFrame(sug_rows).to_excel(writer, sheet_name="质检建议", index=False)

        if review_tasks is not None and not review_tasks.empty:
            export_columns = [
                "task_id", "record_date", "channel_name", "agent_name",
                "issue_type", "score", "response_seconds", "solved_flag",
                "review_reason", "priority", "status", "review_note",
                "suggestion"
            ]
            available_cols = [c for c in export_columns if c in review_tasks.columns]
            review_export = review_tasks[available_cols].copy()
            review_export.columns = [
                "任务ID", "记录日期", "渠道名称", "坐席姓名",
                "问题类型", "质检分数", "响应时长(秒)", "是否解决",
                "复核原因", "优先级", "复核状态", "复核备注",
                "处理建议"
            ]
            review_export.to_excel(writer, sheet_name="复核任务明细", index=False)

        if review_stats is not None:
            summary_rows = [
                {"项目": "复核任务总数", "数值": review_stats.get("total_tasks", 0)},
                {"项目": "待复核", "数值": review_stats.get("pending_count", 0)},
                {"项目": "已确认", "数值": review_stats.get("confirmed_count", 0)},
                {"项目": "已忽略", "数值": review_stats.get("ignored_count", 0)},
                {"项目": "高优先级", "数值": review_stats.get("high_priority_count", 0)},
                {"项目": "中优先级", "数值": review_stats.get("mid_priority_count", 0)},
                {"项目": "低优先级", "数值": review_stats.get("low_priority_count", 0)},
                {"项目": "低分相关", "数值": review_stats.get("low_score_count", 0)},
                {"项目": "超时相关", "数值": review_stats.get("timeout_count", 0)},
                {"项目": "未解决相关", "数值": review_stats.get("unsolved_count", 0)},
                {"项目": "复核完成率(%)", "数值": review_stats.get("completion_rate", 0.0)},
            ]
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="复核状态汇总", index=False)

            if "by_agent" in review_stats and not review_stats["by_agent"].empty:
                by_agent_df = review_stats["by_agent"].copy()
                by_agent_df.columns = ["坐席姓名"] + [str(c) for c in by_agent_df.columns if c != "agent_name"]
                by_agent_df.to_excel(writer, sheet_name="复核-按坐席统计", index=False)

            if "by_channel" in review_stats and not review_stats["by_channel"].empty:
                by_channel_df = review_stats["by_channel"].copy()
                by_channel_df.columns = ["渠道名称"] + [str(c) for c in by_channel_df.columns if c != "channel_name"]
                by_channel_df.to_excel(writer, sheet_name="复核-按渠道统计", index=False)

            if "by_issue_type" in review_stats and not review_stats["by_issue_type"].empty:
                by_issue_df = review_stats["by_issue_type"].copy()
                by_issue_df.columns = ["问题类型"] + [str(c) for c in by_issue_df.columns if c != "issue_type"]
                by_issue_df.to_excel(writer, sheet_name="复核-按问题统计", index=False)

    return output.getvalue()


def generate_filename(prefix: str, suffix: str = "xlsx") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{suffix}"
