import pandas as pd
import io
from datetime import datetime


def export_clean_data(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="清洗后数据", index=False)
    return output.getvalue()


def export_full_report(df: pd.DataFrame, metrics: dict,
                       agent_load: pd.DataFrame,
                       channel_trend: pd.DataFrame,
                       pending_review: pd.DataFrame,
                       suggestions: list) -> bytes:
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

    return output.getvalue()


def generate_filename(prefix: str, suffix: str = "xlsx") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{suffix}"
