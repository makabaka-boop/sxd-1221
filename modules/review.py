import pandas as pd
import numpy as np
from datetime import datetime
import uuid


REVIEW_STATUS_OPTIONS = ["待复核", "已确认", "已忽略"]

DEFAULT_LOW_SCORE_THRESHOLD = 60
DEFAULT_RESPONSE_THRESHOLD = 300


def _generate_task_id() -> str:
    return f"RT{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8].upper()}"


def _determine_priority(review_reason: str, score: float, response_seconds: float, solved: bool) -> str:
    reasons = [r for r in review_reason.split(";") if r]
    score_val = float(score) if pd.notna(score) else 100
    resp_val = float(response_seconds) if pd.notna(response_seconds) else 0

    high_conditions = [
        len(reasons) >= 2,
        score_val < 40,
        resp_val > 600,
        solved == False and score_val < 50
    ]

    if any(high_conditions):
        return "高"

    mid_conditions = [
        score_val < 60,
        resp_val > 300,
        solved == False
    ]

    if any(mid_conditions):
        return "中"

    return "低"


def _generate_suggestion(review_reason: str, score: float, response_seconds: float,
                         solved: bool, issue_type: str, agent_name: str) -> str:
    suggestions = []
    score_val = float(score) if pd.notna(score) else 100
    resp_val = float(response_seconds) if pd.notna(response_seconds) else 0

    if "低分" in review_reason:
        if score_val < 40:
            suggestions.append(f"质检分数仅 {score_val:.0f} 分，建议立即复核会话录音/文本，评估服务态度与专业性")
        elif score_val < 60:
            suggestions.append(f"质检分数 {score_val:.0f} 分，低于及格线，建议复核并了解客户不满原因")

    if "响应超时" in review_reason:
        if resp_val > 600:
            suggestions.append(f"响应时长达 {resp_val:.0f} 秒，严重超时，建议核查系统与坐席排班问题")
        else:
            suggestions.append(f"响应时长 {resp_val:.0f} 秒，超过标准阈值，建议优化知识库检索效率")

    if "未解决" in review_reason:
        suggestions.append("客户问题未解决，建议跟进回访，确认是否需要二次处理或升级")

    if len(suggestions) == 0:
        suggestions.append("建议复核完整会话内容，确认是否存在服务质量问题")

    suggestions.append(f"问题类型为「{issue_type}」，可重点关注坐席「{agent_name}」的相关话术与处理流程")

    return "；".join(suggestions)


def generate_review_tasks(df: pd.DataFrame,
                          low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
                          response_threshold: int = DEFAULT_RESPONSE_THRESHOLD) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "task_id", "record_date", "channel_name", "agent_name",
            "issue_type", "score", "response_seconds", "solved_flag",
            "review_reason", "priority", "status", "review_note",
            "suggestion", "original_index"
        ])

    numeric_resp = pd.to_numeric(df["response_seconds"], errors="coerce")
    numeric_score = pd.to_numeric(df["score"], errors="coerce")

    mask = (
        (numeric_score < low_score_threshold)
        | (numeric_resp > response_threshold)
        | (df["solved_flag"] == False)
    )

    if not mask.any():
        return pd.DataFrame(columns=[
            "task_id", "record_date", "channel_name", "agent_name",
            "issue_type", "score", "response_seconds", "solved_flag",
            "review_reason", "priority", "status", "review_note",
            "suggestion", "original_index"
        ])

    result = df[mask].copy()
    result["original_index"] = result.index

    result["review_reason"] = ""
    result.loc[(numeric_score < low_score_threshold)[mask], "review_reason"] += "低分;"
    result.loc[(numeric_resp > response_threshold)[mask], "review_reason"] += "响应超时;"
    result.loc[(df["solved_flag"] == False)[mask], "review_reason"] += "未解决;"

    result["task_id"] = [_generate_task_id() for _ in range(len(result))]
    result["priority"] = result.apply(
        lambda row: _determine_priority(
            row["review_reason"], row["score"], row["response_seconds"], row["solved_flag"]
        ), axis=1
    )
    result["status"] = "待复核"
    result["review_note"] = ""
    result["suggestion"] = result.apply(
        lambda row: _generate_suggestion(
            row["review_reason"], row["score"], row["response_seconds"],
            row["solved_flag"], row["issue_type"], row["agent_name"]
        ), axis=1
    )

    output_columns = [
        "task_id", "record_date", "channel_name", "agent_name",
        "issue_type", "score", "response_seconds", "solved_flag",
        "review_reason", "priority", "status", "review_note",
        "suggestion", "original_index"
    ]

    result = result[output_columns].sort_values(
        by=["priority", "record_date"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return result


def filter_review_tasks(tasks_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = tasks_df.copy()

    if "date_range" in filters and filters["date_range"] and len(filters["date_range"]) == 2:
        start, end = filters["date_range"]
        if start:
            result = result[result["record_date"] >= pd.Timestamp(start)]
        if end:
            result = result[result["record_date"] <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if "channels" in filters and filters["channels"]:
        result = result[result["channel_name"].isin(filters["channels"])]

    if "agents" in filters and filters["agents"]:
        result = result[result["agent_name"].isin(filters["agents"])]

    if "issue_types" in filters and filters["issue_types"]:
        result = result[result["issue_type"].isin(filters["issue_types"])]

    if "priorities" in filters and filters["priorities"]:
        result = result[result["priority"].isin(filters["priorities"])]

    if "statuses" in filters and filters["statuses"]:
        result = result[result["status"].isin(filters["statuses"])]

    return result


def compute_review_statistics(tasks_df: pd.DataFrame) -> dict:
    if tasks_df.empty:
        return {
            "total_tasks": 0,
            "pending_count": 0,
            "confirmed_count": 0,
            "ignored_count": 0,
            "high_priority_count": 0,
            "mid_priority_count": 0,
            "low_priority_count": 0,
            "low_score_count": 0,
            "timeout_count": 0,
            "unsolved_count": 0,
            "completion_rate": 0.0,
            "by_agent": pd.DataFrame(),
            "by_channel": pd.DataFrame(),
            "by_issue_type": pd.DataFrame(),
            "by_priority": pd.DataFrame(),
            "by_status": pd.DataFrame()
        }

    total = len(tasks_df)
    pending = (tasks_df["status"] == "待复核").sum()
    confirmed = (tasks_df["status"] == "已确认").sum()
    ignored = (tasks_df["status"] == "已忽略").sum()

    high_priority = (tasks_df["priority"] == "高").sum()
    mid_priority = (tasks_df["priority"] == "中").sum()
    low_priority = (tasks_df["priority"] == "低").sum()

    low_score = tasks_df["review_reason"].str.contains("低分", na=False).sum()
    timeout = tasks_df["review_reason"].str.contains("响应超时", na=False).sum()
    unsolved = tasks_df["review_reason"].str.contains("未解决", na=False).sum()

    completion_rate = ((confirmed + ignored) / total * 100) if total > 0 else 0.0

    by_agent = tasks_df.groupby(["agent_name", "status"]).size().reset_index(name="count")
    by_agent = by_agent.pivot(index="agent_name", columns="status", values="count").fillna(0).reset_index()
    by_agent["总计"] = by_agent.sum(axis=1, numeric_only=True)

    by_channel = tasks_df.groupby(["channel_name", "status"]).size().reset_index(name="count")
    by_channel = by_channel.pivot(index="channel_name", columns="status", values="count").fillna(0).reset_index()
    by_channel["总计"] = by_channel.sum(axis=1, numeric_only=True)

    by_issue_type = tasks_df.groupby(["issue_type", "status"]).size().reset_index(name="count")
    by_issue_type = by_issue_type.pivot(index="issue_type", columns="status", values="count").fillna(0).reset_index()
    by_issue_type["总计"] = by_issue_type.sum(axis=1, numeric_only=True)

    by_priority = tasks_df.groupby("priority").size().reset_index(name="count")
    priority_order = pd.Categorical(by_priority["priority"], categories=["高", "中", "低"], ordered=True)
    by_priority["priority"] = priority_order
    by_priority = by_priority.sort_values("priority").reset_index(drop=True)

    by_status = tasks_df.groupby("status").size().reset_index(name="count")

    return {
        "total_tasks": int(total),
        "pending_count": int(pending),
        "confirmed_count": int(confirmed),
        "ignored_count": int(ignored),
        "high_priority_count": int(high_priority),
        "mid_priority_count": int(mid_priority),
        "low_priority_count": int(low_priority),
        "low_score_count": int(low_score),
        "timeout_count": int(timeout),
        "unsolved_count": int(unsolved),
        "completion_rate": round(float(completion_rate), 2),
        "by_agent": by_agent,
        "by_channel": by_channel,
        "by_issue_type": by_issue_type,
        "by_priority": by_priority,
        "by_status": by_status
    }


def update_task_status(tasks_df: pd.DataFrame, task_id: str, new_status: str, note: str = "") -> pd.DataFrame:
    if tasks_df.empty:
        return tasks_df
    if new_status not in REVIEW_STATUS_OPTIONS:
        return tasks_df

    mask = tasks_df["task_id"] == task_id
    if not mask.any():
        return tasks_df

    result = tasks_df.copy()
    result.loc[mask, "status"] = new_status
    if note:
        current_note = result.loc[mask, "review_note"].iloc[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_note = f"[{timestamp}] {note}"
        if current_note:
            result.loc[mask, "review_note"] = f"{current_note}\n{new_note}"
        else:
            result.loc[mask, "review_note"] = new_note

    return result


def batch_update_status(tasks_df: pd.DataFrame, task_ids: list, new_status: str, note: str = "") -> pd.DataFrame:
    result = tasks_df.copy()
    for task_id in task_ids:
        result = update_task_status(result, task_id, new_status, note)
    return result
