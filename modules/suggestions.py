import pandas as pd
import numpy as np
from datetime import timedelta


def generate_quality_suggestions(df: pd.DataFrame,
                                low_score_threshold: float = 60,
                                response_threshold: int = 300,
                                days: int = 30) -> list:
    suggestions = []

    if df.empty:
        return suggestions

    df = df.copy()
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    max_date = df["record_date"].max()
    if pd.isna(max_date):
        return suggestions
    cutoff = max_date - timedelta(days=days)
    recent = df[df["record_date"] >= cutoff].copy()

    if recent.empty:
        return suggestions

    numeric_resp = pd.to_numeric(recent["response_seconds"], errors="coerce")
    numeric_score = pd.to_numeric(recent["score"], errors="coerce")
    recent = recent.assign(_resp=numeric_resp, _score=numeric_score)

    total = len(recent)
    if total == 0:
        return suggestions

    overall_low_score_rate = (recent["_score"] < low_score_threshold).mean()
    overall_timeout_rate = (recent["_resp"] > response_threshold).mean()
    overall_unsolved_rate = (recent["solved_flag"] == False).mean()

    def _format_pct(v):
        return f"{v * 100:.2f}%"

    _suggest_channels(recent, low_score_threshold, response_threshold,
                      overall_low_score_rate, overall_timeout_rate,
                      overall_unsolved_rate, suggestions, _format_pct)

    _suggest_issue_types(recent, low_score_threshold, response_threshold,
                         overall_low_score_rate, overall_timeout_rate,
                         overall_unsolved_rate, suggestions, _format_pct)

    _suggest_agents(recent, low_score_threshold, response_threshold,
                    overall_low_score_rate, overall_timeout_rate,
                    overall_unsolved_rate, suggestions, _format_pct)

    return suggestions


def _suggest_channels(recent, low_score_threshold, response_threshold,
                      overall_low_rate, overall_timeout_rate,
                      overall_unsolved_rate, suggestions, fmt_pct):
    grouped = recent.groupby("channel_name")
    agg = grouped.agg(
        count=("channel_name", "size"),
        low_score=("_score", lambda s: (s < low_score_threshold).mean()),
        timeout=("_resp", lambda s: (s > response_threshold).mean()),
        unsolved=("solved_flag", lambda s: (s == False).mean())
    ).reset_index()

    for _, row in agg.iterrows():
        name = row["channel_name"]
        if pd.isna(name) or row["count"] < 5:
            continue
        issues = []
        if row["low_score"] > max(overall_low_rate * 1.3, 0.15):
            issues.append(f"低分率 {fmt_pct(row['low_score'])}（整体 {fmt_pct(overall_low_rate)}）")
        if row["timeout"] > max(overall_timeout_rate * 1.3, 0.15):
            issues.append(f"超时率 {fmt_pct(row['timeout'])}（整体 {fmt_pct(overall_timeout_rate)}）")
        if row["unsolved"] > max(overall_unsolved_rate * 1.3, 0.15):
            issues.append(f"未解决率 {fmt_pct(row['unsolved'])}（整体 {fmt_pct(overall_unsolved_rate)}）")
        if issues:
            suggestions.append({
                "category": "渠道",
                "target": name,
                "level": "高",
                "reasons": issues,
                "action": f"建议重点复核渠道「{name}」的服务流程，抽样不低于 20% 的待复核会话"
            })


def _suggest_issue_types(recent, low_score_threshold, response_threshold,
                         overall_low_rate, overall_timeout_rate,
                         overall_unsolved_rate, suggestions, fmt_pct):
    grouped = recent.groupby("issue_type")
    agg = grouped.agg(
        count=("issue_type", "size"),
        low_score=("_score", lambda s: (s < low_score_threshold).mean()),
        timeout=("_resp", lambda s: (s > response_threshold).mean()),
        unsolved=("solved_flag", lambda s: (s == False).mean())
    ).reset_index()

    for _, row in agg.iterrows():
        name = row["issue_type"]
        if pd.isna(name) or row["count"] < 5:
            continue
        issues = []
        if row["low_score"] > max(overall_low_rate * 1.3, 0.15):
            issues.append(f"低分率 {fmt_pct(row['low_score'])}")
        if row["timeout"] > max(overall_timeout_rate * 1.3, 0.15):
            issues.append(f"超时率 {fmt_pct(row['timeout'])}")
        if row["unsolved"] > max(overall_unsolved_rate * 1.3, 0.15):
            issues.append(f"未解决率 {fmt_pct(row['unsolved'])}")
        if issues:
            suggestions.append({
                "category": "问题类型",
                "target": name,
                "level": "中",
                "reasons": issues,
                "action": f"建议针对「{name}」问题完善知识库与话术模板，并组织专项培训"
            })


def _suggest_agents(recent, low_score_threshold, response_threshold,
                    overall_low_rate, overall_timeout_rate,
                    overall_unsolved_rate, suggestions, fmt_pct):
    grouped = recent.groupby("agent_name")
    agg = grouped.agg(
        count=("agent_name", "size"),
        low_score=("_score", lambda s: (s < low_score_threshold).mean()),
        timeout=("_resp", lambda s: (s > response_threshold).mean()),
        unsolved=("solved_flag", lambda s: (s == False).mean()),
        avg_score=("_score", "mean")
    ).reset_index()

    for _, row in agg.iterrows():
        name = row["agent_name"]
        if pd.isna(name) or row["count"] < 3:
            continue
        issues = []
        if row["low_score"] > max(overall_low_rate * 1.5, 0.2):
            issues.append(f"低分率 {fmt_pct(row['low_score'])}，平均分 {row['avg_score']:.1f}")
        if row["timeout"] > max(overall_timeout_rate * 1.5, 0.2):
            issues.append(f"超时率 {fmt_pct(row['timeout'])}")
        if row["unsolved"] > max(overall_unsolved_rate * 1.5, 0.2):
            issues.append(f"未解决率 {fmt_pct(row['unsolved'])}")
        if issues:
            suggestions.append({
                "category": "坐席",
                "target": name,
                "level": "高" if len(issues) >= 2 else "中",
                "reasons": issues,
                "action": f"建议安排坐席「{name}」进行一对一辅导，并复核其近 7 天全部低分会话"
            })
