import pandas as pd
import numpy as np


def filter_dataframe(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = df.copy()

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

    if "score_range" in filters and filters["score_range"] is not None:
        min_s, max_s = filters["score_range"]
        result = result[(result["score"] >= min_s) & (result["score"] <= max_s)]

    if "solved_status" in filters and filters["solved_status"] != "全部":
        flag = filters["solved_status"] == "已解决"
        result = result[result["solved_flag"] == flag]

    return result


def compute_overall_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_records": 0,
            "avg_response_seconds": 0,
            "first_resolution_rate": 0,
            "avg_score": 0,
            "low_score_count": 0,
            "unsolved_count": 0
        }
    total = len(df)
    avg_resp = pd.to_numeric(df["response_seconds"], errors="coerce").mean()
    avg_score = pd.to_numeric(df["score"], errors="coerce").mean()
    solved = df["solved_flag"].dropna()
    frr = solved.mean() if len(solved) > 0 else 0.0
    low_score = (pd.to_numeric(df["score"], errors="coerce") < 60).sum()
    unsolved = (df["solved_flag"] == False).sum()

    return {
        "total_records": total,
        "avg_response_seconds": round(float(avg_resp), 1) if not pd.isna(avg_resp) else 0,
        "first_resolution_rate": round(float(frr) * 100, 2),
        "avg_score": round(float(avg_score), 2) if not pd.isna(avg_score) else 0,
        "low_score_count": int(low_score),
        "unsolved_count": int(unsolved)
    }


def groupby_low_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    numeric_score = pd.to_numeric(df["score"], errors="coerce")
    low = df.assign(_score_num=numeric_score)[numeric_score < 60]
    if low.empty:
        return pd.DataFrame(columns=["issue_type", "count"])
    grouped = low.groupby("issue_type").size().reset_index(name="count")
    return grouped.sort_values("count", ascending=False)


def groupby_agent_load(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["agent_name", "record_count", "avg_response_seconds", "avg_score"])
    numeric_resp = pd.to_numeric(df["response_seconds"], errors="coerce")
    numeric_score = pd.to_numeric(df["score"], errors="coerce")
    tmp = df.assign(_resp=numeric_resp, _score=numeric_score)
    grouped = tmp.groupby("agent_name").agg(
        record_count=("agent_name", "size"),
        avg_response_seconds=("_resp", "mean"),
        avg_score=("_score", "mean")
    ).reset_index()
    grouped["avg_response_seconds"] = grouped["avg_response_seconds"].round(1)
    grouped["avg_score"] = grouped["avg_score"].round(2)
    return grouped.sort_values("record_count", ascending=False)


def groupby_channel_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["record_date", "channel_name", "count"])
    tmp = df.copy()
    tmp["record_day"] = pd.to_datetime(tmp["record_date"]).dt.date
    grouped = tmp.groupby(["record_day", "channel_name"]).size().reset_index(name="count")
    grouped = grouped.rename(columns={"record_day": "record_date"})
    return grouped.sort_values(["record_date", "channel_name"])


def get_pending_review_details(df: pd.DataFrame, response_threshold: int = 300) -> pd.DataFrame:
    numeric_resp = pd.to_numeric(df["response_seconds"], errors="coerce")
    numeric_score = pd.to_numeric(df["score"], errors="coerce")
    mask = (
        (numeric_score < 60)
        | (numeric_resp > response_threshold)
        | (df["solved_flag"] == False)
    )
    cols = ["record_date", "channel_name", "agent_name", "response_seconds",
            "score", "issue_type", "solved_flag", "note"]
    result = df[mask][cols].copy() if mask.any() else pd.DataFrame(columns=cols)
    if not result.empty:
        result["_review_reason"] = ""
        result.loc[numeric_score[mask] < 60, "_review_reason"] += "低分;"
        result.loc[numeric_resp[mask] > response_threshold, "_review_reason"] += "响应超时;"
        result.loc[(df["solved_flag"] == False)[mask], "_review_reason"] += "未解决;"
        result = result.rename(columns={"_review_reason": "复核原因"})
    return result
