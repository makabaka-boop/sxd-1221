import pandas as pd
import re
from typing import Dict, List, Tuple

REQUIRED_FIELDS = [
    "record_date", "channel_name", "agent_name",
    "response_seconds", "score", "issue_type",
    "solved_flag", "note"
]

FIELD_LABELS = {
    "record_date": "记录日期",
    "channel_name": "渠道名称",
    "agent_name": "坐席姓名",
    "response_seconds": "响应时长(秒)",
    "score": "质检分数",
    "issue_type": "问题类型",
    "solved_flag": "是否解决",
    "note": "备注"
}

RESPONSE_SECONDS_UPPER = 3600
SCORE_MIN = 0
SCORE_MAX = 100


def _parse_numeric_raw(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    s = re.sub(r"[,%\s]", "", s)
    s = s.replace("分", "").replace("秒", "").replace("s", "").replace("S", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def validate_raw_data(mapped_df: pd.DataFrame) -> dict:
    total_rows = len(mapped_df)
    issues: List[dict] = []

    subset = [c for c in ["record_date", "channel_name", "agent_name",
                          "response_seconds", "score", "issue_type"]
              if c in mapped_df.columns]
    if subset:
        dup_mask = mapped_df.duplicated(subset=subset, keep=False)
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            dup_groups = mapped_df[dup_mask].assign(
                _行号=lambda d: d.index + 2
            ).sort_values(by=subset).groupby(subset)["_行号"].apply(
                lambda x: "、".join(map(str, x))
            ).reset_index(name="重复行号")
            dup_groups.columns = [FIELD_LABELS.get(c, c) if c != "重复行号" else c
                                  for c in dup_groups.columns]
            issues.append({
                "type": "duplicate",
                "level": "warning",
                "title": f"发现 {dup_count} 条重复记录（共 {len(dup_groups)} 组）",
                "detail_df": dup_groups.head(20),
                "tip": "清洗时会自动保留第一条记录，其余重复行将被移除。请检查以上行号是否确为重复。"
            })

    if "response_seconds" in mapped_df.columns:
        col = mapped_df["response_seconds"]
        non_null = col.notna()
        parsed = col.apply(_parse_numeric_raw)
        invalid_mask = non_null & parsed.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            bad_df = mapped_df[invalid_mask].assign(
                _行号=lambda d: d.index + 2,
                _原始值=lambda d: d["response_seconds"].astype(str)
            )[["_行号", "_原始值", "channel_name", "agent_name", "issue_type"]].rename(columns={
                "_行号": "行号", "_原始值": "原始响应时长",
                "channel_name": "渠道", "agent_name": "坐席", "issue_type": "问题类型"
            }).head(20)
            issues.append({
                "type": "response_format",
                "level": "error",
                "title": f"响应时长列有 {invalid_count} 条记录无法解析为数字",
                "detail_df": bad_df,
                "tip": "支持格式：纯数字（如 45）、带单位（如 45秒、45s）。含字母或特殊符号的将被置空。"
            })

        anomaly_mask = non_null & parsed.notna() & ((parsed < 0) | (parsed > RESPONSE_SECONDS_UPPER))
        anomaly_count = int(anomaly_mask.sum())
        if anomaly_count > 0:
            bad_df = mapped_df[anomaly_mask].assign(
                _行号=lambda d: d.index + 2,
                _原始值=lambda d: d["response_seconds"].astype(str)
            )[["_行号", "_原始值", "channel_name", "agent_name", "issue_type"]].rename(columns={
                "_行号": "行号", "_原始值": "原始响应时长",
                "channel_name": "渠道", "agent_name": "坐席", "issue_type": "问题类型"
            }).head(20)
            issues.append({
                "type": "response_anomaly",
                "level": "warning",
                "title": f"响应时长列有 {anomaly_count} 条记录超出合理范围（0~{RESPONSE_SECONDS_UPPER}秒）",
                "detail_df": bad_df,
                "tip": "负值或超过 1 小时的响应时长可能为录入错误，建议核实。"
            })

    if "score" in mapped_df.columns:
        col = mapped_df["score"]
        non_null = col.notna()
        parsed = col.apply(_parse_numeric_raw)
        invalid_mask = non_null & parsed.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            bad_df = mapped_df[invalid_mask].assign(
                _行号=lambda d: d.index + 2,
                _原始值=lambda d: d["score"].astype(str)
            )[["_行号", "_原始值", "channel_name", "agent_name", "issue_type"]].rename(columns={
                "_行号": "行号", "_原始值": "原始分数",
                "channel_name": "渠道", "agent_name": "坐席", "issue_type": "问题类型"
            }).head(20)
            issues.append({
                "type": "score_format",
                "level": "error",
                "title": f"质检分数列有 {invalid_count} 条记录无法解析为数字",
                "detail_df": bad_df,
                "tip": "支持格式：纯数字（如 85）、带单位（如 85分）。非数字内容（如「待评分」）将被置空。"
            })

        out_mask = non_null & parsed.notna() & ((parsed < SCORE_MIN) | (parsed > SCORE_MAX))
        out_count = int(out_mask.sum())
        if out_count > 0:
            bad_df = mapped_df[out_mask].assign(
                _行号=lambda d: d.index + 2,
                _原始值=lambda d: d["score"].astype(str)
            )[["_行号", "_原始值", "channel_name", "agent_name", "issue_type"]].rename(columns={
                "_行号": "行号", "_原始值": "原始分数",
                "channel_name": "渠道", "agent_name": "坐席", "issue_type": "问题类型"
            }).head(20)
            issues.append({
                "type": "score_range",
                "level": "warning",
                "title": f"质检分数列有 {out_count} 条记录超出合理范围（{SCORE_MIN}~{SCORE_MAX}分）",
                "detail_df": bad_df,
                "tip": "分数应为 0-100 之间的数值，超出范围的值建议核实。"
            })

    if "record_date" in mapped_df.columns:
        col = mapped_df["record_date"]
        non_null = col.notna()
        parsed = pd.to_datetime(col, errors="coerce")
        invalid_mask = non_null & parsed.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            bad_df = mapped_df[invalid_mask].assign(
                _行号=lambda d: d.index + 2,
                _原始值=lambda d: d["record_date"].astype(str)
            )[["_行号", "_原始值", "channel_name", "agent_name"]].rename(columns={
                "_行号": "行号", "_原始值": "原始日期",
                "channel_name": "渠道", "agent_name": "坐席"
            }).head(20)
            issues.append({
                "type": "date_format",
                "level": "error",
                "title": f"记录日期列有 {invalid_count} 条记录无法解析为日期",
                "detail_df": bad_df,
                "tip": "推荐格式：YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DD。"
            })

    if "solved_flag" in mapped_df.columns:
        col = mapped_df["solved_flag"]
        non_null = col.notna()
        allowed_true = {1, "1", "是", "Y", "y", "YES", "Yes", "yes",
                        "TRUE", "True", "true", "已解决", "解决"}
        allowed_false = {0, "0", "否", "N", "n", "NO", "No", "no",
                         "FALSE", "False", "false", "未解决", "没解决"}

        def _check(v):
            if v in allowed_true or v in allowed_false:
                return True
            if isinstance(v, bool):
                return True
            s = str(v).strip()
            return s in allowed_true or s in allowed_false

        invalid_mask = non_null & ~col.apply(_check)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            bad_df = mapped_df[invalid_mask].assign(
                _行号=lambda d: d.index + 2,
                _原始值=lambda d: d["solved_flag"].astype(str)
            )[["_行号", "_原始值", "channel_name", "agent_name"]].rename(columns={
                "_行号": "行号", "_原始值": "原始值",
                "channel_name": "渠道", "agent_name": "坐席"
            }).head(20)
            issues.append({
                "type": "solved_format",
                "level": "warning",
                "title": f"是否解决列有 {invalid_count} 条记录格式不规范",
                "detail_df": bad_df,
                "tip": "推荐值：是/否、1/0、True/False、已解决/未解决。"
            })

    return {
        "total_rows": total_rows,
        "issue_count": len(issues),
        "issues": issues
    }


def detect_missing_fields(df: pd.DataFrame) -> list:
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            missing.append(field)
        else:
            all_null = df[field].isna().all()
            if all_null and field != "note":
                missing.append(field)
    return missing


def check_duplicates(df: pd.DataFrame) -> dict:
    subset = [c for c in ["record_date", "channel_name", "agent_name", "response_seconds", "score", "issue_type"]
              if c in df.columns]
    dup_mask = df.duplicated(subset=subset, keep=False)
    dup_count = int(dup_mask.sum())
    dup_df = df[dup_mask].sort_values(by=subset).head(10) if dup_count > 0 else pd.DataFrame()
    return {
        "duplicate_count": dup_count,
        "duplicate_rows": dup_df,
        "unique_rows": int(len(df) - dup_count + (dup_count // 2 if dup_count > 0 else 0))
    }


def check_response_anomalies(df: pd.DataFrame) -> dict:
    if "response_seconds" not in df.columns:
        return {"anomaly_count": 0, "anomaly_rows": pd.DataFrame()}

    col = pd.to_numeric(df["response_seconds"], errors="coerce")
    anomaly_mask = (col < 0) | (col > RESPONSE_SECONDS_UPPER)
    anomaly_mask = anomaly_mask.fillna(False)
    anomaly_count = int(anomaly_mask.sum())
    anomaly_rows = df[anomaly_mask][["record_date", "channel_name", "agent_name", "response_seconds", "issue_type"]].head(10) if anomaly_count > 0 else pd.DataFrame()
    return {
        "anomaly_count": anomaly_count,
        "anomaly_rows": anomaly_rows,
        "max_threshold": RESPONSE_SECONDS_UPPER
    }


def check_score_format(df: pd.DataFrame) -> dict:
    if "score" not in df.columns:
        return {"invalid_count": 0, "out_of_range": 0, "invalid_rows": pd.DataFrame()}

    raw = df["score"]
    non_null_mask = raw.notna()
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid_mask = non_null_mask & numeric.isna()
    out_of_range_mask = (numeric < SCORE_MIN) | (numeric > SCORE_MAX)
    out_of_range_mask = out_of_range_mask.fillna(False)

    invalid_count = int(invalid_mask.sum())
    out_of_range_count = int(out_of_range_mask.sum())

    bad_mask = invalid_mask | out_of_range_mask
    bad_rows = df[bad_mask][["record_date", "channel_name", "agent_name", "score", "issue_type"]].head(10) if bad_mask.any() else pd.DataFrame()

    return {
        "invalid_count": invalid_count,
        "out_of_range": out_of_range_count,
        "invalid_rows": bad_rows,
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX
    }


def validate_numeric_columns(df: pd.DataFrame) -> dict:
    errors = {}
    for field in ["response_seconds", "score"]:
        if field in df.columns:
            col = df[field]
            non_null = col.notna()
            converted = pd.to_numeric(col, errors="coerce")
            failed_mask = non_null & converted.isna()
            failed_count = int(failed_mask.sum())
            if failed_count > 0:
                samples = col[failed_mask].head(5).astype(str).tolist()
                errors[field] = {
                    "failed_count": failed_count,
                    "samples": samples,
                    "hint": f"字段「{field}」共 {failed_count} 条记录无法解析为数字，请检查：{', '.join(samples)}"
                }
    return errors


def run_all_validations(df: pd.DataFrame) -> dict:
    return {
        "missing_fields": detect_missing_fields(df),
        "duplicates": check_duplicates(df),
        "response_anomalies": check_response_anomalies(df),
        "score_format": check_score_format(df),
        "numeric_errors": validate_numeric_columns(df)
    }
