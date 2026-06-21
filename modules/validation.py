import pandas as pd

REQUIRED_FIELDS = [
    "record_date", "channel_name", "agent_name",
    "response_seconds", "score", "issue_type",
    "solved_flag", "note"
]

RESPONSE_SECONDS_UPPER = 3600
SCORE_MIN = 0
SCORE_MAX = 100


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
