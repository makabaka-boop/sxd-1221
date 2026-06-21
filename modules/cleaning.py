import pandas as pd
import re

REQUIRED_FIELDS = {
    "record_date": "记录日期",
    "channel_name": "渠道名称",
    "agent_name": "坐席姓名",
    "response_seconds": "响应时长(秒)",
    "score": "质检分数",
    "issue_type": "问题类型",
    "solved_flag": "是否解决",
    "note": "备注"
}

NUMERIC_FIELDS = ["response_seconds", "score"]


def clean_record_date(series: pd.Series) -> pd.Series:
    try:
        cleaned = pd.to_datetime(series, errors="coerce")
        return cleaned
    except Exception:
        return pd.Series([pd.NaT] * len(series))


def _parse_numeric(val):
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


def clean_numeric(series: pd.Series, field_name: str) -> tuple:
    raw_count = series.notna().sum()
    cleaned = series.apply(_parse_numeric)
    success_count = cleaned.notna().sum()
    failed_mask = series.notna() & cleaned.isna()
    failed_examples = series[failed_mask].head(5).tolist()
    return cleaned, {
        "field": field_name,
        "raw_nonnull": int(raw_count),
        "parsed_nonnull": int(success_count),
        "failed_count": int(raw_count - success_count),
        "failed_examples": failed_examples
    }


def clean_solved_flag(series: pd.Series) -> pd.Series:
    mapping_true = {1, "1", "是", "Y", "y", "YES", "Yes", "yes", "TRUE", "True", "true", "已解决", "解决"}
    mapping_false = {0, "0", "否", "N", "n", "NO", "No", "no", "FALSE", "False", "false", "未解决", "没解决"}

    def _parse(v):
        if pd.isna(v):
            return None
        if isinstance(v, bool):
            return v
        if v in mapping_true:
            return True
        if v in mapping_false:
            return False
        s = str(v).strip()
        if s in mapping_true:
            return True
        if s in mapping_false:
            return False
        return None

    return series.apply(_parse)


def clean_text_field(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": None, "None": None, "": None})


def clean_dataframe(df: pd.DataFrame, field_mapping: dict) -> tuple:
    renamed = df.rename(columns={v: k for k, v in field_mapping.items() if v})
    result = pd.DataFrame()
    report = {}

    for std_field in REQUIRED_FIELDS.keys():
        if std_field in renamed.columns:
            result[std_field] = renamed[std_field]
        else:
            result[std_field] = pd.Series([None] * len(renamed))

    result["record_date"] = clean_record_date(result["record_date"])
    date_valid = result["record_date"].notna().sum()
    report["record_date"] = {
        "valid": int(date_valid),
        "invalid": int(len(result) - date_valid)
    }

    for field in NUMERIC_FIELDS:
        result[field], info = clean_numeric(result[field], field)
        report[field] = info

    result["solved_flag"] = clean_solved_flag(result["solved_flag"])
    solved_valid = result["solved_flag"].notna().sum()
    report["solved_flag"] = {
        "valid": int(solved_valid),
        "invalid": int(len(result) - solved_valid)
    }

    for field in ["channel_name", "agent_name", "issue_type", "note"]:
        result[field] = clean_text_field(result[field])

    result = result.drop_duplicates().reset_index(drop=True)

    return result, report
