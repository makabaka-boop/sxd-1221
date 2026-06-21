import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    channels = ["电话客服", "在线客服", "APP客服", "微信客服", "邮件客服"]
    agents = [f"坐席{i:02d}" for i in range(1, 16)]
    issue_types = ["账户问题", "订单查询", "退款申请", "产品咨询", "投诉建议", "物流问题", "技术支持", "会员服务"]
    solved_flags = [True, True, True, True, False]
    notes = [
        "", "", "",
        "客户情绪较激动",
        "需要主管介入",
        "转其他部门处理",
        "客户对解释满意",
        "建议优化流程"
    ]

    end_date = datetime(2026, 6, 20)
    start_date = end_date - timedelta(days=60)
    delta_days = (end_date - start_date).days

    data = []
    for _ in range(n):
        date = start_date + timedelta(days=np.random.randint(0, delta_days + 1),
                                      hours=np.random.randint(8, 22),
                                      minutes=np.random.randint(0, 60))

        channel = np.random.choice(channels, p=[0.3, 0.3, 0.15, 0.15, 0.1])
        agent = np.random.choice(agents)
        issue_type = np.random.choice(issue_types)

        base_resp = np.random.exponential(60)
        if issue_type in ["投诉建议", "技术支持"]:
            base_resp *= 1.8
        if channel == "邮件客服":
            base_resp *= 2.5
        response_seconds = int(round(min(base_resp, 4000)))

        base_score = np.random.normal(82, 12)
        if issue_type == "投诉建议":
            base_score -= 10
        if response_seconds > 300:
            base_score -= response_seconds / 100
        score = int(round(max(0, min(100, base_score))))

        if np.random.random() < 0.05:
            score_val = f"{score}分"
        elif np.random.random() < 0.02:
            score_val = "待评分"
        else:
            score_val = score

        if np.random.random() < 0.03:
            resp_val = f"{response_seconds}s"
        else:
            resp_val = response_seconds

        solved = np.random.choice(solved_flags)
        note = np.random.choice(notes)

        data.append({
            "record_date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "channel_name": channel,
            "agent_name": agent,
            "response_seconds": resp_val,
            "score": score_val,
            "issue_type": issue_type,
            "solved_flag": "是" if solved else "否",
            "note": note
        })

    df = pd.DataFrame(data)

    dup_idx = np.random.choice(n, size=8, replace=False)
    for idx in dup_idx:
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)

    return df


if __name__ == "__main__":
    sample_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(sample_dir, "sample_customer_service.csv")

    df = generate_sample_data()
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"示例数据已生成: {csv_path}，共 {len(df)} 条记录")
