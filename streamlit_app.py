# streamlit_app.py — Simple Leaderboard (No password, no admin)
import os, io, hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import streamlit as st
from sqlalchemy import create_engine, text

# =========================
# 설정
# =========================
ANSWER_CSV_PATH = os.getenv("ANSWER_CSV_PATH", "./secret_data/answer.csv")
DB_URL = os.getenv("DB_URL", "sqlite:///leaderboard.db")
SUBMIT_LIMIT_PER_DAY = int(os.getenv("SUBMIT_LIMIT_PER_DAY", "5"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Seoul")
KEEP_BEST_ONLY = True  # 팀별 최고 점수만 표시에 사용

def utcnow():
    return datetime.now(timezone.utc)

@st.cache_data(show_spinner=False)
def load_answer():
    if not os.path.exists(ANSWER_CSV_PATH):
        st.error("정답 CSV(ANSWER_CSV_PATH)가 없습니다.")
        st.stop()
    df = pd.read_csv(ANSWER_CSV_PATH)
    needed = {"id", "price"}
    if not needed.issubset(df.columns):
        st.error("정답 CSV에는 'id','price' 컬럼이 필요합니다.")
        st.stop()
    df["id"] = df["id"].astype(str)
    return df[["id","price"]].rename(columns={"price":"y_true"})

@st.cache_resource(show_spinner=False)
def get_engine():
    eng = create_engine(DB_URL, future=True)
    with eng.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS submissions(
                sid INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                rmse REAL NOT NULL,
                mae REAL NOT NULL,
                r2 REAL NOT NULL,
                n_samples INTEGER NOT NULL,
                file_sha256 TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            """
        )
    return eng

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def check_submission_columns(df: pd.DataFrame):
    need = {"id","price"}
    if not need.issubset(df.columns):
        missing = ", ".join(sorted(need - set(df.columns)))
        raise ValueError(f"제출 파일에 필수 컬럼 누락: {missing}")

def evaluate_whole(pred_df: pd.DataFrame, ans_df: pd.DataFrame):
    pred = pred_df.copy()
    pred["id"] = pred["id"].astype(str)
    pred = pred[["id","price"]].rename(columns={"price":"y_pred"})
    merged = ans_df.merge(pred, on="id", how="left", validate="one_to_one")

    if merged["y_pred"].isna().any() or len(merged) != len(ans_df):
        raise ValueError("id 불일치/누락/중복이 있습니다. 테스트셋 id와 1:1 일치해야 합니다.")

    y_true = pd.to_numeric(merged["y_true"], errors="raise").to_numpy()
    y_pred = pd.to_numeric(merged["y_pred"], errors="raise").to_numpy()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(merged)}

# =========================
# UI
# =========================
st.set_page_config(page_title="머신러닝 프로젝트 리더보드", layout="centered")
st.title("머신러닝 프로젝트 리더보드")

with st.expander("규정/가이드", expanded=True):
    st.markdown(f"""
- 제출 파일 형식: **CSV** (`id, price`)
- 지표: **RMSE**(주), 참고로 MAE, R²도 함께 표시
- 제출 제한: **팀당 최근 24시간 {SUBMIT_LIMIT_PER_DAY}회**
- 서버 시간대 표기: **{TIMEZONE}**
""")

team = st.text_input("팀명(필수)")

ans_df = load_answer()
engine = get_engine()

uploaded = st.file_uploader("submission.csv 업로드 (id,price)", type=["csv"])

if uploaded is not None:
    if not team.strip():
        st.warning("팀명을 입력하세요.")
        st.stop()

    raw = uploaded.read()
    file_hash = sha256_bytes(raw)

    try:
        sub_df = pd.read_csv(io.BytesIO(raw))
        check_submission_columns(sub_df)
        metrics = evaluate_whole(sub_df, ans_df)
    except Exception as e:
        st.error(f"제출 파일 확인/평가 중 오류: {e}")
        st.stop()

    # 제출 제한
    with engine.begin() as conn:
        since = (utcnow() - timedelta(days=1)).isoformat()
        n_today = conn.execute(
            text("SELECT COUNT(*) FROM submissions WHERE team=:team AND created_at_utc>=:since"),
            {"team": team, "since": since}
        ).scalar_one()
        if n_today >= SUBMIT_LIMIT_PER_DAY:
            st.error(f"제출 한도를 초과했습니다. (최근 24시간 {SUBMIT_LIMIT_PER_DAY}회)")
            st.stop()

    # 저장
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO submissions(team, rmse, mae, r2, n_samples, file_sha256, created_at_utc)
            VALUES(:team, :rmse, :mae, :r2, :n, :sha, :ts)
        """), {
            "team": team.strip(),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "n": metrics["n"],
            "sha": file_hash,
            "ts": utcnow().isoformat()
        })

    st.success("✅ 제출 평가 완료!")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{metrics['rmse']:.5f}")
    c2.metric("MAE",  f"{metrics['mae']:.5f}")
    c3.metric("R²",   f"{metrics['r2']:.4f}")

st.divider()
st.subheader("🏆 Leaderboard (팀별 최고 RMSE 기준)")

query_best = """
SELECT team, MIN(rmse) AS best_rmse, MIN(mae) AS best_mae, MIN(r2) AS best_r2,
       MIN(created_at_utc) AS first_submit_time
FROM submissions
GROUP BY team
""" if KEEP_BEST_ONLY else """
SELECT team, rmse AS best_rmse, mae AS best_mae, r2 AS best_r2, created_at_utc AS first_submit_time
FROM submissions
"""

with engine.begin() as conn:
    lb = pd.read_sql(text(query_best), conn)

if lb.empty:
    st.info("아직 제출이 없습니다. 아래에서 sample_submission.csv를 내려받아 업로드해 보세요!")
else:
    lb = lb.sort_values(["best_rmse","first_submit_time"], ascending=[True, True]).reset_index(drop=True)
    lb["rank"] = range(1, len(lb)+1)
    st.dataframe(lb[["rank","team","best_rmse","best_mae","best_r2","first_submit_time"]],
                 hide_index=True, use_container_width=True)

with st.expander("📄 sample_submission.csv 내려받기"):
    sample = load_answer()[["id"]].copy()
    sample["price"] = 0
    st.download_button("Download sample_submission.csv",
                       sample.to_csv(index=False).encode(),
                       "sample_submission.csv", "text/csv")
