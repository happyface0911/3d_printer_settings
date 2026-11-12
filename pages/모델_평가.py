import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="모델 평가", page_icon="📏", layout="wide")

st.title("📏 선형회귀 모델 성능 평가")
st.caption("main.py의 '선형회귀 예측' 탭에서 만든 모델(레이어 높이 → 표면 거칠기)을 평가합니다.")

model = st.session_state.get("reg_model")
if model is None:
    st.warning(
        "평가할 모델이 아직 없습니다. 메인 페이지에서 데이터를 불러오고 "
        "'선형회귀 예측' 탭이 표시된 뒤 이 페이지로 다시 들어와 주세요."
    )
    st.stop()

slope, intercept = model["slope"], model["intercept"]
train, test, data = model["train"], model["test"], model["data"]
X_COL, Y_COL = model["x_col"], model["y_col"]
X_NAME, Y_NAME = "레이어 높이(mm)", "표면 거칠기(μm)"


def predict(v):
    return slope * np.asarray(v) + intercept


def calc_metrics(d, k=1):
    """회귀 성능 지표 계산 (k = 독립 변수 개수)"""
    y = d[Y_COL].to_numpy(dtype=float)
    p = predict(d[X_COL].to_numpy(dtype=float))
    e = y - p
    n = len(y)
    sse = float((e**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")
    nonzero = y != 0
    mape = float(np.mean(np.abs(e[nonzero] / y[nonzero])) * 100) if nonzero.any() else float("nan")
    ev = 1 - float(np.var(e)) / float(np.var(y)) if np.var(y) > 0 else float("nan")
    return {
        "R²": r2,
        "수정 R²": adj,
        "설명 분산 비율": ev,
        "MSE": sse / n,
        "RMSE": float(np.sqrt(sse / n)),
        "MAE": float(np.mean(np.abs(e))),
        "중앙값 절대오차": float(np.median(np.abs(e))),
        "최대 오차": float(np.max(np.abs(e))),
        "MAPE(%)": mape,
        "평균 잔차": float(e.mean()),
        "잔차 표준편차": float(e.std(ddof=1)) if n > 1 else float("nan"),
    }


m_train, m_test = calc_metrics(train), calc_metrics(test)

# ------------------------------------------------------------ 핵심 지표
st.subheader("핵심 지표 (테스트 데이터)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", f"{m_test['R²']:.3f}", f"학습 대비 {m_test['R²'] - m_train['R²']:+.3f}")
c2.metric("RMSE (μm)", f"{m_test['RMSE']:.2f}", f"학습 대비 {m_test['RMSE'] - m_train['RMSE']:+.2f}", delta_color="inverse")
c3.metric("MAE (μm)", f"{m_test['MAE']:.2f}", f"학습 대비 {m_test['MAE'] - m_train['MAE']:+.2f}", delta_color="inverse")
c4.metric("MAPE (%)", f"{m_test['MAPE(%)']:.1f}", f"학습 대비 {m_test['MAPE(%)'] - m_train['MAPE(%)']:+.1f}", delta_color="inverse")
st.caption(
    f"학습 {len(train)}건 / 테스트 {len(test)}건 (테스트 비율 {model['test_ratio']:.0%}, 시드 {model['seed']}). "
    "작은 화살표는 학습 데이터 점수와의 차이입니다. 테스트 점수가 크게 나빠지면 과적합을 의심할 수 있습니다."
)

# ------------------------------------------------------------ 전체 지표표
st.subheader("전체 성능 지표")
EXPLAIN = {
    "R²": "모델이 거칠기 변동의 몇 %를 설명하는지 (1에 가까울수록 좋음)",
    "수정 R²": "독립 변수 개수를 고려해 보정한 R²",
    "설명 분산 비율": "예측 오차의 분산을 제외하고 설명되는 분산의 비율",
    "MSE": "오차 제곱의 평균 (큰 오차에 민감)",
    "RMSE": "MSE의 제곱근. 거칠기와 같은 단위(μm)",
    "MAE": "오차 절댓값의 평균 (μm)",
    "중앙값 절대오차": "이상치 영향이 적은 대표 오차 (μm)",
    "최대 오차": "가장 크게 빗나간 예측의 오차 (μm)",
    "MAPE(%)": "실제값 대비 평균 오차 비율",
    "평균 잔차": "0에 가까우면 예측이 한쪽으로 치우치지 않음",
    "잔차 표준편차": "오차가 흩어진 정도 (μm)",
}
table = pd.DataFrame(
    {
        "지표": list(m_train.keys()),
        "학습": [round(v, 4) for v in m_train.values()],
        "테스트": [round(v, 4) for v in m_test.values()],
        "설명": [EXPLAIN[k] for k in m_train.keys()],
    }
).set_index("지표")
st.dataframe(table)
st.caption("R²·수정 R²는 클수록, 나머지 오차 지표는 작을수록 좋은 모델입니다.")

# ------------------------------------------------------------ 회귀 계수 통계
st.subheader("회귀 계수 통계 (학습 데이터)")
n_tr = len(train)
if n_tr > 2:
    x = train[X_COL].to_numpy(dtype=float)
    y = train[Y_COL].to_numpy(dtype=float)
    resid = y - predict(x)
    s2 = float((resid**2).sum()) / (n_tr - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    se_slope = float(np.sqrt(s2 / sxx))
    se_int = float(np.sqrt(s2 * (1 / n_tr + x.mean() ** 2 / sxx)))
    coef = pd.DataFrame(
        {
            "추정값": [slope, intercept],
            "표준오차": [se_slope, se_int],
            "t 통계량": [slope / se_slope, intercept / se_int],
            "95% 신뢰구간(하한)": [slope - 1.96 * se_slope, intercept - 1.96 * se_int],
            "95% 신뢰구간(상한)": [slope + 1.96 * se_slope, intercept + 1.96 * se_int],
        },
        index=["기울기 (레이어 높이)", "절편"],
    ).round(3)
    st.dataframe(coef)
    st.caption(
        "t 통계량의 절댓값이 약 2 이상이고 신뢰구간이 0을 포함하지 않으면, 레이어 높이가 거칠기에 "
        "의미 있는 영향을 준다고 볼 수 있습니다. (신뢰구간은 1.96 근사값입니다.)"
    )
else:
    st.info("학습 데이터가 부족해 계수 통계를 계산할 수 없습니다.")

# ------------------------------------------------------------ 그래프
st.subheader("예측 vs 실제")
all_pred = pd.concat(
    [
        pd.DataFrame({"실제": train[Y_COL], "예측": predict(train[X_COL]), "구분": "학습"}),
        pd.DataFrame({"실제": test[Y_COL], "예측": predict(test[X_COL]), "구분": "테스트"}),
    ]
)
fig = px.scatter(all_pred, x="실제", y="예측", color="구분", title="실제값 vs 예측값")
lo, hi = float(all_pred[["실제", "예측"]].min().min()), float(all_pred[["실제", "예측"]].max().max())
fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완벽한 예측", line=dict(dash="dash")))
fig.update_layout(xaxis_title="실제 거칠기(μm)", yaxis_title="예측 거칠기(μm)")
st.plotly_chart(fig)
st.caption("점이 점선(완벽한 예측)에 가까울수록 정확합니다. 점선 위쪽은 과대 예측, 아래쪽은 과소 예측입니다.")

st.subheader("잔차 분석")
left, right = st.columns(2)
test_resid = test[Y_COL].to_numpy(dtype=float) - predict(test[X_COL])
all_resid = all_pred["실제"].to_numpy() - all_pred["예측"].to_numpy()

rfig = px.scatter(
    x=predict(test[X_COL]), y=test_resid,
    labels={"x": "예측 거칠기(μm)", "y": "잔차(μm)"}, title="예측값별 잔차 (테스트)",
)
rfig.add_hline(y=0, line_dash="dash")
left.plotly_chart(rfig)
hfig = px.histogram(
    x=all_resid, nbins=15, labels={"x": "잔차(μm)", "count": "빈도"}, title="잔차 분포 (학습+테스트)"
)
right.plotly_chart(hfig)
st.caption(
    "왼쪽: 점이 0선 주변에 무작위로 흩어져야 좋습니다. 곡선 모양이나 부채꼴 패턴이 보이면 직선 모델이 "
    "맞지 않거나 오차 크기가 일정하지 않다는 신호입니다. 오른쪽: 0을 중심으로 좌우 대칭인 종 모양에 가까울수록 좋습니다."
)

st.subheader("레이어 높이별 오차")
by_x = (
    data.assign(abs_err=(data[Y_COL] - predict(data[X_COL])).abs())
    .groupby(X_COL)["abs_err"].mean().reset_index()
)
bfig = px.bar(by_x, x=X_COL, y="abs_err", labels={X_COL: X_NAME, "abs_err": "평균 절대오차(μm)"},
              title="레이어 높이별 평균 절대오차 (전체 데이터)")
st.plotly_chart(bfig)
st.caption("어느 레이어 높이 구간에서 모델이 더 많이 빗나가는지 보여줍니다. 오차가 큰 구간에서는 이 모델을 조심해서 써야 합니다.")

# ------------------------------------------------------------ 교차검증
st.subheader("K-겹 교차검증")
k = st.slider("분할 수 (K)", 3, 10, 5)
seed = st.number_input("교차검증 시드", value=model["seed"], step=1)
if len(data) < k * 2:
    st.warning("데이터가 부족해 이 K값으로 교차검증을 할 수 없습니다.")
else:
    idx = np.random.default_rng(int(seed)).permutation(len(data))
    folds = np.array_split(idx, k)
    rows = []
    for i, fold in enumerate(folds, start=1):
        tr = data.iloc[np.setdiff1d(idx, fold)]
        va = data.iloc[fold]
        if tr[X_COL].nunique() < 2:
            continue
        s, b = np.polyfit(tr[X_COL], tr[Y_COL], 1)
        y_va = va[Y_COL].to_numpy(dtype=float)
        p_va = s * va[X_COL].to_numpy(dtype=float) + b
        sst = ((y_va - y_va.mean()) ** 2).sum()
        rows.append(
            {
                "겹": f"{i}",
                "R²": 1 - ((y_va - p_va) ** 2).sum() / sst if sst > 0 else np.nan,
                "RMSE": float(np.sqrt(np.mean((y_va - p_va) ** 2))),
                "MAE": float(np.mean(np.abs(y_va - p_va))),
            }
        )
    cv = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {"평균": cv[["R²", "RMSE", "MAE"]].mean(), "표준편차": cv[["R²", "RMSE", "MAE"]].std()}
    ).T.round(3)
    st.dataframe(summary)
    cfig = px.bar(cv, x="겹", y="R²", title=f"{k}-겹 교차검증 R²", labels={"겹": "겹 번호"})
    st.plotly_chart(cfig)
    st.caption(
        "데이터를 K등분해 번갈아 검증용으로 쓴 결과입니다. 한 번의 분할보다 안정적인 성능 추정치이며, "
        "겹마다 R²가 크게 들쭉날쭉하면 모델 성능이 데이터 구성에 민감하다는 뜻입니다."
    )

# ------------------------------------------------------------ 종합 해석
st.subheader("종합 해석")
r2_t = m_test["R²"]
if np.isnan(r2_t):
    level = "테스트 데이터가 너무 적어 R²를 계산할 수 없습니다."
elif r2_t >= 0.7:
    level = "레이어 높이만으로 거칠기 변동의 상당 부분을 설명하는 좋은 모델입니다."
elif r2_t >= 0.4:
    level = "레이어 높이가 거칠기에 영향을 주지만, 설명되지 않는 부분도 꽤 남아 있습니다."
else:
    level = "레이어 높이만으로는 거칠기를 충분히 설명하기 어렵습니다."
st.markdown(f"- **설명력:** 테스트 R² = {r2_t:.3f}. {level}")

gap = m_train["R²"] - m_test["R²"]
if not np.isnan(gap):
    if gap > 0.1:
        st.markdown(f"- **일반화:** 학습 R²가 테스트보다 {gap:.2f} 높아 과적합 가능성이 있습니다.")
    else:
        st.markdown("- **일반화:** 학습과 테스트 성능 차이가 크지 않아 새 데이터에도 비슷하게 동작할 것으로 보입니다.")

st.markdown(
    f"- **오차 크기:** 예측은 평균적으로 실제값에서 약 {m_test['MAE']:.2f}μm "
    f"(약 {m_test['MAPE(%)']:.1f}%) 벗어납니다."
)
st.markdown(
    "- **한계:** 노즐 온도, 출력 속도, 팬 속도 등 다른 변수를 쓰지 않은 단순 모델이고, 데이터가 50건 안팎으로 "
    "적어 테스트 비율과 시드에 따라 지표가 달라질 수 있습니다. 성능을 높이려면 다중회귀로 확장해 보세요."
)
