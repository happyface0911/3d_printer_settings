import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="3D 프린터 데이터 분석", page_icon="🖨️", layout="wide")

KAGGLE_HANDLE = "afumetto/3dprinter"

# ---------------------------------------------------------------- 한글 표시용 사전
LABELS = {
    "layer_height": "레이어 높이(mm)",
    "wall_thickness": "외벽 두께(mm)",
    "infill_density": "채움 밀도(%)",
    "infill_pattern": "채움 패턴",
    "nozzle_temperature": "노즐 온도(°C)",
    "bed_temperature": "베드 온도(°C)",
    "print_speed": "출력 속도(mm/s)",
    "material": "소재",
    "fan_speed": "팬 속도(%)",
    "roughness": "표면 거칠기(μm)",
    "tension_strength": "인장 강도(MPa)",
    "elongation": "연신율(%)",
    "count": "빈도",
}
VALUE_KO = {"abs": "ABS", "pla": "PLA", "grid": "격자형(grid)", "honeycomb": "벌집형(honeycomb)"}

DESCRIPTIONS = {
    "layer_height": ("입력", "한 층의 두께. 작을수록 표면이 매끄럽지만 출력 시간이 길어짐"),
    "wall_thickness": ("입력", "출력물 외벽의 두께"),
    "infill_density": ("입력", "내부를 채우는 밀도"),
    "infill_pattern": ("입력", "내부 채움 패턴 (격자형, 벌집형)"),
    "nozzle_temperature": ("입력", "필라멘트를 녹이는 노즐 온도"),
    "bed_temperature": ("입력", "출력물이 놓이는 베드 온도"),
    "print_speed": ("입력", "노즐이 움직이는 출력 속도"),
    "material": ("입력", "사용한 소재 (ABS, PLA)"),
    "fan_speed": ("입력", "냉각 팬 속도"),
    "roughness": ("결과", "표면 거칠기. 낮을수록 표면이 좋음"),
    "tension_strength": ("결과", "끊어지기 전까지 버티는 인장 강도"),
    "elongation": ("결과", "끊어지기 전까지 늘어나는 비율(연신율)"),
}
TARGETS = ["roughness", "tension_strength", "elongation"]


def ko(c):
    return LABELS.get(c, c)


def to_view(d):
    """화면 표시용: 범주형 값을 한글로, 컬럼명을 한글로 바꾼 복사본."""
    out = d.copy()
    for c in out.select_dtypes(exclude="number").columns:
        out[c] = out[c].map(VALUE_KO).fillna(out[c])
    return out


# ---------------------------------------------------------------- 데이터 로딩
@st.cache_data(show_spinner="Kaggle에서 데이터를 불러오는 중...")
def load_from_kaggle():
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_HANDLE)
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("다운로드한 폴더에서 CSV 파일을 찾지 못했습니다.")
    return pd.read_csv(csv_files[0])


def get_raw_data():
    st.sidebar.header("📂 데이터 불러오기")
    source = st.sidebar.radio(
        "데이터 출처",
        ["Kaggle에서 자동 불러오기", "data.csv 업로드", "작업 폴더의 data.csv"],
    )

    try:
        if source == "Kaggle에서 자동 불러오기":
            return load_from_kaggle()
        if source == "data.csv 업로드":
            file = st.sidebar.file_uploader("CSV 파일", type="csv")
            return pd.read_csv(file) if file else None
        if os.path.exists("data.csv"):
            return pd.read_csv("data.csv")
        st.sidebar.warning("작업 폴더에 data.csv가 없습니다.")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"불러오기 실패: {e}")
        st.sidebar.info("Kaggle 페이지에서 data.csv를 직접 내려받은 뒤 '업로드'를 이용하세요.")
    return None


# ---------------------------------------------------------------- 전처리
def preprocess(raw, remove_outliers):
    log = []
    df = raw.copy()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "tension_strenght" in df.columns:
        df = df.rename(columns={"tension_strenght": "tension_strength"})
        log.append("컬럼명 오타 수정: tension_strenght → tension_strength")

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"중복 행 {n_dup}개 제거")

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    log.append(f"범주형 컬럼 {len(cat_cols)}개 문자열 정리(공백 제거·소문자): {cat_cols}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    n_missing = int(df.isna().sum().sum())
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])
    log.append(f"결측치 {n_missing}개 처리 (수치형: 중앙값, 범주형: 최빈값)")

    q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
    iqr = q3 - q1
    mask = ((df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)).any(axis=1)
    n_out = int(mask.sum())
    if remove_outliers:
        df = df[~mask].reset_index(drop=True)
        log.append(f"IQR 기준 이상치 행 {n_out}개 제거")
    else:
        log.append(f"IQR 기준 이상치 행 {n_out}개 발견 (제거하지 않음)")

    return df, log


def make_model_table(df, encode, scale):
    out = df.copy()
    if encode:
        cats = out.select_dtypes(exclude="number").columns.tolist()
        out = pd.get_dummies(out, columns=cats, dtype=int)
    if scale:
        num = out.select_dtypes(include="number").columns
        rng = (out[num].max() - out[num].min()).replace(0, 1)
        out[num] = (out[num] - out[num].min()) / rng
    return out


# ---------------------------------------------------------------- 화면
st.title("🖨️ 3D 프린터 출력 데이터 분석")
st.caption("Kaggle: 3D Printer Dataset for Mechanical Engineers (afumetto/3dprinter)")

raw = get_raw_data()
if raw is None:
    st.info("왼쪽 사이드바에서 데이터를 불러와 주세요.")
    st.stop()

st.sidebar.header("⚙️ 전처리 옵션")
remove_outliers = st.sidebar.checkbox("이상치(IQR) 제거", value=False)
encode = st.sidebar.checkbox("범주형 원-핫 인코딩 (다운로드용)", value=True)
scale = st.sidebar.checkbox("Min-Max 스케일링 (다운로드용)", value=False)

df, log = preprocess(raw, remove_outliers)
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
dview = to_view(df)  # 차트 표시용(범주형 값 한글화)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📋 개요", "🧹 전처리", "📖 변수 설명", "📊 시각화", "🔍 소재·패턴 비교", "📈 선형회귀 예측"]
)

# ---- 개요
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 수", f"{len(df):,}")
    c2.metric("열 수", df.shape[1])
    c3.metric("수치형 변수", len(num_cols))
    c4.metric("범주형 변수", len(cat_cols))

    st.subheader("데이터 미리보기")
    st.dataframe(dview.head(20).rename(columns=LABELS))

    st.subheader("기초 통계량")
    desc = df[num_cols].describe().T.round(3).rename(
        index=LABELS,
        columns={"count": "개수", "mean": "평균", "std": "표준편차", "min": "최솟값",
                 "25%": "25%", "50%": "중앙값", "75%": "75%", "max": "최댓값"},
    )
    st.dataframe(desc)
    st.caption("각 수치형 변수의 평균, 흩어진 정도(표준편차), 최솟값·최댓값을 한눈에 보여주는 표입니다.")

    if cat_cols:
        st.subheader("범주형 변수 분포")
        cols = st.columns(len(cat_cols))
        for col, name in zip(cols, cat_cols):
            counts = dview[name].value_counts().rename_axis(name).reset_index(name="개수")
            col.plotly_chart(px.pie(counts, names=name, values="개수", title=ko(name)))
        st.caption("소재와 채움 패턴이 전체 실험에서 각각 어떤 비율로 사용되었는지 보여줍니다.")

# ---- 전처리
with tab2:
    st.subheader("전처리 로그")
    for line in log:
        st.write("✅ " + line)

    a, b = st.columns(2)
    with a:
        st.markdown("**원본 데이터 정보**")
        st.write(f"형태: {raw.shape[0]}행 × {raw.shape[1]}열")
        st.write(f"결측치: {int(raw.isna().sum().sum())}개")
    with b:
        st.markdown("**전처리 후 데이터 정보**")
        st.write(f"형태: {df.shape[0]}행 × {df.shape[1]}열")
        st.write(f"결측치: {int(df.isna().sum().sum())}개")

    st.markdown("**컬럼별 자료형 / 결측치 / 고유값 수**")
    info = pd.DataFrame(
        {"자료형": df.dtypes.astype(str), "결측치": df.isna().sum(), "고유값 수": df.nunique()}
    ).rename(index=LABELS)
    st.dataframe(info)

    model_df = make_model_table(df, encode, scale)
    st.markdown("**최종 전처리 테이블 (인코딩·스케일링 옵션 반영)**")
    st.dataframe(model_df.head(20).rename(columns=LABELS))
    st.caption("원-핫 인코딩은 소재·패턴 같은 글자 값을 0/1 숫자 열로 바꾸고, 스케일링은 모든 값을 0~1 범위로 맞춥니다.")

    d1, d2 = st.columns(2)
    d1.download_button(
        "정리된 데이터 다운로드 (CSV)",
        df.to_csv(index=False).encode("utf-8-sig"),
        "data_cleaned.csv",
        "text/csv",
    )
    d2.download_button(
        "모델링용 데이터 다운로드 (CSV)",
        model_df.to_csv(index=False).encode("utf-8-sig"),
        "data_model.csv",
        "text/csv",
    )

# ---- 변수 설명
with tab3:
    st.subheader("변수 설명")
    rows = []
    for c in df.columns:
        kind, desc_text = DESCRIPTIONS.get(c, ("-", "-"))
        rows.append({"변수": ko(c), "컬럼명": c, "구분": kind, "설명": desc_text, "자료형": str(df[c].dtype)})
    st.dataframe(pd.DataFrame(rows))
    st.info("입력 변수(출력 설정)가 결과 변수(거칠기·인장강도·연신율)에 어떤 영향을 주는지 살펴보는 데이터입니다.")

# ---- 시각화
with tab4:
    st.subheader("분포")
    var = st.selectbox("변수 선택", num_cols, format_func=ko, key="dist")
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(df, x=var, nbins=20, labels=LABELS, title=f"{ko(var)} 히스토그램")
    )
    right.plotly_chart(px.box(df, y=var, labels=LABELS, title=f"{ko(var)} 박스플롯"))
    st.caption(
        f"{ko(var)}의 값이 어느 구간에 많이 몰려 있는지(왼쪽)와 중앙값·사분위 범위(오른쪽)를 보여줍니다. "
        f"평균 {df[var].mean():.2f}, 중앙값 {df[var].median():.2f}."
    )

    st.subheader("두 변수 관계")
    x_col, y_col = st.columns(2)
    x = x_col.selectbox("X축", num_cols, index=0, format_func=ko, key="x")
    y_default = num_cols.index(TARGETS[0]) if TARGETS[0] in num_cols else min(1, len(num_cols) - 1)
    y = y_col.selectbox("Y축", num_cols, index=y_default, format_func=ko, key="y")
    color = st.selectbox("색상 구분", ["없음"] + cat_cols, format_func=lambda c: c if c == "없음" else ko(c))
    st.plotly_chart(
        px.scatter(
            dview, x=x, y=y, color=None if color == "없음" else color,
            labels=LABELS, title=f"{ko(x)} vs {ko(y)}",
        )
    )
    r = df[x].corr(df[y])
    strength = "강한" if abs(r) >= 0.7 else "중간 정도의" if abs(r) >= 0.4 else "약한"
    direction = "양(+)" if r > 0 else "음(-)"
    st.caption(
        f"점 하나가 실험 한 건입니다. 두 변수의 상관계수는 {r:.2f}로, {strength} {direction}의 관계를 보입니다."
    )

    st.subheader("상관관계 히트맵")
    corr = df[num_cols].corr().round(2)
    st.plotly_chart(
        px.imshow(
            corr.rename(index=LABELS, columns=LABELS),
            text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
    )
    st.caption("빨간색은 함께 커지는 관계(+), 파란색은 반대로 움직이는 관계(−)이며, 색이 진할수록 관계가 강합니다.")

    available = [t for t in TARGETS if t in corr.columns]
    if available:
        target = st.selectbox("결과 변수와의 상관계수 순위", available, format_func=ko)
        rank = corr[target].drop(labels=available, errors="ignore")
        rank = rank.reindex(rank.abs().sort_values(ascending=False).index)
        st.dataframe(rank.rename("상관계수").rename(index=LABELS))
        st.caption(f"{ko(target)}과(와) 관계가 큰 입력 변수 순서입니다. 절댓값이 클수록 영향이 큰 변수로 볼 수 있습니다.")

# ---- 소재·패턴 비교
with tab5:
    group_cols = [c for c in ["material", "infill_pattern"] if c in df.columns]
    result_cols = [t for t in TARGETS if t in df.columns]
    if not group_cols or not result_cols:
        st.warning("비교할 컬럼(소재, 채움 패턴 등)이 없습니다.")
    else:
        g = st.selectbox("그룹 기준", group_cols, format_func=ko)
        st.subheader(f"{ko(g)}별 결과 평균")
        st.dataframe(dview.groupby(g)[result_cols].mean().round(3).rename(columns=LABELS))
        st.caption(f"{ko(g)}에 따라 거칠기·인장 강도·연신율의 평균이 어떻게 달라지는지 비교한 표입니다.")
        for t in result_cols:
            st.plotly_chart(
                px.box(dview, x=g, y=t, color=g, labels=LABELS, title=f"{ko(g)}별 {ko(t)}")
            )
            st.caption(f"{ko(g)}별 {ko(t)}의 분포입니다. 상자가 높을수록 값이 크고, 상자가 길수록 편차가 큽니다.")

# ---- 선형회귀 예측 (layer_height → roughness)
with tab6:
    st.subheader("단순 선형회귀: 레이어 높이 → 표면 거칠기")
    X_COL, Y_COL = "layer_height", "roughness"

    if X_COL not in df.columns or Y_COL not in df.columns:
        st.warning(f"{ko(X_COL)}, {ko(Y_COL)} 컬럼이 필요합니다.")
    else:
        c1, c2 = st.columns(2)
        test_ratio = c1.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05)
        seed = c2.number_input("랜덤 시드", value=42, step=1)

        data = df[[X_COL, Y_COL]].dropna()
        order = np.random.default_rng(int(seed)).permutation(len(data))
        n_test = max(1, int(round(len(data) * test_ratio)))
        test = data.iloc[order[:n_test]]
        train = data.iloc[order[n_test:]]

        if len(train) < 2 or train[X_COL].nunique() < 2:
            st.warning("학습 데이터가 부족합니다. 테스트 비율을 낮춰 보세요.")
        else:
            slope, intercept = np.polyfit(train[X_COL], train[Y_COL], 1)

            def predict(v):
                return slope * np.asarray(v) + intercept

            def evaluate(d):
                y = d[Y_COL].to_numpy()
                p = predict(d[X_COL].to_numpy())
                ss_res = float(((y - p) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rmse = float(np.sqrt(np.mean((y - p) ** 2)))
                mae = float(np.mean(np.abs(y - p)))
                return {"R²": round(r2, 4), "RMSE": round(rmse, 3), "MAE": round(mae, 3)}

            st.markdown(f"**회귀식:** `표면 거칠기 = {slope:.3f} × 레이어 높이 + {intercept:.3f}`")
            st.caption(
                f"레이어 높이가 0.1mm 늘어날 때 거칠기는 약 {slope * 0.1:.2f}μm 변하는 것으로 추정됩니다. "
                f"(상관계수 r = {data[X_COL].corr(data[Y_COL]):.3f})"
            )

            m = pd.DataFrame({"학습": evaluate(train), "테스트": evaluate(test)}).T
            m["데이터 수"] = [len(train), len(test)]
            st.markdown("**모델 성능**")
            st.dataframe(m)
            st.caption(
                "R²는 1에 가까울수록 모델이 데이터를 잘 설명한다는 뜻이고, RMSE·MAE는 예측이 실제와 평균적으로 "
                "얼마나 벗어나는지(μm)를 나타내므로 작을수록 좋습니다. 테스트 점수가 학습 점수보다 크게 낮으면 과적합을 의심해 볼 수 있습니다."
            )

            plot_df = pd.concat([train.assign(구분="학습"), test.assign(구분="테스트")])
            fig = px.scatter(plot_df, x=X_COL, y=Y_COL, color="구분", labels=LABELS, title="실제값과 회귀선")
            xs = np.linspace(data[X_COL].min(), data[X_COL].max(), 100)
            fig.add_trace(go.Scatter(x=xs, y=predict(xs), mode="lines", name="회귀선"))
            st.plotly_chart(fig)
            st.caption("점은 실제 측정값, 직선은 모델이 예측한 값입니다. 점들이 직선 가까이에 모여 있을수록 예측이 정확합니다.")

            resid = test[Y_COL].to_numpy() - predict(test[X_COL].to_numpy())
            rfig = px.scatter(
                x=predict(test[X_COL].to_numpy()), y=resid,
                labels={"x": "예측값(μm)", "y": "잔차(실제−예측, μm)"},
                title="테스트 데이터 잔차",
            )
            rfig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(rfig)
            st.caption("잔차는 실제값에서 예측값을 뺀 오차입니다. 점이 0선 주변에 고르게 흩어져 있으면 모델이 적절하고, 한쪽으로 치우치거나 곡선 모양이 보이면 직선 모델의 한계를 뜻합니다.")

            st.markdown("**거칠기 예측하기**")
            x_min, x_max = float(data[X_COL].min()), float(data[X_COL].max())
            new_x = st.number_input(
                "레이어 높이 (mm)", value=float(round((x_min + x_max) / 2, 3)),
                step=0.01, format="%.3f",
            )
            st.metric("예측 표면 거칠기 (μm)", f"{float(predict(new_x)):.2f}")
            if new_x < x_min or new_x > x_max:
                st.warning(f"학습 데이터 범위({x_min}~{x_max}mm) 밖의 값이라 예측 신뢰도가 낮습니다.")
