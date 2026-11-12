import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="3D 프린터 데이터 분석", page_icon="🖨️", layout="wide")

KAGGLE_HANDLE = "afumetto/3dprinter"

# 변수 설명 (Kaggle 데이터셋 기준)
DESCRIPTIONS = {
    "layer_height": ("입력", "층 높이 (mm). 작을수록 표면이 매끄럽지만 출력 시간이 길어짐"),
    "wall_thickness": ("입력", "외벽 두께 (mm)"),
    "infill_density": ("입력", "내부 채움 밀도 (%)"),
    "infill_pattern": ("입력", "내부 채움 패턴 (grid, honeycomb)"),
    "nozzle_temperature": ("입력", "노즐 온도 (°C)"),
    "bed_temperature": ("입력", "베드 온도 (°C)"),
    "print_speed": ("입력", "출력 속도 (mm/s)"),
    "material": ("입력", "소재 (abs, pla)"),
    "fan_speed": ("입력", "냉각 팬 속도 (%)"),
    "roughness": ("결과", "표면 거칠기 (μm). 낮을수록 표면이 좋음"),
    "tension_strength": ("결과", "인장 강도 (MPa)"),
    "elongation": ("결과", "연신율 (%)"),
}
TARGETS = ["roughness", "tension_strength", "elongation"]


# ---------------------------------------------------------------- 데이터 로딩
@st.cache_data(show_spinner="Kaggle에서 데이터를 불러오는 중...")
def load_from_kaggle() -> pd.DataFrame:
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
        st.sidebar.info(
            "Kaggle 페이지에서 data.csv를 직접 내려받은 뒤 '업로드'를 이용하세요."
        )
    return None


# ---------------------------------------------------------------- 전처리
def preprocess(raw, remove_outliers):
    log = []
    df = raw.copy()

    # 1) 컬럼명 정리 (공백 제거, 소문자, 오타 수정)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "tension_strenght" in df.columns:
        df = df.rename(columns={"tension_strenght": "tension_strength"})
        log.append("컬럼명 오타 수정: tension_strenght → tension_strength")

    # 2) 중복 제거
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"중복 행 {n_dup}개 제거")

    # 3) 문자열 정리
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    log.append(f"범주형 컬럼 {len(cat_cols)}개 문자열 정리(공백 제거·소문자): {cat_cols}")

    # 4) 결측치 처리 (수치형=중앙값, 범주형=최빈값)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    n_missing = int(df.isna().sum().sum())
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])
    log.append(f"결측치 {n_missing}개 처리 (수치형: 중앙값, 범주형: 최빈값)")

    # 5) 이상치 (IQR 기준)
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
    st.dataframe(df.head(20))

    st.subheader("기초 통계량")
    st.dataframe(df[num_cols].describe().T.round(3))

    if cat_cols:
        st.subheader("범주형 변수 분포")
        cols = st.columns(len(cat_cols))
        for col, name in zip(cols, cat_cols):
            counts = df[name].value_counts().rename_axis(name).reset_index(name="개수")
            col.plotly_chart(px.pie(counts, names=name, values="개수", title=name))

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
        {
            "자료형": df.dtypes.astype(str),
            "결측치": df.isna().sum(),
            "고유값 수": df.nunique(),
        }
    )
    st.dataframe(info)

    model_df = make_model_table(df, encode, scale)
    st.markdown("**최종 전처리 테이블 (인코딩·스케일링 옵션 반영)**")
    st.dataframe(model_df.head(20))

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
        kind, desc = DESCRIPTIONS.get(c, ("-", "-"))
        rows.append({"변수": c, "구분": kind, "설명": desc, "자료형": str(df[c].dtype)})
    st.dataframe(pd.DataFrame(rows))
    st.info("입력 변수(출력 설정)가 결과 변수(거칠기·인장강도·연신율)에 어떤 영향을 주는지 살펴보는 데이터입니다.")

# ---- 시각화
with tab4:
    st.subheader("분포")
    var = st.selectbox("변수 선택", num_cols, key="dist")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(df, x=var, nbins=20, title=f"{var} 히스토그램"))
    right.plotly_chart(px.box(df, y=var, title=f"{var} 박스플롯"))

    st.subheader("두 변수 관계")
    x_col, y_col = st.columns(2)
    x = x_col.selectbox("X축", num_cols, index=0, key="x")
    y_default = num_cols.index(TARGETS[0]) if TARGETS[0] in num_cols else min(1, len(num_cols) - 1)
    y = y_col.selectbox("Y축", num_cols, index=y_default, key="y")
    color = st.selectbox("색상 구분", ["없음"] + cat_cols)
    st.plotly_chart(
        px.scatter(df, x=x, y=y, color=None if color == "없음" else color, title=f"{x} vs {y}")
    )

    st.subheader("상관관계 히트맵")
    corr = df[num_cols].corr().round(2)
    st.plotly_chart(
        px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    )

    available = [t for t in TARGETS if t in corr.columns]
    if available:
        target = st.selectbox("결과 변수와의 상관계수 순위", available)
        rank = corr[target].drop(labels=available, errors="ignore")
        rank = rank.reindex(rank.abs().sort_values(ascending=False).index)
        st.dataframe(rank.rename("상관계수"))

# ---- 소재·패턴 비교
with tab5:
    group_cols = [c for c in ["material", "infill_pattern"] if c in df.columns]
    result_cols = [t for t in TARGETS if t in df.columns]
    if not group_cols or not result_cols:
        st.warning("비교할 컬럼(material, infill_pattern 등)이 없습니다.")
    else:
        g = st.selectbox("그룹 기준", group_cols)
        st.subheader(f"{g}별 결과 평균")
        st.dataframe(df.groupby(g)[result_cols].mean().round(3))
        for t in result_cols:
            st.plotly_chart(px.box(df, x=g, y=t, color=g, title=f"{g}별 {t}"))

# ---- 선형회귀 예측 (layer_height → roughness)
with tab6:
    st.subheader("단순 선형회귀: 레이어 높이 → 표면 거칠기")
    X_COL, Y_COL = "layer_height", "roughness"

    if X_COL not in df.columns or Y_COL not in df.columns:
        st.warning(f"{X_COL}, {Y_COL} 컬럼이 필요합니다.")
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
            # 최소제곱법으로 직선 적합: y = slope * x + intercept
            slope, intercept = np.polyfit(train[X_COL], train[Y_COL], 1)

            def predict(x):
                return slope * np.asarray(x) + intercept

            def evaluate(d):
                y = d[Y_COL].to_numpy()
                p = predict(d[X_COL].to_numpy())
                ss_res = float(((y - p) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rmse = float(np.sqrt(np.mean((y - p) ** 2)))
                mae = float(np.mean(np.abs(y - p)))
                return {"R²": round(r2, 4), "RMSE": round(rmse, 3), "MAE": round(mae, 3)}

            st.markdown(f"**회귀식:** `roughness = {slope:.3f} × layer_height + {intercept:.3f}`")
            st.caption(
                f"레이어 높이가 0.1mm 늘어날 때 거칠기는 약 {slope * 0.1:.2f}μm 변하는 것으로 추정됩니다. "
                f"(상관계수 r = {data[X_COL].corr(data[Y_COL]):.3f})"
            )

            m = pd.DataFrame(
                {"학습": evaluate(train), "테스트": evaluate(test)}
            ).T
            m["데이터 수"] = [len(train), len(test)]
            st.markdown("**모델 성능**")
            st.dataframe(m)

            # 산점도 + 회귀선
            plot_df = pd.concat(
                [train.assign(구분="학습"), test.assign(구분="테스트")]
            )
            fig = px.scatter(plot_df, x=X_COL, y=Y_COL, color="구분", title="실제값과 회귀선")
            xs = np.linspace(data[X_COL].min(), data[X_COL].max(), 100)
            fig.add_trace(go.Scatter(x=xs, y=predict(xs), mode="lines", name="회귀선"))
            st.plotly_chart(fig)

            # 잔차 그림
            resid = test[Y_COL].to_numpy() - predict(test[X_COL].to_numpy())
            rfig = px.scatter(
                x=predict(test[X_COL].to_numpy()), y=resid,
                labels={"x": "예측값", "y": "잔차(실제−예측)"},
                title="테스트 데이터 잔차",
            )
            rfig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(rfig)

            # 새 값 예측
            st.markdown("**거칠기 예측하기**")
            x_min, x_max = float(data[X_COL].min()), float(data[X_COL].max())
            new_x = st.number_input(
                "레이어 높이 (mm)", value=float(round((x_min + x_max) / 2, 3)),
                step=0.01, format="%.3f",
            )
            st.metric("예측 거칠기 (μm)", f"{float(predict(new_x)):.2f}")
            if new_x < x_min or new_x > x_max:
                st.warning(f"학습 데이터 범위({x_min}~{x_max}mm) 밖의 값이라 예측 신뢰도가 낮습니다.")
