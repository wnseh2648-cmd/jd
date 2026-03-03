import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from openai import OpenAI
import platform
import requests
import xml.etree.ElementTree as ET
import os
import re
import urllib.parse
from matplotlib.patches import FancyBboxPatch

# 💡 머신러닝 라이브러리 (선형 회귀, 스케일러, 랜덤 포레스트)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =================================================
# [STEP 1] 대시보드 기본 설정 및 프리미엄 CSS 테마 적용
# =================================================
st.set_page_config(page_title="부동산 AI 분석 대시보드", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    /* 기본 폰트 설정 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* KPI 카드 스타일링 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 6px solid #1E3A8A;
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
    }
    
    /* 제목 및 텍스트 스타일링 */
    h1 { color: #0F172A; font-weight: 800; padding-bottom: 0.5rem; }
    h3 { color: #1E40AF; font-weight: 700; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #DBEAFE; padding-bottom: 0.5rem;}
    h4 { color: #334155; font-weight: 600; margin-bottom: 1rem;}
    
    /* 사이드바 커스텀 */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    
    /* 알림창 커스텀 */
    div.stAlert {
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        color: #0369A1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* ⭐ 프리미엄 해석 가이드 커스텀 카드 스타일 (모든 섹션 공통 적용) */
    .analysis-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .analysis-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
    }
    .analysis-content {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
    }
    .factor-positive { color: #10B981; font-weight: 600; }
    .factor-negative { color: #EF4444; font-weight: 600; }
    .highlight { background-color: #FEF3C7; font-weight: 600; padding: 0 4px; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

# 스트림릿 시크릿에서 키를 불러옵니다.
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)
if "client" not in globals():
    st.error("OpenAI client가 정의되지 않았습니다.")
    st.stop()

# 공공데이터 API 설정
MOLIT_SERVICE_KEY = st.secrets["MOLIT_SERVICE_KEY"]
MOLIT_MONTHS_BACK = 12       
MOLIT_NUM_ROWS = 1000        
MOLIT_MAX_PAGES = 30         

# 폰트 및 시각화 설정
if platform.system() == "Windows":
    mpl.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    mpl.rcParams["font.family"] = "AppleGothic"
else:
    mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams['figure.dpi'] = 150  

# =================================================
# [STEP 2] 거시경제(매크로) 데이터 및 지역 맵핑 로직
# =================================================
@st.cache_data
def load_macro():
    df = pd.read_csv("아파트_거래량_가격_지수_금리_통합본.csv", encoding="cp949")
    df["연월"] = pd.to_datetime(df["연월"], errors="coerce")
    return df

df = load_macro()
REGIONS = sorted(df["지역"].dropna().unique())

def detect_region_from_question(q: str):
    q = (q or "").strip()
    if not q: return None
    candidates = [r for r in REGIONS if r in q]
    return sorted(candidates, key=len, reverse=True)[0] if candidates else None

SIDO_PREFIX_TO_REGION = {
    "11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주", "30": "대전",
    "31": "울산", "36": "세종", "41": "경기", "42": "강원", "43": "충북", "44": "충남",
    "45": "전북", "46": "전남", "47": "경북", "48": "경남", "50": "제주"
}

def region_from_lawd_cd(lawd_cd: str):
    if not lawd_cd or len(lawd_cd) < 2: return None
    return SIDO_PREFIX_TO_REGION.get(lawd_cd[:2])

def resolve_macro_region_name(candidate: str):
    if not candidate: return None
    if candidate in REGIONS: return candidate
    hits = [r for r in REGIONS if candidate in r]
    return sorted(hits, key=len)[0] if hits else None

LAWD_CACHE_PATH = "lawd_codes.csv"
LAWD_SOURCE_URL = "https://raw.githubusercontent.com/yunzae/districtCodeTextToSql/main/%E1%84%87%E1%85%A5%E1%86%B8%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%83%E1%85%A9%E1%86%BC%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3.csv"

@st.cache_data(ttl=60 * 60 * 24)
def load_lawd_codes() -> pd.DataFrame:
    if os.path.exists(LAWD_CACHE_PATH):
        raw = pd.read_csv(LAWD_CACHE_PATH, header=None, dtype=str, encoding="utf-8")
    else:
        try:
            raw = pd.read_csv(LAWD_SOURCE_URL, header=None, dtype=str)
            raw.to_csv(LAWD_CACHE_PATH, index=False, header=False, encoding="utf-8")
        except Exception:
            return pd.DataFrame(columns=["code10", "name"])
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["code10", "name"]
    raw["code10"] = raw["code10"].astype(str).str.strip()
    raw["name"] = raw["name"].astype(str).str.strip()
    return raw[raw["code10"].str.match(r"^\d{10}$", na=False)].reset_index(drop=True)

def build_lawd_maps(lawd_df: pd.DataFrame):
    sigungu_map, dong_tmp, dong_base_tmp = {}, {}, {}
    if lawd_df.empty: return sigungu_map, dong_tmp, dong_base_tmp
    
    for _, row in lawd_df.iterrows():
        code5 = row["code10"][:5]
        tokens = str(row["name"]).split()
        if len(tokens) >= 2:
            sido, sigungu = tokens[0], tokens[1]
            if sigungu.endswith(("시", "군", "구")):
                sigungu_map[sigungu] = code5
                sigungu_map[f"{sido} {sigungu}"] = code5
        if len(tokens) >= 3:
            dong = tokens[2]
            if dong.endswith(("동", "읍", "면", "리")):
                dong_tmp.setdefault(dong, set()).add(code5)
                base = dong[:-1]
                if base: dong_base_tmp.setdefault(base, set()).add(code5)

    dong_map_unique = {d: list(codes)[0] for d, codes in dong_tmp.items() if len(codes) == 1}
    dong_base_unique = {d: list(codes)[0] for d, codes in dong_base_tmp.items() if len(codes) == 1}
    return sigungu_map, dong_map_unique, dong_base_unique

SIGUNGU_MAP, DONG_UNIQUE_MAP, DONG_BASE_UNIQUE_MAP = build_lawd_maps(load_lawd_codes())
SIGUNGU_KEYS = sorted(SIGUNGU_MAP.keys(), key=len, reverse=True)
DONG_KEYS = sorted(DONG_UNIQUE_MAP.keys(), key=len, reverse=True)
DONG_BASE_KEYS = sorted(DONG_BASE_UNIQUE_MAP.keys(), key=len, reverse=True)

def detect_lawd_cd_from_question(q: str):
    q2 = (q or "").replace(" ", "")
    for k in SIGUNGU_KEYS:
        if k.replace(" ", "") in q2: return SIGUNGU_MAP[k], k
    for d in DONG_KEYS:
        if d.replace(" ", "") in q2: return DONG_UNIQUE_MAP[d], d
    for b in DONG_BASE_KEYS:
        if b.replace(" ", "") in q2: return DONG_BASE_UNIQUE_MAP[b], b
    return None, None

# =================================================
# [STEP 3] 국토부 API 수집 및 데이터 정규화 모듈
# =================================================
def norm_txt(s: str) -> str:
    s = re.sub(r"[()\[\]{}<>.,·\-_/\s]", "", str(s or "").lower())
    return s

def bigram_set(s: str) -> set:
    s = norm_txt(s)
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else set()

def recent_yms(months_back: int) -> list:
    end = pd.Timestamp.today().to_period("M").to_timestamp()
    return [(end - pd.DateOffset(months=i)).strftime("%Y%m") for i in range(months_back)]

@st.cache_data(ttl=60 * 60 * 6)
def fetch_molit_month_allpages(lawd_cd: str, deal_ymd: str, service_key: str,
                               num_rows: int = 1000, max_pages: int = 30) -> pd.DataFrame:
    url = "http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
    all_rows = []
    page = 1
    decoded_key = urllib.parse.unquote(service_key) 

    while True:
        params = {"serviceKey": decoded_key, "LAWD_CD": lawd_cd, "DEAL_YMD": deal_ymd, 
                  "pageNo": page, "numOfRows": num_rows}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            raise RuntimeError("XML 파싱 오류 (API 서버 에러 가능성)")

        if root.tag == "OpenAPI_ServiceResponse":
            raise RuntimeError(f"API 요청 실패: {root.findtext('.//returnAuthMsg') or '인증 오류'}")

        if (root.findtext(".//header/resultCode") or "").strip() not in ("00", "000"):
            raise RuntimeError("MOLIT API 정상 응답 아님")

        items = root.findall(".//item")
        for it in items:
            row_dict = {}
            for child in list(it):
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag 
                row_dict[tag] = (child.text or "").strip()
            all_rows.append(row_dict)

        try: total_count = int(root.findtext(".//totalCount") or "0")
        except ValueError: total_count = 0

        if len(items) == 0 or page * num_rows >= total_count or page >= max_pages:
            break
        page += 1

    return pd.DataFrame(all_rows)

def normalize_trade_df(trade: pd.DataFrame) -> pd.DataFrame:
    if trade is None or trade.empty or len(trade.columns) == 0: return pd.DataFrame()
    t = trade.copy()

    rename_dict = {
        "aptNm": "아파트", "APT_NM": "아파트", "아파트명": "아파트", "단지명": "아파트", "단지": "아파트",
        "dealAmount": "거래금액", "excluUseAr": "전용면적", "floor": "층", "umdNm": "법정동",
        "jibun": "지번", "roadNm": "도로명", "dealYear": "년", "dealMonth": "월", "dealDay": "일"
    }
    t = t.rename(columns=rename_dict)

    if "거래금액" in t.columns:
        t["거래금액"] = pd.to_numeric(t["거래금액"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        t["거래금액_억원"] = t["거래금액"] / 10000.0
    if "전용면적" in t.columns:
        t["전용면적"] = pd.to_numeric(t["전용면적"], errors="coerce")
    if all(c in t.columns for c in ["년", "월", "일"]):
        t["계약일"] = pd.to_datetime(t["년"].astype(str).str.zfill(4) + "-" + 
                                  t["월"].astype(str).str.zfill(2) + "-" + 
                                  t["일"].astype(str).str.zfill(2), errors="coerce")

    keep = [c for c in ["아파트", "거래금액_억원", "전용면적", "층", "법정동", "계약일"] if c in t.columns]
    return t[keep].copy() if keep else t

def summarize_trade(tr: pd.DataFrame) -> dict:
    if tr is None or tr.empty: return {"n": 0}
    out = {"n": int(len(tr))}
    if "거래금액_억원" in tr.columns and len(s := tr["거래금액_억원"].dropna()) > 0:
        out.update({"avg_eok": float(s.mean()), "med_eok": float(s.median()), "min_eok": float(s.min()), "max_eok": float(s.max())})
    if "계약일" in tr.columns and "거래금액_억원" in tr.columns:
        ts = tr.dropna(subset=["계약일", "거래금액_억원"]).copy()
        if len(ts) >= 3:
            ts["ym"] = ts["계약일"].dt.strftime("%Y-%m")
            m = ts.groupby("ym")["거래금액_억원"].median().sort_index()
            out["monthly_median"] = m.tail(12).to_dict()
    return out

# =================================================
# [STEP 4] Streamlit UI 구성 (사이드바 및 메인 컨텐츠)
# =================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("데이터 필터링")
    st.markdown("분석할 지역과 기간을 선택하세요.")
    
    region = st.selectbox("📍 지역 선택", sorted(df["지역"].dropna().unique()))
    
    df["연월_str"] = df["연월"].dt.strftime("%Y-%m")
    months = sorted(df["연월_str"].dropna().unique())
    
    start_month = st.selectbox("🗓️ 시작 월", months, index=0)
    end_month = st.selectbox("🗓️ 종료 월", months, index=len(months) - 1)
    
    st.markdown("---")
    st.markdown("👨‍💻 **데이터 파이프라인**\n- 공공데이터포털 실거래가 API\n- 부동산 통계정보시스템\n- 한국은행 경제통계시스템 API\n- 머신러닝 회귀분석 \n- OpenAI GPT-5.2")

filtered = df[(df["지역"] == region) & 
              (df["연월"] >= pd.to_datetime(start_month)) & 
              (df["연월"] <= pd.to_datetime(end_month) + pd.offsets.MonthEnd(0))].sort_values("연월")

st.title(f"🏢 {region} 부동산 AI 리서치 대시보드")
st.markdown("거시경제 지표, 실거래 데이터, 머신러닝 회귀분석을 결합한 실시간 시장 동향 예측 시스템입니다.")
st.markdown("---")

if filtered.empty:
    st.warning("선택 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# 주요 지표 KPI
col1, col2, col3, col4 = st.columns(4)
col1.metric("월 평균 거래량", f"{int(filtered['거래량'].dropna().mean()):,}건")
col2.metric("평균 매매가격", f"{round(filtered['평균매매가격(천만원)'].dropna().mean(), 1)}천만원")
col3.metric("평균 매매지수", round(filtered["매매지수"].dropna().mean(), 1), help="기준 시점(100) 대비 아파트 가격의 변동 수준을 나타내는 지표입니다.")

current_rate = filtered['기준금리(%)'].dropna().iloc[-1] if not filtered['기준금리(%)'].dropna().empty else 0
col4.metric("현재 기준금리", f"{round(current_rate, 2)}%")
st.caption("※ **매매지수란?** 특정 기준 시점의 아파트 가격을 '100'으로 두고, 현재 가격이 얼마나 오르고 내렸는지를 한눈에 비교하기 쉽게 만든 지표입니다.")
st.markdown("<br>", unsafe_allow_html=True)

# =================================================
# 시각화 차트 1: 다중 지역 비교 (프리미엄 라인)
# =================================================
st.markdown("### 📊 다중 지역 크로스 분석 (매매량 vs 기준금리)")

compare_regions = st.multiselect(
    "비교할 지역들을 자유롭게 선택하세요.",
    options=REGIONS,
    default=[region]
)

if compare_regions:
    fig_comp, ax1_comp = plt.subplots(figsize=(12, 4.5))
    fig_comp.patch.set_facecolor('white')
    mask_date = (df["연월"] >= pd.to_datetime(start_month)) & (df["연월"] <= pd.to_datetime(end_month) + pd.offsets.MonthEnd(0))
    
    for r in compare_regions:
        r_data = df[(df["지역"] == r) & mask_date].sort_values("연월")
        if not r_data.empty:
            ax1_comp.plot(r_data["연월"], r_data["거래량"], marker='o', markersize=5, label=f"{r} 매매량", linewidth=2, alpha=0.8)

    ax1_comp.set_ylabel("매매량 (건)", fontweight='bold')
    ax1_comp.set_xlabel("연월", fontweight='bold')
    ax1_comp.grid(alpha=0.3, linestyle='--')
    ax1_comp.spines['top'].set_visible(False)
    ax1_comp.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=len(compare_regions), frameon=False)
    
    ax2_comp = ax1_comp.twinx()
    rate_data = df[mask_date][["연월", "기준금리(%)"]].drop_duplicates().sort_values("연월")
    
    ax2_comp.plot(rate_data["연월"], rate_data["기준금리(%)"], color="#DC2626", linestyle=":", linewidth=3, label="기준금리(%)")
    ax2_comp.set_ylabel("기준금리 (%)", color="#DC2626", fontweight='bold')
    ax2_comp.spines['top'].set_visible(False)
    ax2_comp.legend(loc="upper right", bbox_to_anchor=(1.0, 1.15), frameon=False)

    fig_comp.tight_layout()
    st.pyplot(fig_comp)
else:
    st.warning("비교할 지역을 하나 이상 선택해주세요.")

st.markdown("<br>", unsafe_allow_html=True)

# =================================================
# 시각화 차트 2: 매크로 시장 추세 분석 및 히트맵
# =================================================
st.markdown("### 📈 매크로 시장 추세 분석")
colA, colB = st.columns(2)

with colA:
    st.markdown("#### 1. 거래량 추세 및 2026 선형 회귀 예측")
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    fig1.patch.set_facecolor('white')
    df_pred = filtered[["연월", "거래량"]].dropna().copy().reset_index(drop=True)
    
    if len(df_pred) >= 3:
        df_pred["t"] = np.arange(len(df_pred))
        coef = np.polyfit(df_pred["t"], df_pred["거래량"], 1)
        all_future = pd.date_range(start=df_pred["연월"].iloc[-1] + pd.offsets.MonthBegin(1), end=pd.Timestamp("2026-12-01"), freq="MS")
        
        ax1.fill_between(df_pred["연월"], df_pred["거래량"], color="#3B82F6", alpha=0.15)
        ax1.plot(df_pred["연월"], df_pred["거래량"], color="#1D4ED8", linewidth=2.5, label="실제 거래량")
        
        if len(all_future) > 0:
            last_date = df_pred["연월"].iloc[-1]
            future_dates = [last_date] + list(all_future[all_future.year == 2026])
            last_t = len(df_pred) - 1
            future_t = np.arange(last_t, last_t + len(all_future[all_future.year == 2026]) + 1)
            future_vals = coef[0] * future_t + coef[1]
            ax1.plot(future_dates, future_vals, linestyle="--", color="#F59E0B", linewidth=2.5, label="2026 예측")
        
        ax1.grid(alpha=0.3, linestyle='--', color='#CBD5E1')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#94A3B8')
        ax1.spines['bottom'].set_color('#94A3B8')
        ax1.tick_params(colors='#475569', labelsize=9)
        ax1.set_ylabel("매매량 (건)", color='#475569', fontweight='bold', fontsize=10)
        
        ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9)
        plt.xticks(rotation=45, ha='right')
        fig1.tight_layout()
        st.pyplot(fig1)
        
        st.markdown("""
        <div class="analysis-card" style="margin-top: 10px; min-height: 220px;">
            <div class="analysis-title">📊 거래량 회귀 예측 가이드</div>
            <div class="analysis-content">
                과거 매매량 추이(<span style="color:#1D4ED8; font-weight:bold;">파란 실선</span>)를 바탕으로 선형 회귀 알고리즘이 <b>2026년 매수 심리(<span style="color:#F59E0B; font-weight:bold;">노란 점선</span>)</b>를 예측합니다.<br><br>
                <span style="font-size:0.85rem; color:#64748B;">※ 점선이 우상향할 경우, 향후 시장의 매수 활력이 회복될 가능성이 높음을 시사합니다.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with colB:
    st.markdown("#### 2. 거시 지표 상관관계 (Heatmap)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor('white')
    cols = ["거래량", "평균매매가격(천만원)", "매매지수", "기준금리(%)"]
    tmp_corr = filtered[cols].dropna()
    
    if len(tmp_corr) >= 3:
        corr = tmp_corr.corr()
        im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect='auto', alpha=0.9)
        
        ax2.set_xticks(range(len(cols)))
        ax2.set_yticks(range(len(cols)))
        ax2.set_xticklabels(cols, rotation=30, ha="right", fontsize=9, color='#475569')
        ax2.set_yticklabels(cols, fontsize=9, color='#475569')
        
        for edge, spine in ax2.spines.items():
            spine.set_visible(False)
        ax2.set_xticks(np.arange(corr.shape[1]+1)-.5, minor=True)
        ax2.set_yticks(np.arange(corr.shape[0]+1)-.5, minor=True)
        ax2.grid(which="minor", color="white", linestyle='-', linewidth=2.5)
        ax2.tick_params(which="minor", bottom=False, left=False)
        ax2.tick_params(axis='both', colors='#475569')
        
        for i in range(len(cols)):
            for j in range(len(cols)):
                text_col = "white" if abs(corr.iloc[i, j]) > 0.5 else "#1E293B"
                ax2.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color=text_col, fontweight='bold', fontsize=10)
                
        cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors='#475569')
        
        fig2.tight_layout()
        st.pyplot(fig2)
        
        st.markdown("""
        <div class="analysis-card" style="margin-top: 10px; min-height: 220px;">
            <div class="analysis-title">🔥 상관관계(Heatmap) 읽는 법</div>
            <div class="analysis-content">
                두 지표가 함께 움직이는지 반대로 움직이는지(-1 ~ 1)를 나타냅니다.
                <ul style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                    <li><span style="color:#DC2626; font-weight:bold;">빨간색(+):</span> 비례 (함께 상승/하락)</li>
                    <li><span style="color:#2563EB; font-weight:bold;">파란색(-):</span> 반비례 (반대로 상승/하락)</li>
                </ul>
                <span style="font-size:0.85rem; color:#64748B;">※ AI가 시장 방향성을 도출하는 데 팩트 기반 자료로 활용됩니다.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =================================================
# [STEP 4-2] 머신러닝 다중 모델 검증 (선형회귀 & 랜덤포레스트)
# =================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📊 머신러닝 다중 모델: 시장 핵심 동인(Driver) 추출")
st.caption("선택한 지역의 가격 변동에 대해 '기준금리(매크로)'와 '매매량(수요)' 중 어떤 요인이 실질적으로 더 강력한 영향을 미치는지 선형 회귀와 앙상블 모델로 교차 분석합니다.")

df_ml = filtered[["기준금리(%)", "거래량", "평균매매가격(천만원)"]].dropna()

if len(df_ml) > 5:
    X = df_ml[["기준금리(%)", "거래량"]]
    y = df_ml["평균매매가격(천만원)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. 선형 회귀 모델
    model_lr = LinearRegression()
    model_lr.fit(X_scaled, y)
    r2_score = model_lr.score(X_scaled, y)
    coef_rate = model_lr.coef_[0]
    coef_vol = model_lr.coef_[1]
    
    # 2. 랜덤 포레스트 모델 (변수 중요도 파악)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_scaled, y)
    importances = model_rf.feature_importances_

    # 3단 컬럼 구성
    col_ml1, col_ml2, col_ml3 = st.columns([1, 1, 1.2]) 
    
    with col_ml1:
        st.markdown("#### 1. 회귀계수 (영향의 방향)")
        fig_ml, ax_ml = plt.subplots(figsize=(5, 5))
        fig_ml.patch.set_facecolor('white')
        
        features = ['기준금리', '매매량']
        coefs = [coef_rate, coef_vol]
        colors = ['#F87171' if c < 0 else '#1D4ED8' for c in coefs]
        
        y_pos = np.arange(len(features))
        ax_ml.barh(y_pos, coefs, color=colors, alpha=0.9, height=0.4, edgecolor='#475569', linewidth=1)
        ax_ml.axvline(0, color='#475569', linewidth=1.5, linestyle='-')
        
        max_abs = max(abs(coef_rate), abs(coef_vol))
        ax_ml.set_xlim(-max_abs * 1.6, max_abs * 1.6) 
        ax_ml.set_yticks(y_pos)
        ax_ml.set_yticklabels(features, fontsize=12, fontweight='bold', color='#1E293B')
        ax_ml.tick_params(axis='y', length=0) 

        for i, c in enumerate(coefs):
            text_x = c + (max_abs * 0.05) if c > 0 else c - (max_abs * 0.05)
            align = 'left' if c > 0 else 'right'
            label_color = '#1D4ED8' if c > 0 else '#F87171'
                
            text_obj = ax_ml.text(text_x, i, f"{c:.2f}", ha=align, va='center', fontweight='bold', fontsize=12, color=label_color)
            text_obj.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
        ax_ml.spines['top'].set_visible(False)
        ax_ml.spines['right'].set_visible(False)
        ax_ml.spines['left'].set_color('#94A3B8') 
        ax_ml.spines['bottom'].set_color('#94A3B8')
        fig_ml.tight_layout()
        st.pyplot(fig_ml)

    with col_ml2:
        st.markdown("#### 2. 모델 특성 중요도 (지분율)")
        fig_rf, ax_rf = plt.subplots(figsize=(5, 5))
        fig_rf.patch.set_facecolor('white')
        
        colors_rf = ['#F43F5E', '#3B82F6'] 
        
        # 💡 [핵심 수정] pctdistance=0.75로 텍스트를 도넛 빵 중앙으로 이동시킴
        wedges, texts, autotexts = ax_rf.pie(
            importances, labels=features, autopct='%1.1f%%', startangle=140, 
            colors=colors_rf, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
            textprops=dict(color="#1E293B", fontweight="bold", fontsize=12),
            pctdistance=0.75 
        )
        
        # 💡 [핵심 수정] 하얀 글씨가 무조건 잘 보이도록 네이비색 반투명 박스 적용
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_bbox(dict(facecolor='#1E293B', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))
        
        fig_rf.tight_layout()
        st.pyplot(fig_rf)
        
    with col_ml3:
        st.markdown("#### 💡 앙상블 머신러닝 해석 가이드")
        st.markdown(f"""
        <div class="analysis-card" style="min-height: 280px;">
            <div class="analysis-title">🔬 다중 알고리즘 교차 검증</div>
            <div class="analysis-content" style="font-size:0.9rem;">
                <b>1. 선형 회귀 (Linear Regression):</b> 금리가 오르면 가격이 하락(▼)하는 등의 <b>방향성</b>을 추적합니다. (현재 모델 신뢰도: {r2_score*100:.1f}%)<br><br>
                <b>2. 랜덤 포레스트 (Random Forest):</b> 비선형적인 관계까지 고려하여, 집값을 결정하는 데 두 요인이 각각 <b>몇 %의 기여도(지분)</b>를 가지는지 절대적 파워를 분석합니다.<br><br>
                <span style="color:#2563EB; font-weight:bold;">➡ 결론: 도넛 차트의 파이가 더 큰 변수가 이 지역 집값의 멱살을 잡고 끌고 가는 '진짜 대장(Driver)'입니다.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if importances[0] > importances[1]:
        stronger_feature_name = "기준금리(매크로)"
        stronger_feature_reason = "수급(거래량)의 변화보다 거시경제의 금리 인상/인하 충격에 훨씬 더 취약하고 민감하게 반응"
    else:
        stronger_feature_name = "거래량(지역 수급)"
        stronger_feature_reason = "외부 금리 충격보다는 해당 지역 내의 실질적인 매수세(거래 활성화) 여부에 따라 가격이 직접적으로 연동"

    st.markdown(f"""
    <div class="analysis-card" style="border-left: 5px solid #F59E0B; background-color: #FFFBEB; margin-top: 0.5rem;">
        <div class="analysis-title" style="color: #B45309;">📌 애널리스트 최종 결론</div>
        <div class="analysis-content" style="color: #92400E; font-weight:500; font-size: 1rem; line-height: 1.7;">
            두 가지 머신러닝 알고리즘을 종합 분석한 결과, 현재 <span class="highlight">{filtered['지역'].iloc[0]}</span> 지역의 집값은 
            <b>'{stronger_feature_name}'</b> 요인에 의해 지배적인 영향을 받고 있습니다. 
            이는 본 지역이 {stronger_feature_reason}하고 있음을 통계적으로 증명합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("📊 머신러닝 분석을 수행하기 위한 유효 데이터(최소 6개월 이상의 통합 데이터)가 부족합니다. 사이드바에서 분석 기간을 더 길게 설정해 주세요.")

# =================================================
# [STEP 5] OpenAI LLM 기반 부동산 전문가 리포트 생성
# =================================================
st.markdown("---")
st.markdown("### 🤖 실시간 AI 부동산 애널리스트 리포트")

if prompt := st.chat_input("질문을 입력하세요 (예: 경기지역 26년도 아파트 전망 어때?, 부산진구 롯데캐슬 라센트 집값 전망)"):
    with st.spinner("빅데이터 분석 및 GPT-5.2 추론을 진행 중입니다... 잠시만 기다려주세요."):
        lawd_cd, matched_area = detect_lawd_cd_from_question(prompt)
        target_region = detect_region_from_question(prompt) or resolve_macro_region_name(region_from_lawd_cd(lawd_cd)) or region
        
        macro_df = df[df["지역"] == target_region].sort_values("연월")
        recent_macro = macro_df.tail(24).copy()
        
        trade_context, apt_context = "", ""

        if MOLIT_SERVICE_KEY and lawd_cd:
            try:
                month_dfs = []
                for ym in recent_yms(MOLIT_MONTHS_BACK):
                    df_m = fetch_molit_month_allpages(lawd_cd, ym, MOLIT_SERVICE_KEY, MOLIT_NUM_ROWS, MOLIT_MAX_PAGES)
                    if not df_m.empty: month_dfs.append(df_m)

                trade_raw = pd.concat(month_dfs, ignore_index=True) if month_dfs else pd.DataFrame()
                trade = normalize_trade_df(trade_raw)

                if not trade.empty and "법정동" in trade.columns and matched_area:
                    if not str(matched_area).endswith(("구", "군", "시")):
                        key_base = matched_area[:-1] if matched_area.endswith(("동", "읍", "면", "리")) else matched_area
                        trade = trade[trade["법정동"].astype(str).str.contains(key_base, na=False)]

                if trade.empty:
                    trade_context = f"- 해당 기간/지역 내 거래 내역이 없습니다. (LAWD_CD: {lawd_cd})\n"
                else:
                    overall_sum = summarize_trade(trade)
                    trade_context = f"- 수집된 표본: {overall_sum.get('n')}건\n- 가격(억원): 평균 {overall_sum.get('avg_eok', 0):.2f} / 중앙값 {overall_sum.get('med_eok', 0):.2f}\n"

                    if "아파트" in trade.columns:
                        apt_names = trade["아파트"].dropna().unique()
                        p_bg = bigram_set(prompt)
                        scored = []
                        for name in apt_names:
                            n_bg = bigram_set(name)
                            if not n_bg: continue
                            score = 1.0 if (norm_txt(name) in norm_txt(prompt)) else (len(n_bg & p_bg) / max(len(n_bg), 1))
                            scored.append((name, score))
                        
                        top_match = sorted(scored, key=lambda x: x[1], reverse=True)[0] if scored else (None, 0.0)
                        if top_match[1] >= 0.45:
                            apt_sum = summarize_trade(trade[trade["아파트"] == top_match[0]])
                            apt_context = f"- 매칭 단지: {top_match[0]}\n- 단지 평균 매매가: {apt_sum.get('avg_eok', 0):.2f}억"
            except Exception as e:
                trade_context = f"실거래가 API 조회 실패: {e}"

        with st.expander("🛠️ 실시간 데이터 파이프라인 수집 로그", expanded=False):
            st.write(f"**타겟 지역:** {target_region} | **LAWD_CD:** {lawd_cd} ({matched_area})")
            st.text(trade_context + "\n" + apt_context)

        macro_table = recent_macro[["연월", "거래량", "평균매매가격(천만원)", "매매지수", "기준금리(%)"]].copy()
        macro_table["연월"] = macro_table["연월"].dt.strftime("%Y-%m")

        system_msg = (
            "당신은 15년 차 탑티어 부동산 리서치 애널리스트입니다. "
            "거시경제 지표(금리, 거래량, 가격지수)와 미시적 실거래 데이터를 종합하여 날카롭고 전문적인 시장 전망 리포트를 작성합니다. "
            "말투는 객관적이고 신뢰감 있는 전문가 톤을 유지하세요. "
            "사용자의 질문이 '특정 지역'에 대한 광범위한 전망인지, '특정 아파트 단지'에 대한 디테일한 전망인지 파악하여 유연하게 대답해야 합니다."
        )

        user_msg = f"""
[데이터 팩트]
- 지역: {target_region}
- 최근 매크로 요약:
{macro_table.to_string(index=False)}

[실거래 데이터 컨텍스트]
{trade_context}
{apt_context}

[사용자 질문] "{prompt}"

[출력 가이드라인]
1. 📊 [시장 현황 진단] : 매크로 데이터와 실거래 컨텍스트를 바탕으로 현재 시장 상황을 브리핑.
2. 🔮 [2026년 핵심 전망 및 시나리오] : 지역 전체 질문이면 지역의 방향성과 변동률(%), 단지 질문이면 단지의 가격 방어력과 상승 여력을 분석.
3. 💡 [투자 및 실거주 관점 인사이트] : 2026년에 취해야 할 전략적 포지션(매수/관망 등) 제시.
4. 📌 [One-Line Summary] : 2026년 최종 결론 한 줄 요약.
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.6 
            )
            st.success("데이터 파이프라인 분석 및 AI 추론이 완료되었습니다.")
            
            with st.container():
                st.markdown(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"GPT 연동 중 오류가 발생했습니다: {e}")