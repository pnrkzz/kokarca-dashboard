import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
from textwrap import dedent
import matplotlib.pyplot as plt
import geopandas as gpd

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Kahverengi Kokarca GeoVisual Analytics Dashboard",
    page_icon="🪲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# STYLE
# ==================================================
st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1550px;
}
.hero-box {
    border-radius: 22px;
    padding: 24px 28px 20px 28px;
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 55%, #e0f2fe 100%);
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.35rem;
    color: #1f2937;
}
.sub-title {
    font-size: 1.02rem;
    color: #4b5563;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.65rem;
    color: #1f2937;
}
.note-box {
    background: #f8fafc;
    border-left: 5px solid #2563eb;
    padding: 14px 16px;
    border-radius: 10px;
    color: #334155;
    margin-bottom: 1rem;
}
.small-note {
    font-size: 0.92rem;
    color: #475569;
    margin-top: 0.35rem;
}
.metric-card {
    background: white;
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 14px;
    padding: 10px 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.record-card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: white;
}
.record-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.3rem;
}
.record-meta {
    font-size: 0.88rem;
    color: #475569;
    margin-bottom: 0.4rem;
}
.record-summary {
    font-size: 0.93rem;
    color: #334155;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==================================================
# CONFIG
# ==================================================
CSV_PATH = r"C:\Users\depoc\kokarca_project\data_processed\MASTER_CORPUS_FINAL_ANALYSIS_READY.csv"
DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2025

CITY_COORDS = {
    "Adana": (37.0, 35.3213), "Adıyaman": (37.7648, 38.2786), "Afyonkarahisar": (38.7569, 30.5387),
    "Ağrı": (39.7191, 43.0503), "Aksaray": (38.3687, 34.037), "Amasya": (40.6539, 35.8331),
    "Ankara": (39.9334, 32.8597), "Antalya": (36.8969, 30.7133), "Ardahan": (41.1105, 42.7022),
    "Artvin": (41.1828, 41.8183), "Aydın": (37.845, 27.8396), "Balıkesir": (39.6484, 27.8826),
    "Bartın": (41.5811, 32.461), "Batman": (37.8812, 41.1351), "Bayburt": (40.2552, 40.2249),
    "Bilecik": (40.15, 29.9833), "Bingöl": (38.8847, 40.4983), "Bitlis": (38.3938, 42.1232),
    "Bolu": (40.7395, 31.6116), "Burdur": (37.7203, 30.2908), "Bursa": (40.1885, 29.061),
    "Çanakkale": (40.1467, 26.4086), "Çankırı": (40.6013, 33.6134), "Çorum": (40.5506, 34.9556),
    "Denizli": (37.7833, 29.0963), "Diyarbakır": (37.9144, 40.2306), "Düzce": (40.8438, 31.1565),
    "Edirne": (41.6771, 26.5557), "Elazığ": (38.6748, 39.2232), "Erzincan": (39.75, 39.5),
    "Erzurum": (39.9043, 41.2679), "Eskişehir": (39.7767, 30.5206), "Gaziantep": (37.0662, 37.3833),
    "Giresun": (40.9128, 38.3895), "Gümüşhane": (40.4386, 39.5086), "Hakkari": (37.5744, 43.7408),
    "Hatay": (36.4018, 36.3498), "Iğdır": (39.9167, 44.0333), "Isparta": (37.7648, 30.5566),
    "İstanbul": (41.0082, 28.9784), "İzmir": (38.4237, 27.1428), "Kahramanmaraş": (37.5736, 36.9371),
    "Karabük": (41.2061, 32.6204), "Karaman": (37.1811, 33.215), "Kars": (40.6013, 43.0975),
    "Kastamonu": (41.3781, 33.7753), "Kayseri": (38.7312, 35.4787), "Kırıkkale": (39.8468, 33.5153),
    "Kırklareli": (41.7333, 27.2167), "Kırşehir": (39.1425, 34.1709), "Kilis": (36.7184, 37.1212),
    "Kocaeli": (40.7654, 29.9408), "Konya": (37.8746, 32.4932), "Kütahya": (39.4167, 29.9833),
    "Malatya": (38.3552, 38.3095), "Manisa": (38.6191, 27.4289), "Mardin": (37.3122, 40.7351),
    "Mersin": (36.8, 34.6333), "Muğla": (37.2153, 28.3636), "Muş": (38.9462, 41.7539),
    "Nevşehir": (38.6247, 34.724), "Niğde": (37.9667, 34.6833), "Ordu": (40.9862, 37.8797),
    "Osmaniye": (37.0742, 36.2478), "Rize": (41.0201, 40.5234), "Sakarya": (40.7569, 30.3781),
    "Samsun": (41.2867, 36.33), "Siirt": (37.9333, 41.95), "Sinop": (42.0268, 35.1511),
    "Sivas": (39.7477, 37.0179), "Şanlıurfa": (37.1674, 38.7955), "Şırnak": (37.4187, 42.4918),
    "Tekirdağ": (40.978, 27.511), "Tokat": (40.3167, 36.55), "Trabzon": (41.0053, 39.7269),
    "Tunceli": (39.1081, 39.5483), "Uşak": (38.6823, 29.4082), "Van": (38.4942, 43.38),
    "Yalova": (40.655, 29.2769), "Yozgat": (39.82, 34.8086), "Zonguldak": (41.4564, 31.7987),
}

SOURCE_ICON = {
    "news": "📰",
    "youtube": "▶️",
    "academic": "🎓",
    "official": "🏛️",
}

GPT_SYSTEM_PROMPT = dedent(
    """
    Sen Türkiye’de kahverengi kokarca (Halyomorpha halys) yayılımı üzerine çalışan bir araştırma asistanısın.
    Görevin, haber, YouTube transkripti, akademik metin ve resmi raporlardan yapılandırılmış bilgi çıkarmaktır.

    Önceliğin şunları belirlemektir:
    1. Yer bilgisi (il, ilçe, mahalle, köy düzeyi)
    2. Zaman bilgisi (yıl, tarih, dönem)
    3. Etkilenen ürünler (özellikle fındık)
    4. Olay türü
    5. Devlet/kurum müdahaleleri
    6. İklim sinyalleri ve bunların yayılımla ilişkisi
    7. Zaman içindeki yön bilgisi (artış/azalış/sabit/belirsiz)
    8. Kısa Türkçe özet
    """
).strip()

# ==================================================
# HELPERS
# ==================================================
def parse_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    x = str(x).strip()
    if not x:
        return []
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return [str(i).strip() for i in val if str(i).strip()]
        return [str(val).strip()]
    except Exception:
        return [i.strip() for i in x.split(",") if i.strip()]


def has_content(x):
    return int(isinstance(x, list) and len(x) > 0)


def gov_cat(lst):
    text = " ".join(lst).lower() if isinstance(lst, list) else str(lst).lower()
    if "biyolojik" in text:
        return "biyolojik"
    if "kimyasal" in text or "ilaç" in text:
        return "kimyasal"
    if "izleme" in text or "tarama" in text or "tuzak" in text:
        return "izleme"
    if "eğitim" in text or "bilgilendirme" in text or "farkındalık" in text or "toplantı" in text:
        return "farkındalık/koordinasyon"
    if not text.strip():
        return "yok"
    return "diğer"


def climate_cat(lst):
    text = " ".join(lst).lower() if isinstance(lst, list) else str(lst).lower()
    if "kurak" in text or "sıcak" in text or "ısı" in text or "ılıman" in text:
        return "sıcaklık/kuraklık"
    if "yağış" in text or "nem" in text or "rutubet" in text:
        return "yağış/nem"
    if "iklim" in text or "don" in text:
        return "genel iklim"
    if not text.strip():
        return "yok"
    return "diğer"


def lag_group_func(v):
    if pd.isna(v):
        return "belirsiz"
    if v <= 0:
        return "anlık"
    if v <= 2:
        return "kısa gecikme"
    if v <= 5:
        return "orta gecikme"
    return "uzun gecikme"


def build_period(year):
    if pd.isna(year):
        return "Belirsiz"
    year = int(year)
    if 2016 <= year <= 2018:
        return "2016–2018"
    if 2019 <= year <= 2021:
        return "2019–2021"
    if 2022 <= year <= 2023:
        return "2022–2023"
    if 2024 <= year <= 2025:
        return "2024–2025"
    return str(year)


def add_location_columns(df, location_col="locations_raw"):
    out = df.copy().explode(location_col)
    out["location_clean"] = out[location_col].astype(str).str.strip()
    out = out[
        out["location_clean"].notna()
        & (out["location_clean"] != "")
        & (out["location_clean"].str.lower() != "none")
        & (out["location_clean"].isin(CITY_COORDS.keys()))
    ].copy()
    out["lat"] = out["location_clean"].map(lambda x: CITY_COORDS[x][0])
    out["lon"] = out["location_clean"].map(lambda x: CITY_COORDS[x][1])
    return out


@st.cache_data
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)

    for col in [
        "locations_raw",
        "crops_raw",
        "government_action_raw",
        "climate_signal_raw",
        "event_type_raw",
        "institution_raw",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    df["event_year"] = pd.to_numeric(df.get("event_year"), errors="coerce")
    df["published_dt"] = pd.to_datetime(df.get("published_dt"), errors="coerce")
    df["published_year"] = df["published_dt"].dt.year
    df["source_type"] = df.get("source_type", "unknown").astype(str).str.strip().str.lower()
    df["summary_tr"] = df.get("summary_tr", "").fillna("")
    df["title"] = df.get("title", "Başlıksız kayıt").fillna("Başlıksız kayıt")
    df["url"] = df.get("url", "").fillna("")

    # çalışma çerçevesi
    df = df[
        (df["event_year"].between(DEFAULT_START_YEAR, DEFAULT_END_YEAR, inclusive="both"))
        | (df["published_year"].between(DEFAULT_START_YEAR, DEFAULT_END_YEAR, inclusive="both"))
    ].copy()

    df["has_gov"] = df["government_action_raw"].apply(has_content)
    df["has_climate"] = df["climate_signal_raw"].apply(has_content)
    df["gov_cat"] = df["government_action_raw"].apply(gov_cat)
    df["climate_cat"] = df["climate_signal_raw"].apply(climate_cat)
    df["lag"] = df["published_year"] - df["event_year"]
    df["lag_group"] = df["lag"].apply(lag_group_func)

    df_city = add_location_columns(df)

    df_safe = df.copy()
    df_safe["crop_count"] = df_safe["crops_raw"].apply(len)
    df_safe["loc_count"] = df_safe["locations_raw"].apply(len)
    df_safe_product = df_safe[df_safe["crop_count"] == 1].copy()
    df_safe_product["single_crop"] = df_safe_product["crops_raw"].apply(
        lambda x: str(x[0]).strip().lower() if len(x) > 0 else None
    )
    df_safe_product = add_location_columns(df_safe_product)
    df_safe_product["has_gov"] = df_safe_product["government_action_raw"].apply(has_content)
    df_safe_product["has_climate"] = df_safe_product["climate_signal_raw"].apply(has_content)
    df_safe_product["gov_cat"] = df_safe_product["government_action_raw"].apply(gov_cat)
    df_safe_product["climate_cat"] = df_safe_product["climate_signal_raw"].apply(climate_cat)

    return df, df_city, df_safe_product


# ==================================================
# LOAD DATA
# ==================================================
try:
    df_base, df_city, df_safe_product = load_and_prepare(CSV_PATH)
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.markdown("## Filtreler")

all_sources = sorted(df_base["source_type"].dropna().unique().tolist())
selected_sources = st.sidebar.multiselect("Kaynak türü", all_sources, default=all_sources)

selected_years = st.sidebar.slider(
    "Yıl aralığı",
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR,
    (DEFAULT_START_YEAR, DEFAULT_END_YEAR),
)

time_basis = st.sidebar.radio("Zaman temeli", ["Olay yılı", "Yayın yılı"], index=0)
use_safe = st.sidebar.checkbox("Sadece güvenli ürün verisi", value=False)

city_options = sorted(df_city["location_clean"].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("İl seçimi", options=["Tümü"] + city_options, index=0)

product_options = sorted(df_safe_product["single_crop"].dropna().unique().tolist())
default_products = ["fındık"] if "fındık" in product_options else product_options[:1]
selected_products = st.sidebar.multiselect("Ürün", options=product_options, default=default_products)

filter_gov = st.sidebar.selectbox("Devlet müdahalesi", ["Tümü", "Var", "Yok"], index=0)
filter_climate = st.sidebar.selectbox("İklim sinyali", ["Tümü", "Var", "Yok"], index=0)

# ==================================================
# ACTIVE DATAFRAME
# ==================================================
active_df = df_safe_product.copy() if use_safe else df_city.copy()
active_df = active_df[active_df["source_type"].isin(selected_sources)].copy()

year_col = "event_year" if time_basis == "Olay yılı" else "published_year"
active_df = active_df[active_df[year_col].between(selected_years[0], selected_years[1], inclusive="both")].copy()

if selected_city != "Tümü":
    active_df = active_df[active_df["location_clean"] == selected_city].copy()

if use_safe and selected_products:
    active_df = active_df[active_df["single_crop"].isin(selected_products)].copy()

if filter_gov == "Var":
    active_df = active_df[active_df["has_gov"] == 1].copy()
elif filter_gov == "Yok":
    active_df = active_df[active_df["has_gov"] == 0].copy()

if filter_climate == "Var":
    active_df = active_df[active_df["has_climate"] == 1].copy()
elif filter_climate == "Yok":
    active_df = active_df[active_df["has_climate"] == 0].copy()

# ==================================================
# HERO
# ==================================================
st.markdown(
    """
<div class="hero-box">
    <div class="main-title">Kahverengi Kokarca GeoVisual Analytics Dashboard</div>
    <div class="sub-title">
        Bu dashboard, haberler, YouTube içerikleri, akademik yayınlar ve resmi belgelerde raporlanan kahverengi kokarca olaylarının
        mekânsal-zamansal örüntülerini göstermektedir.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="note-box">
Bu görselleştirmeler doğrudan saha gözlemlerini değil, metinlerde raporlanan olayların GPT tabanlı bilgi çıkarımı ile yapılandırılmış hale getirilmiş temsillerini sunmaktadır.
Ana odak; yayılım, zaman, mekân, müdahale, ürün etkisi ve kaynak örüntülerini analiz etmektir.
</div>
""",
    unsafe_allow_html=True,
)

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Genel Bakış",
    "Yayılım Analizi",
    "Kaynak Explorer",
    "Akademik",
    "Müdahale & İklim",
    "STC",
    "Metodoloji",
    "Etki Haritası",
])
# ==================================================
# TAB 1 - GENEL BAKIŞ
# ==================================================
with tab1:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Toplam İl-Kayıt", f"{len(active_df):,}")
    k2.metric("Farklı İl", f"{active_df['location_clean'].nunique():,}")
    k3.metric("Kaynak Türü", f"{active_df['source_type'].nunique():,}")
    k4.metric("Müdahale İçeren", f"{int(active_df['has_gov'].sum()):,}")
    k5.metric("İklim İçeren", f"{int(active_df['has_climate'].sum()):,}")
    academic_count = int(df_base[df_base["source_type"] == "academic"].shape[0])
    k6.metric("Akademik Kayıt", f"{academic_count:,}")

    c1, c2 = st.columns([1.15, 1])

    with c1:
        ts = (
            active_df.groupby(year_col)
            .size()
            .reset_index(name="count")
            .sort_values(year_col)
        )
        if len(ts) > 0:
            fig_ts = px.line(
                ts,
                x=year_col,
                y="count",
                markers=True,
                title=f"Zaman Serisi ({time_basis})",
                labels={year_col: time_basis, "count": "Kayıt Sayısı"},
            )
            fig_ts.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Zaman serisi üretilemedi.")

    with c2:
        src_counts = (
            active_df.groupby("source_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        if len(src_counts) > 0:
            fig_src = px.bar(
                src_counts,
                x="source_type",
                y="count",
                text="count",
                title="Kaynak Türü Dağılımı",
                labels={"source_type": "Kaynak Türü", "count": "Kayıt Sayısı"},
            )
            fig_src.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_src, use_container_width=True)
        else:
            st.info("Kaynak dağılımı üretilemedi.")

    map_counts = (
        active_df.groupby(["location_clean", "lat", "lon"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    if len(map_counts) > 0:
        fig_map = px.scatter_geo(
            map_counts,
            lat="lat",
            lon="lon",
            size="count",
            color="count",
            hover_name="location_clean",
            title="Genel Mekânsal Yoğunluk",
            size_max=38,
            projection="natural earth",
        )
        fig_map.update_geos(
            center=dict(lat=39, lon=35),
            projection_scale=6.2,
            showcountries=True,
            showland=True,
            landcolor="rgb(245,245,245)",
        )
        fig_map.update_layout(height=700, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Genel mekânsal yoğunluk haritası üretilemedi.")

# ==================================================
# TAB 2 - YAYILIM ANALİZİ
# ==================================================
with tab2:
    st.markdown('<div class="section-title">Animasyonlu Yayılım Haritası</div>', unsafe_allow_html=True)

    yearly_counts = (
        active_df.groupby([year_col, "location_clean", "lat", "lon"])
        .size()
        .reset_index(name="count")
    )

    frames = []
    years = sorted(yearly_counts[year_col].dropna().astype(int).unique().tolist())
    for y in years:
        temp = yearly_counts[yearly_counts[year_col] <= y].copy()
        temp = temp.groupby(["location_clean", "lat", "lon"], as_index=False)["count"].sum()
        temp["frame_year"] = int(y)
        frames.append(temp)

    if frames:
        anim_df = pd.concat(frames, ignore_index=True)
        anim_df["size_scaled"] = np.log1p(anim_df["count"]) * 10

        fig_anim = px.scatter_geo(
            anim_df,
            lat="lat",
            lon="lon",
            size="size_scaled",
            color="count",
            hover_name="location_clean",
            hover_data={"frame_year": True, "count": True, "size_scaled": False, "lat": False, "lon": False},
            animation_frame="frame_year",
            projection="natural earth",
            title=f"Zamansal İlerleyiş ({time_basis})",
            size_max=44,
        )
        fig_anim.update_geos(
            center=dict(lat=39, lon=35),
            projection_scale=6.3,
            showcountries=True,
            showland=True,
            landcolor="rgb(245,245,245)",
        )
        fig_anim.update_layout(height=820, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info("Animasyonlu yayılım haritası üretilemedi.")

    st.markdown('<div class="section-title">Dönemsel Kesitler</div>', unsafe_allow_html=True)
    slice_core = yearly_counts.copy()
    slice_core["period"] = slice_core[year_col].apply(build_period)
    slice_core = (
        slice_core.groupby(["period", "location_clean", "lat", "lon"], as_index=False)["count"]
        .sum()
    )

    if len(slice_core) > 0:
        fig_slice = px.scatter(
            slice_core,
            x="lon",
            y="lat",
            size="count",
            color="count",
            facet_col="period",
            hover_name="location_clean",
            title="Dönemsel Kesitler: Kahverengi Kokarca Yayılımı",
            labels={"lon": "Boylam", "lat": "Enlem"},
            size_max=35,
            range_color=[0, slice_core["count"].max()],
        )
        fig_slice.update_layout(width=1250, height=460)
        fig_slice.update_xaxes(showgrid=False)
        fig_slice.update_yaxes(showgrid=False)
        st.plotly_chart(fig_slice, use_container_width=True)
    else:
        st.info("Dönemsel kesitler üretilemedi.")

# ==================================================
# TAB 3 - KAYNAK EXPLORER
# ==================================================
with tab3:
    st.markdown('<div class="section-title">İkonlu Kaynak Yoğunluğu</div>', unsafe_allow_html=True)

    explorer_group = (
        active_df.groupby(["source_type", "location_clean", "lat", "lon"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    if len(explorer_group) > 0:
        fig_explorer = px.scatter_geo(
            explorer_group,
            lat="lat",
            lon="lon",
            size="count",
            color="source_type",
            hover_name="location_clean",
            hover_data={"source_type": True, "count": True, "lat": False, "lon": False},
            title="Kaynak Türüne Göre Mekânsal Dağılım",
            size_max=38,
            projection="natural earth",
        )
        fig_explorer.update_geos(
            center=dict(lat=39, lon=35),
            projection_scale=6.3,
            showcountries=True,
            showland=True,
            landcolor="rgb(245,245,245)",
        )
        fig_explorer.update_layout(height=700, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_explorer, use_container_width=True)
    else:
        st.info("Kaynak explorer haritası üretilemedi.")

    st.markdown('<div class="section-title">İncelenebilir Kayıtlar</div>', unsafe_allow_html=True)

    display_df = active_df.copy().sort_values([year_col, "location_clean"], ascending=[False, True])
    display_df = display_df.drop_duplicates(subset=["title", "location_clean", year_col, "source_type"])

    selected_record_city = st.selectbox(
        "Kayıt kartları için il seç",
        options=["Tümü"] + sorted(display_df["location_clean"].dropna().unique().tolist()),
        index=0,
        key="record_city_select"
    )

    if selected_record_city != "Tümü":
        display_df = display_df[display_df["location_clean"] == selected_record_city].copy()

    max_cards = st.slider("Gösterilecek kayıt kartı sayısı", 5, 50, 12, key="record_card_slider")
    card_df = display_df.head(max_cards)

    if len(card_df) == 0:
        st.info("Bu filtrelerle gösterilecek kayıt bulunamadı.")
    else:
        for _, row in card_df.iterrows():
            icon = SOURCE_ICON.get(row["source_type"], "📄")
            url_html = f'<a href="{row["url"]}" target="_blank">Kaynağı aç</a>' if str(row["url"]).strip() else "Bağlantı yok"
            gov_text = "Var" if row.get("has_gov", 0) == 1 else "Yok"
            clim_text = "Var" if row.get("has_climate", 0) == 1 else "Yok"
            st.markdown(
                f"""
                <div class="record-card">
                    <div class="record-title">{icon} {row['title']}</div>
                    <div class="record-meta">
                        İl: {row.get('location_clean', '')} | {time_basis}: {int(row[year_col]) if pd.notna(row[year_col]) else 'Belirsiz'} |
                        Kaynak: {row['source_type']} | Müdahale: {gov_text} | İklim: {clim_text}
                    </div>
                    <div class="record-summary">{row.get('summary_tr', '')}</div>
                    <div class="small-note">{url_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==================================================
# TAB 4 - AKADEMİK
# ==================================================
with tab4:
    st.markdown('<div class="section-title">Akademik Panel</div>', unsafe_allow_html=True)
    academic_df = df_base[df_base["source_type"] == "academic"].copy()
    academic_df = academic_df[
        academic_df["published_year"].between(selected_years[0], selected_years[1], inclusive="both")
        | academic_df["event_year"].between(selected_years[0], selected_years[1], inclusive="both")
    ].copy()

    a1, a2 = st.columns([1, 1.4])
    with a1:
        academic_year = (
            academic_df.groupby("published_year")
            .size()
            .reset_index(name="count")
            .dropna()
        )
        if len(academic_year) > 0:
            fig_ac = px.bar(
                academic_year,
                x="published_year",
                y="count",
                title="Akademik Kayıtların Yıllara Göre Dağılımı",
                text="count",
            )
            fig_ac.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_ac, use_container_width=True)
        else:
            st.info("Akademik yıl dağılımı üretilemedi.")

    with a2:
        preview_cols = [c for c in ["title", "published_year", "summary_tr", "url"] if c in academic_df.columns]
        if len(academic_df) > 0 and preview_cols:
            st.dataframe(
                academic_df[preview_cols].drop_duplicates().sort_values("published_year", ascending=False),
                use_container_width=True,
                height=430,
            )
        else:
            st.info("Akademik kayıt önizlemesi üretilemedi.")

# ==================================================
# TAB 5 - MÜDAHALE & İKLİM
# ==================================================
with tab5:
    c1, c2 = st.columns(2)

    with c1:
        gov_year = (
            active_df[active_df["has_gov"] == 1]
            .groupby([year_col, "gov_cat"])
            .size()
            .reset_index(name="count")
        )
        if len(gov_year) > 0:
            fig_gov = px.bar(
                gov_year,
                x=year_col,
                y="count",
                color="gov_cat",
                barmode="stack",
                title="Devlet Müdahalesi Türleri",
                labels={year_col: time_basis, "count": "Kayıt Sayısı", "gov_cat": "Müdahale Türü"},
            )
            fig_gov.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_gov, use_container_width=True)
        else:
            st.info("Müdahale grafiği üretilemedi.")

    with c2:
        clim_year = (
            active_df[active_df["has_climate"] == 1]
            .groupby([year_col, "climate_cat"])
            .size()
            .reset_index(name="count")
        )
        if len(clim_year) > 0:
            fig_clim = px.bar(
                clim_year,
                x=year_col,
                y="count",
                color="climate_cat",
                barmode="stack",
                title="İklim Sinyalleri",
                labels={year_col: time_basis, "count": "Kayıt Sayısı", "climate_cat": "İklim Kategorisi"},
            )
            fig_clim.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_clim, use_container_width=True)
        else:
            st.info("İklim grafiği üretilemedi.")

    m_map = (
        active_df[(active_df["has_gov"] == 1) | (active_df["has_climate"] == 1)]
        .groupby(["location_clean", "lat", "lon"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if len(m_map) > 0:
        fig_mix_map = px.scatter_geo(
            m_map,
            lat="lat",
            lon="lon",
            size="count",
            color="count",
            hover_name="location_clean",
            title="Müdahale / İklim İçeren Yerler",
            size_max=40,
            projection="natural earth",
        )
        fig_mix_map.update_geos(
            center=dict(lat=39, lon=35),
            projection_scale=6.2,
            showcountries=True,
            showland=True,
        )
        fig_mix_map.update_layout(height=650, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_mix_map, use_container_width=True)
    else:
        st.info("Müdahale/iklim haritası üretilemedi.")

# ==================================================
# TAB 6 - STC
# ==================================================
with tab6:
    st.markdown('<div class="section-title">Heatmap</div>', unsafe_allow_html=True)
    stc_source = active_df.copy()
    stc = (
        stc_source.groupby(["location_clean", year_col])
        .size()
        .reset_index(name="count")
        .dropna()
    )

    if len(stc) > 0:
        pivot = stc.pivot_table(index="location_clean", columns=year_col, values="count", fill_value=0)
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            labels=dict(x=time_basis, y="İl", color="Yoğunluk"),
            title="İl x Yıl Yoğunluk Matrisi",
        )
        fig_heat.update_layout(height=700, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Heatmap üretilemedi.")

    st.markdown('<div class="section-title">3B STC</div>', unsafe_allow_html=True)
    if len(stc) > 0:
        loc_order = (
            stc.groupby("location_clean")["count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        stc["loc_index"] = stc["location_clean"].map({loc: i for i, loc in enumerate(loc_order)})

        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(
            x=stc["loc_index"],
            y=np.zeros(len(stc)),
            z=stc[year_col],
            mode="markers",
            marker=dict(
                size=np.log1p(stc["count"]) * 8,
                color=stc["count"],
                colorscale="Viridis",
                opacity=0.85,
                colorbar=dict(title="Kayıt"),
            ),
            text=(
                "İl: " + stc["location_clean"].astype(str)
                + "<br>Yıl: " + stc[year_col].astype(int).astype(str)
                + "<br>Kayıt: " + stc["count"].astype(str)
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
        fig_3d.update_layout(
            title="3B STC Görünümü",
            height=850,
            margin=dict(l=0, r=0, t=60, b=0),
            scene=dict(
                xaxis=dict(title="İller", tickmode="array", tickvals=list(range(len(loc_order))), ticktext=loc_order),
                yaxis=dict(title="", showticklabels=False),
                zaxis=dict(title=time_basis, dtick=1),
                aspectmode="manual",
                aspectratio=dict(x=2.4, y=0.7, z=2.0),
                camera=dict(eye=dict(x=1.8, y=1.2, z=1.4)),
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("3B STC üretilemedi.")

# ==================================================
# TAB 7 - METODOLOJİ
# ==================================================
with tab7:
    st.markdown('<div class="section-title">Veri ve Yöntem</div>', unsafe_allow_html=True)
    st.markdown(
        """
Bu dashboard, haber, YouTube transkripti, akademik yayın ve resmi belge metinlerinden yapılandırılmış bilgi çıkarımı ile üretilmiştir.
Gösterilen mekânsal-zamansal örüntüler, doğrudan saha gözlemi değil; metinlerde raporlanan olayların analitik temsilleridir.
"""
    )

    st.markdown('<div class="section-title">Çıkarılan Başlıca Alanlar</div>', unsafe_allow_html=True)
    st.markdown(
        """
- Yer bilgisi (il, ilçe, mahalle, köy)
- Zaman bilgisi (olay yılı / tarih ifadesi)
- Etkilenen ürünler
- Olay türü
- Devlet/kurum müdahalesi
- İklim sinyalleri
- Değişim yönü
- Kısa özet
"""
    )

    with st.expander("GPT Analiz Promptu"):
        st.code(GPT_SYSTEM_PROMPT, language="text")

    with st.expander("Sınırlılıklar"):
        st.markdown(
            """
- Dashboard doğrudan saha sayımı veya resmi istatistik verisi değildir.
- Metinlerde raporlanmayan olaylar görünmez kalabilir.
- Lokasyon ve zaman bilgisi, yalnızca metinde açıkça geçen ifadelere bağlıdır.
- Bazı kayıtlarda olay yılı ile yayın yılı farklı olabilir.
- Kaynak yoğunluğu, gerçek biyolojik yoğunluğu birebir temsil etmeyebilir.
"""
        )

# ==================================================
# TAB 8 - Müdahale Sonrası Yayılım Değişimi
# ==================================================

with tab8:
    st.markdown('<div class="section-title">Fındık: Müdahale Sonrası Yayılım Değişimi</div>', unsafe_allow_html=True)
    st.info("Bu harita, sadece fındık içeren kayıtlarda müdahale öncesi ve sonrası yayılım değişimini göstermektedir.")

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        # =========================
        # 1. HARİTA DOSYASI
        # =========================
        turkey = gpd.read_file("data/turkey.geojson")

        # =========================
        # 2. ETKİ VERİSİ
        # =========================
        map_df = impact_pivot.reset_index()[["location_clean", "change"]].copy()

        # =========================
        # 3. İSİM NORMALİZASYONU
        # =========================
        def normalize_name(s):
            if pd.isna(s):
                return None

            s = str(s).strip()

            fixes = {
                "IÄdir": "Iğdır",
                "Sirnak": "Şırnak",
                "Corum": "Çorum",
                "Kirsehir": "Kırşehir",
                "Kutahya": "Kütahya",
                "Mugla": "Muğla",
                "Sanliurfa": "Şanlıurfa",
                "Usak": "Uşak",
                "Nevsehir": "Nevşehir",
                "Igdir": "Iğdır",
                "Izmir": "İzmir",
                "Istanbul": "İstanbul"
            }

            return fixes.get(s, s)

        turkey["name_clean"] = turkey["name"].apply(normalize_name)
        map_df["location_clean2"] = map_df["location_clean"].apply(normalize_name)

        # =========================
        # 4. JOIN
        # =========================
        merged = turkey.merge(
            map_df,
            left_on="name_clean",
            right_on="location_clean2",
            how="left"
        )

        # =========================
        # 5. HARİTA
        # =========================
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        merged.plot(
            column="change",
            cmap="RdYlGn_r",
            linewidth=0.5,
            edgecolor="black",
            legend=True,
            ax=ax,
            missing_kwds={"color": "lightgrey", "label": "Veri yok"}
        )

        ax.set_title("Fındık: Müdahale Sonrası Yayılım Değişimi")
        ax.axis("off")

        st.pyplot(fig)

        # =========================
        # 6. AÇIKLAMA
        # =========================
        st.markdown("""
        **Yorum:**
        - 🟢 Yeşil → yayılım azalmış (müdahale etkili olabilir)  
        - 🔴 Kırmızı → yayılım artmış (müdahale yetersiz)  
        - ⚪ Gri → veri yok  

        Bu harita, müdahale etkinliğinin mekânsal olarak homojen olmadığını göstermektedir.
        """)

    except Exception as e:
        st.error(f"Harita yüklenemedi: {e}")
