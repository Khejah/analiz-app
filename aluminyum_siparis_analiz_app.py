from typing import Iterable, Optional

import gradio as gr
import pandas as pd
import plotly.express as px
import os
import hashlib
import json
import re
import unicodedata
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

CACHE_DIR = "/tmp/cache_data"
REPORT_DIR = "/tmp/generated_reports"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

COLUMN_ALIASES = {
    "tarih": ["Tarih", "TARIH", "date", "tarih_"],
    "profil": ["Profil No", "Profil", "profil", "profil_kodu", "profil_no"],
    "siparis_no": ["Siparis No", "Sipariş No", "siparis_no", "order_no", "sip_no"],
    "musteri": ["Mus.Siparis No", "Müşteri Sipariş No", "Musteri", "Müşteri", "mus_siparis_no", "musteri_siparis_no"],
    "adet": ["Adet", "adet", "boy_adedi"],
    "kg": ["Kg", "kg", "kilogram"],
    "firma": ["Firma Adi", "Firma", "firma_adi", "firma_adi_"],
    "pres": ["Pres Adi", "Pres", "pres", "pres_adi"],
    "termin": ["Termin", "termin_tarihi"],
    "termin_hafta": ["Termin Hafta", "termin_hafta"],
}

UI_COLS = {
    "profil": "Profil Kodu",
    "toplam_kayit": "Toplam Kayıt Sayısı",
    "farkli_siparis": "Farklı Sipariş Sayısı",
    "toplam_uretim": "Toplam Üretim (Boy)",
    "ilk_tarih": "İlk Görülen Sipariş Tarihi",
    "son_tarih": "Son Görülen Sipariş Tarihi",
    "siparis_ort": "Sipariş Başına Ortalama Üretim",
    "yillik_tuketim": "Yıllık Ortalama Tüketim",
    "onerilen_parti": "Önerilen Stok / Parti Boyu",
    "kumulatif_pay": "Toplam İçindeki Kümülatif Pay (%)",
    "stok_karari": "Stok Kararı",
    "abc_sinifi": "ABC Sınıfı",
    "toplam_kg": "Toplam Kg",
    "siparis_sayisi": "Sipariş Sayısı",
}

def normalize_col(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("ı", "i").replace("İ", "i").replace("I", "i")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")
    
def get_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
    
NORMALIZED_ALIAS_MAP = {
    key: {normalize_col(x) for x in vals} for key, vals in COLUMN_ALIASES.items()
}


def detect_header_row(excel_path, sheet_name: Optional[str] = None, max_rows: int = 15) -> int:
    preview = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        nrows=max_rows
    )
    preview = preview.fillna("")

    best_idx = 0
    best_score = -1
    for idx in range(len(preview)):
        row = [normalize_col(x) for x in preview.iloc[idx].tolist() if pd.notna(x)]
        score = 0
        rowset = set(row)
        for aliases in NORMALIZED_ALIAS_MAP.values():
            if rowset & aliases:
                score += 1
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def find_column(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    aliases = NORMALIZED_ALIAS_MAP[logical_name]
    normalized_columns = {col: normalize_col(col) for col in df.columns}

    for col, norm in normalized_columns.items():
        if norm in aliases:
            return col

    return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_FILE = os.path.join(BASE_DIR, "musteri_mapping.json")

def load_customer_mapping():
    if not os.path.exists(MAPPING_FILE):
        return {}

    try:
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    cleaned = {}
    for ana_musteri, altlar in data.items():
        key = str(ana_musteri).strip().upper()
        if not key:
            continue

        if isinstance(altlar, list):
            cleaned[key] = [str(x).strip().upper() for x in altlar if str(x).strip()]
        else:
            cleaned[key] = []

    return cleaned

def normalize_musteri_value(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def get_customer_group_list(mapping: dict, ana_musteri: str):
    if not ana_musteri:
        return []

    altlar = mapping.get(ana_musteri, [])
    tum_liste = [ana_musteri] + altlar

    benzersiz = []
    seen = set()
    for item in tum_liste:
        val = str(item).strip().upper()
        if val and val not in seen:
            seen.add(val)
            benzersiz.append(val)

    return benzersiz
    
def load_excel(excel_file) -> pd.DataFrame:
    if excel_file is None:
        raise ValueError("Lütfen bir Excel dosyası yükleyin.")

    if hasattr(excel_file, "name") and os.path.exists(excel_file.name):
        excel_path = excel_file.name
    else:
        excel_path = str(excel_file)
        if not os.path.exists(excel_path):
            raise ValueError("Dosya yolu alınamadı")

    file_hash = get_file_hash(excel_path)
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")
    parquet_path = os.path.join(CACHE_DIR, f"{file_hash}.parquet")

    # 🚀 ULTRA FAST PARQUET CACHE
    if os.path.exists(parquet_path):
        try:
            df_fast = pd.read_parquet(parquet_path)
            required_cols = {"tarih", "profil", "siparis_no", "adet", "yil", "ay"}
    
            if required_cols.issubset(set(df_fast.columns)):
                return df_fast
        except Exception:
            pass

    # CACHE VARSA → direkt yükle
    if os.path.exists(cache_path):
        try:
            cached = pd.read_pickle(cache_path)
            required_cols = {"tarih", "profil", "siparis_no", "adet", "yil", "ay"}
            
            if required_cols.issubset(set(cached.columns)):
                optional_defaults = {
                    "musteri_siparis_no": "",
                    "firma_adi": "",
                    "kg": 0,
                    "pres": "Bilinmiyor",
                    "termin": pd.NaT,
                    "termin_hafta": pd.NA,
                }
                for col, default_value in optional_defaults.items():
                    if col not in cached.columns:
                        cached[col] = default_value
                return cached
    
        except Exception:
            pass
    
        try:
            os.remove(cache_path)
        except Exception:
            pass
    try:
        xls = pd.ExcelFile(excel_path)
    except Exception as e:
        raise ValueError(f"Excel okunamadı: {str(e)}")

    sheet_name = xls.sheet_names[0]
    header_row = detect_header_row(excel_path, sheet_name=sheet_name)
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header_row)
    df = df.dropna(axis=1, how="all").dropna(how="all").copy()

    tarih_col = find_column(df, "tarih")
    profil_col = find_column(df, "profil")
    siparis_col = find_column(df, "siparis_no")
    adet_col = find_column(df, "adet")
    kg_col = find_column(df, "kg")
    musteri_col = find_column(df, "musteri")
    firma_col = find_column(df, "firma")
    pres_col = find_column(df, "pres")
    termin_col = find_column(df, "termin")
    termin_hafta_col = find_column(df, "termin_hafta")    

    missing = [
        name for name, col in {
            "Tarih": tarih_col,
            "Profil": profil_col,
            "Sipariş": siparis_col,
            "Adet": adet_col,
        }.items() if col is None
    ]
    if missing:
        raise ValueError(f"Gerekli kolonlar bulunamadı: {', '.join(missing)}")

    work = pd.DataFrame({
        "tarih": pd.to_datetime(df[tarih_col], errors="coerce", dayfirst=True),
        "profil": df[profil_col].fillna("").astype(str).str.strip(),
        "siparis_no": df[siparis_col].fillna("").astype(str).str.strip(),
        "adet": pd.to_numeric(df[adet_col], errors="coerce"),
    })

    if musteri_col:
        work["musteri_siparis_no"] = df[musteri_col].apply(normalize_musteri_value)
    else:
        work["musteri_siparis_no"] = ""
        
    if firma_col:
        work["firma_adi"] = df[firma_col].fillna("").astype(str).str.strip()
    else:
        work["firma_adi"] = ""

    if kg_col:
        work["kg"] = pd.to_numeric(df[kg_col], errors="coerce")
    else:
        work["kg"] = 0
    # PRES
    if pres_col:
        work["pres"] = df[pres_col].fillna("").astype(str).str.strip()
        work["pres"] = work["pres"].replace("", "Bilinmiyor")
    else:
        work["pres"] = "Bilinmiyor"
    
    # TERMIN
    if termin_col:
        work["termin"] = pd.to_datetime(df[termin_col], errors="coerce", dayfirst=True)
    else:
        work["termin"] = pd.NaT
    
    # TERMIN HAFTA
    if termin_hafta_col:
        work["termin_hafta"] = pd.to_numeric(df[termin_hafta_col], errors="coerce")
    else:
        work["termin_hafta"] = None    
        
    work = work.dropna(subset=["tarih", "adet"]).copy()
    work = work[
        work["profil"].astype(str).str.strip().ne("") &
        work["siparis_no"].astype(str).str.strip().ne("")
    ].copy()
    work = work[(work["adet"] >= 1) & (work["adet"] <= 100000)]
    work["adet"] = work["adet"].astype(int)
    work["yil"] = work["tarih"].dt.year.astype(int)
    work["ay"] = work["tarih"].dt.to_period("M").astype(str)
    
    # ✅ mevcut cache (dokunma)
    try:
        work.to_pickle(cache_path)
    except Exception:
        pass
    
    # 🚀 YENİ EKLEDİĞİMİZ (SADECE BU)
    try:
        work.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    
    return work

def filter_data(df: pd.DataFrame, secilen_boy: int, mod: str, profil_ara: str = "") -> pd.DataFrame:
    filtered = df.copy()

    if mod == "Seçilen boy ve altı":
        filtered = filtered[filtered["adet"] <= secilen_boy]
    else:
        filtered = filtered[filtered["adet"] == secilen_boy]

    profil_ara = (profil_ara or "").strip().upper()
    if profil_ara:
        filtered = filtered[filtered["profil"].str.upper().str.contains(profil_ara, na=False)]

    return filtered


def filter_scope_data(df: pd.DataFrame, yillar: Iterable[int], profil_ara: str = "") -> pd.DataFrame:
    filtered = df.copy()

    if yillar:
        filtered = filtered[filtered["yil"].isin([int(str(y)) for y in yillar])]

    profil_ara = (profil_ara or "").strip().upper()
    if profil_ara:
        filtered = filtered[filtered["profil"].str.upper().str.contains(profil_ara, na=False)]

    return filtered

def filter_never_exceed_profiles(scope_df: pd.DataFrame, secilen_boy: int, profil_ara: str = "") -> pd.DataFrame:
    """
    Sadece seçilen boy eşiğini HİÇ aşmamış profilleri getirir.
    Örnek: secilen_boy=10 ise, geçmişte 10 üstüne çıkmış hiçbir profil gelmez.
    """
    if scope_df.empty:
        return scope_df.copy()

    filtered = scope_df.copy()

    profil_ara = (profil_ara or "").strip().upper()
    if profil_ara:
        filtered = filtered[filtered["profil"].str.upper().str.contains(profil_ara, na=False)]

    max_adet_by_profile = filtered.groupby("profil")["adet"].max()
    uygun_profiller = max_adet_by_profile[max_adet_by_profile <= secilen_boy].index

    result = filtered[filtered["profil"].isin(uygun_profiller)].copy()
    return result
    
def build_boy_breakdown(filtered: pd.DataFrame, secilen_boy: int) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(columns=[
            "Boy", "Toplam Kayıt Sayısı", "Farklı Sipariş Sayısı",
            "Farklı Profil Sayısı", "Toplam Üretilen Boy", "Toplam Kg"
        ])

    grouped = filtered.groupby("adet").agg(
        toplam_satir=("siparis_no", "size"),
        farkli_siparis=("siparis_no", pd.Series.nunique),
        farkli_profil=("profil", pd.Series.nunique),
        toplam_boy=("adet", "sum"),
        toplam_kg=("kg", "sum"),
    ).reset_index()

    rows = []
    grouped_map = {int(row["adet"]): row for _, row in grouped.iterrows()}

    for boy in range(secilen_boy, 0, -1):
        row = grouped_map.get(boy)
        if row is None:
            rows.append({
                "Boy": boy,
                "Toplam Kayıt Sayısı": 0,
                "Farklı Sipariş Sayısı": 0,
                "Farklı Profil Sayısı": 0,
                "Toplam Üretilen Boy": 0,
                "Toplam Kg": 0.0,
            })
        else:
            rows.append({
                "Boy": boy,
                "Toplam Kayıt Sayısı": int(row["toplam_satir"]),
                "Farklı Sipariş Sayısı": int(row["farkli_siparis"]),
                "Farklı Profil Sayısı": int(row["farkli_profil"]),
                "Toplam Üretilen Boy": int(row["toplam_boy"]),
                "Toplam Kg": float(round(row["toplam_kg"], 2)),
            })

    result = pd.DataFrame(rows)
    result = result.rename(columns={
        "Boy": "Sipariş Boyu",
        "Toplam Kayıt Sayısı": "Toplam Kayıt Sayısı",
        "Farklı Sipariş Sayısı": "Bu Boydaki Farklı Sipariş Sayısı",
        "Farklı Profil Sayısı": "Bu Boydaki Farklı Profil Sayısı",
        "Toplam Üretilen Boy": "Bu Boydaki Toplam Üretim",
        "Toplam Kg": "Bu Boydaki Toplam Ağırlık (Kg)"
    })
    return result

def build_profile_detail(df: pd.DataFrame, profil_kodu: str, yillar):
    data = df[df["profil"] == profil_kodu].copy()

    if yillar:
        data = data[data["yil"].isin([int(str(y)) for y in yillar])]

    yearly = data.groupby("yil").agg(
        toplam_boy=("adet", "sum"),
        siparis_sayisi=("siparis_no", "nunique")
    ).reset_index()

    boy_dist = data.groupby("adet").agg(
        kac_siparis=("siparis_no", "count"),
        toplam_boy=("adet", "sum")
    ).reset_index().sort_values("adet", ascending=False)

    toplam = int(data["adet"].sum())
    siparis = int(data["siparis_no"].nunique())

    return yearly, boy_dist, toplam, siparis


def build_profile_summary(filtered: pd.DataFrame, hedef_uretim: int) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(columns=[
            UI_COLS["profil"],
            UI_COLS["toplam_kayit"],
            UI_COLS["farkli_siparis"],
            UI_COLS["toplam_uretim"],
            UI_COLS["ilk_tarih"],
            UI_COLS["son_tarih"],
            UI_COLS["siparis_ort"],
            UI_COLS["yillik_tuketim"],
            UI_COLS["onerilen_parti"],
        ])

    profile = filtered.groupby("profil", as_index=False).agg(
        toplam_siparis_kalemi=("siparis_no", "size"),
        farkli_siparis_sayisi=("siparis_no", pd.Series.nunique),
        toplam_uretilen_boy=("adet", "sum"),
        ilk_tarih=("tarih", "min"),
        son_tarih=("tarih", "max"),
    )

    profile["siparis_basina_ortalama_boy"] = (
        profile["toplam_uretilen_boy"] / profile["farkli_siparis_sayisi"].replace(0, pd.NA)
    ).round(2).fillna(0)

    profile["ilk_tarih"] = profile["ilk_tarih"].dt.strftime("%Y-%m-%d")
    profile["son_tarih"] = profile["son_tarih"].dt.strftime("%Y-%m-%d")

    yil_sayisi = max(filtered["yil"].nunique(), 1)

    profile[UI_COLS["yillik_tuketim"]] = (
        profile["toplam_uretilen_boy"] / yil_sayisi
    ).round(0)
    
    profile[UI_COLS["onerilen_parti"]] = (
        profile[UI_COLS["yillik_tuketim"]] / hedef_uretim
    ).round(0)

    profile = profile.rename(columns={
        "profil": UI_COLS["profil"],
        "toplam_siparis_kalemi": UI_COLS["toplam_kayit"],
        "farkli_siparis_sayisi": UI_COLS["farkli_siparis"],
        "toplam_uretilen_boy": UI_COLS["toplam_uretim"],
        "ilk_tarih": UI_COLS["ilk_tarih"],
        "son_tarih": UI_COLS["son_tarih"],
        "siparis_basina_ortalama_boy": UI_COLS["siparis_ort"],
    })
    
    return profile.sort_values(
        [UI_COLS["toplam_kayit"], UI_COLS["toplam_uretim"]],
        ascending=[False, False]
    )

def build_high_volume_profile_summary(scope_df: pd.DataFrame, min_boy: int, hedef_uretim: int) -> pd.DataFrame:
    filtered = scope_df[scope_df["adet"] >= min_boy].copy()

    if filtered.empty:
        return pd.DataFrame(columns=[
            UI_COLS["profil"],
            UI_COLS["toplam_kayit"],
            UI_COLS["farkli_siparis"],
            UI_COLS["toplam_uretim"],
            UI_COLS["ilk_tarih"],
            UI_COLS["son_tarih"],
            UI_COLS["siparis_ort"],
            UI_COLS["yillik_tuketim"],
            UI_COLS["onerilen_parti"],
        ])

    profile = filtered.groupby("profil", as_index=False).agg(
        toplam_siparis_kalemi=("siparis_no", "size"),
        farkli_siparis_sayisi=("siparis_no", pd.Series.nunique),
        toplam_uretilen_boy=("adet", "sum"),
        ilk_tarih=("tarih", "min"),
        son_tarih=("tarih", "max"),
    )

    profile["siparis_basina_ortalama_boy"] = (
        profile["toplam_uretilen_boy"] / profile["farkli_siparis_sayisi"].replace(0, pd.NA)
    ).round(2).fillna(0)

    profile["ilk_tarih"] = profile["ilk_tarih"].dt.strftime("%Y-%m-%d")
    profile["son_tarih"] = profile["son_tarih"].dt.strftime("%Y-%m-%d")

    yil_sayisi = max(filtered["yil"].nunique(), 1)

    profile[UI_COLS["yillik_tuketim"]] = (
        profile["toplam_uretilen_boy"] / yil_sayisi
    ).round(0)
    
    profile[UI_COLS["onerilen_parti"]] = (
        profile[UI_COLS["yillik_tuketim"]] / hedef_uretim
    ).round(0)
    
    profile = profile.rename(columns={
        "profil": UI_COLS["profil"],
        "toplam_siparis_kalemi": UI_COLS["toplam_kayit"],
        "farkli_siparis_sayisi": UI_COLS["farkli_siparis"],
        "toplam_uretilen_boy": UI_COLS["toplam_uretim"],
        "ilk_tarih": UI_COLS["ilk_tarih"],
        "son_tarih": UI_COLS["son_tarih"],
        "siparis_basina_ortalama_boy": UI_COLS["siparis_ort"],
    })
    
    return profile.sort_values(
        [UI_COLS["toplam_uretim"], UI_COLS["toplam_kayit"]],
        ascending=[False, False]
    )

def build_high_volume_year_summary(scope_df: pd.DataFrame, min_boy: int) -> pd.DataFrame:
    filtered = scope_df[scope_df["adet"] >= min_boy].copy()

    if filtered.empty:
        return pd.DataFrame(columns=[
            "yil",
            "satir_sayisi",
            "benzersiz_siparis",
            "benzersiz_profil",
            "toplam_adet",
            "toplam_kg"
        ])

    year = filtered.groupby("yil", as_index=False).agg(
        satir_sayisi=("siparis_no", "size"),
        benzersiz_siparis=("siparis_no", pd.Series.nunique),
        benzersiz_profil=("profil", pd.Series.nunique),
        toplam_adet=("adet", "sum"),
        toplam_kg=("kg", "sum"),
    )
    year["toplam_kg"] = year["toplam_kg"].round(2)
    return year.sort_values("yil")

def build_dashboard_kpis(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=["KPI", "Değer"])

    toplam_satir = len(scope_df)
    benzersiz_siparis = int(scope_df["siparis_no"].nunique())
    benzersiz_profil = int(scope_df["profil"].nunique())
    toplam_adet = int(scope_df["adet"].sum())
    toplam_kg = float(scope_df["kg"].fillna(0).sum())

    kpi_rows = [
        {"KPI": "Toplam Kayıt Sayısı", "Değer": f"{toplam_satir:,}"},
        {"KPI": "Farklı Sipariş Sayısı", "Değer": f"{benzersiz_siparis:,}"},
        {"KPI": "Farklı Profil Sayısı", "Değer": f"{benzersiz_profil:,}"},
        {"KPI": "Toplam Üretim (Boy)", "Değer": f"{toplam_adet:,}"},
        {"KPI": "Toplam Ağırlık (Kg)", "Değer": f"{toplam_kg:,.2f}"},
    ]
    return pd.DataFrame(kpi_rows)


def build_dashboard_monthly(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=["ay", "toplam_adet", "toplam_kg", "siparis_sayisi"])

    monthly = scope_df.groupby("ay", as_index=False).agg(
        toplam_adet=("adet", "sum"),
        toplam_kg=("kg", "sum"),
        siparis_sayisi=("siparis_no", pd.Series.nunique),
    ).sort_values("ay")

    monthly["toplam_kg"] = monthly["toplam_kg"].round(2)
    return monthly

def build_seasonality_table(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=["Ay No", "Ay", "Toplam Boy", "Toplam Kg", "Sipariş Sayısı"])

    df = scope_df.copy()
    df["ay_no"] = df["tarih"].dt.month

    sezon = df.groupby("ay_no", as_index=False).agg(
        toplam_adet=("adet", "sum"),
        toplam_kg=("kg", "sum"),
        siparis_sayisi=("siparis_no", pd.Series.nunique),
    )

    ay_map = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
        5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
        9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }

    sezon["Ay"] = sezon["ay_no"].map(ay_map)
    sezon["toplam_kg"] = sezon["toplam_kg"].round(2)

    sezon = sezon[["ay_no", "Ay", "toplam_adet", "toplam_kg", "siparis_sayisi"]]
    sezon.columns = ["Ay No", "Ay", "Toplam Boy", "Toplam Kg", "Sipariş Sayısı"]

    return sezon.sort_values("Ay No")


def build_year_month_pivot(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame()

    df = scope_df.copy()
    df["ay_no"] = df["tarih"].dt.month

    pivot = pd.pivot_table(
        df,
        index="ay_no",
        columns="yil",
        values="adet",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    ay_map = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
        5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
        9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }

    pivot.insert(1, "Ay", pivot["ay_no"].map(ay_map))
    pivot = pivot.rename(columns={"ay_no": "Ay No"})
    return pivot.sort_values("Ay No")


def seasonality_chart(season_df: pd.DataFrame):
    if season_df.empty:
        return None

    fig = px.bar(
        season_df,
        x="Ay",
        y="Toplam Boy",
        title="Sezonsallık Analizi - Aylara Göre Toplam Üretim",
        text="Toplam Boy",
        hover_data=["Toplam Kg", "Sipariş Sayısı"],
    )
    fig.update_layout(height=420)
    return fig


def moving_average_chart(monthly_df: pd.DataFrame):
    if monthly_df.empty:
        return None

    chart_df = monthly_df.copy().sort_values("ay")
    chart_df["hareketli_ortalama_3"] = chart_df["toplam_adet"].rolling(3, min_periods=1).mean().round(2)

    fig = px.line(
        chart_df,
        x="ay",
        y=["toplam_adet", "hareketli_ortalama_3"],
        markers=True,
        title="Aylık Üretim Trendi ve 3 Aylık Hareketli Ortalama",
    )
    fig.update_layout(height=420)
    return fig
    
def build_dashboard_top_profiles(scope_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=[
            UI_COLS["profil"],
            UI_COLS["toplam_uretim"],
            UI_COLS["toplam_kg"],
            UI_COLS["siparis_sayisi"],
        ])

    prof = scope_df.groupby("profil", as_index=False).agg(
        toplam_adet=("adet", "sum"),
        toplam_kg=("kg", "sum"),
        siparis_sayisi=("siparis_no", pd.Series.nunique),
    )

    prof["toplam_kg"] = prof["toplam_kg"].round(2)

    prof = prof.sort_values(
        ["toplam_adet", "toplam_kg"],
        ascending=[False, False]
    ).head(top_n)

    prof.columns = [
        UI_COLS["profil"],
        UI_COLS["toplam_uretim"],
        UI_COLS["toplam_kg"],
        UI_COLS["siparis_sayisi"],
    ]
    return prof


def build_termin_dashboard(scope_df: pd.DataFrame):
    if scope_df.empty or "termin" not in scope_df.columns:
        return pd.DataFrame([
            {"Metrik": "Termin Uyum Oranı", "Değer": "Termin verisi yok"}
        ])

    df = scope_df.copy()

    df = df.dropna(subset=["termin", "tarih"])

    if df.empty:
        return pd.DataFrame([
            {"Metrik": "Termin Uyum Oranı", "Değer": "Veri yok"}
        ])

    df["termin_fark"] = (df["termin"] - df["tarih"]).dt.days

    toplam = len(df)
    zamaninda = (df["termin_fark"] >= 0).sum()
    geciken = (df["termin_fark"] < 0).sum()

    uyum_orani = (zamaninda / toplam * 100) if toplam > 0 else 0

    return pd.DataFrame([
        {"Metrik": "Toplam İş", "Değer": f"{toplam:,}"},
        {"Metrik": "Zamanında", "Değer": f"{zamaninda:,}"},
        {"Metrik": "Geciken", "Değer": f"{geciken:,}"},
        {"Metrik": "Termin Uyum Oranı", "Değer": f"%{uyum_orani:.2f}"},
    ])
    
def build_high_volume_raw(scope_df: pd.DataFrame, min_boy: int) -> pd.DataFrame:
    filtered = scope_df[scope_df["adet"] >= min_boy].copy()

    if filtered.empty:
        return pd.DataFrame(columns=["tarih", "firma_adi", "siparis_no", "musteri_siparis_no", "profil", "adet", "kg"])

    raw_cols = ["tarih", "firma_adi", "siparis_no", "musteri_siparis_no", "profil", "adet", "kg"]
    raw = filtered[raw_cols].sort_values("adet", ascending=False).copy()
    raw["tarih"] = raw["tarih"].dt.strftime("%Y-%m-%d")
    return raw.head(500)


def high_volume_summary_markdown(scope_df: pd.DataFrame, min_boy: int) -> str:
    filtered = scope_df[scope_df["adet"] >= min_boy].copy()

    if filtered.empty:
        return "### Sonuç\nSeçilen filtrelere göre yüksek üretim kaydı bulunamadı."

    yil_min = int(filtered["yil"].min())
    yil_max = int(filtered["yil"].max())

    toplam_satir = len(filtered)
    toplam_siparis = filtered["siparis_no"].nunique()
    toplam_profil = filtered["profil"].nunique()
    toplam_adet = int(filtered["adet"].sum())
    toplam_kg = float(filtered["kg"].fillna(0).sum())
    ort_boy = round(filtered["adet"].mean(), 2)

    en_cok = (
        filtered.groupby("profil", as_index=False)
        .agg(toplam_adet=("adet", "sum"))
        .sort_values("toplam_adet", ascending=False)
        .head(1)
    )

    if not en_cok.empty:
        lider_profil = en_cok.iloc[0]["profil"]
        lider_adet = int(en_cok.iloc[0]["toplam_adet"])
    else:
        lider_profil = "-"
        lider_adet = 0

    lines = [
        "## 🟢 Büyük Sipariş Analizi (Core Üretim)",
        "",
        "### En Çok Üretime Giren Ürünler Özeti",
        f"- Kriter: **{min_boy} boy ve üstü**",
        f"- Tarih aralığı: **{yil_min} - {yil_max}**",
        f"- Toplam kayıt sayısı: **{toplam_satir:,}**",
        f"- Farklı sipariş sayısı: **{toplam_siparis:,}**",
        f"- Farklı profil sayısı: **{toplam_profil:,}**",
        f"- Toplam üretilen boy: **{toplam_adet:,}**",
        f"- Toplam kg: **{toplam_kg:,.2f}**",
        f"- Sipariş başına ortalama boy: **{ort_boy:,}**",
        f"- En çok üretime giren profil: **{lider_profil}**",
        f"- Bu profilin toplam üretim miktarı: **{lider_adet:,} boy**",
    ]

    return "\n".join(lines)


def build_abc_analysis(scope_df: pd.DataFrame, hedef_uretim: int) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=[
            "Profil Kodu",
            "Toplam Üretilen Boy",
            "Toplam Kayıt Sayısı",
            "Farklı Sipariş Sayısı",
            "Yıllık Tüketim",
            "Yeni Akıllı Öneri (Boy)",
            "Kümülatif Pay (%)",
            "ABC Sınıfı",
            "Stok Önerisi"
        ])

    profile = scope_df.groupby("profil", as_index=False).agg(
        toplam_uretilen_boy=("adet", "sum"),
        toplam_siparis_kalemi=("siparis_no", "size"),
        farkli_siparis_sayisi=("siparis_no", pd.Series.nunique),
    )

    yil_sayisi = max(scope_df["yil"].nunique(), 1)
    toplam_genel = profile["toplam_uretilen_boy"].sum()

    profile = profile.sort_values("toplam_uretilen_boy", ascending=False).reset_index(drop=True)

    profile[UI_COLS["yillik_tuketim"]] = (
        profile["toplam_uretilen_boy"] / yil_sayisi
    ).round(0)
    
    profile[UI_COLS["onerilen_parti"]] = (
        profile[UI_COLS["yillik_tuketim"]] / hedef_uretim
    ).round(0)
    
    if toplam_genel > 0:
        profile["pay"] = profile["toplam_uretilen_boy"] / toplam_genel * 100
        profile[UI_COLS["kumulatif_pay"]] = profile["pay"].cumsum().round(2)
    else:
        profile["pay"] = 0
        profile[UI_COLS["kumulatif_pay"]] = 0

    def abc_label(kum_pay):
        if kum_pay <= 80:
            return "A"
        elif kum_pay <= 95:
            return "B"
        return "C"

    profile[UI_COLS["abc_sinifi"]] = profile[UI_COLS["kumulatif_pay"]].apply(abc_label)
    
    def stok_onerisi(row):
        if row[UI_COLS["abc_sinifi"]] == "A":
            return "Evet"
        elif row[UI_COLS["abc_sinifi"]] == "B":
            return "Planlı Üret"
        return "Hayır"
    
    profile[UI_COLS["stok_karari"]] = profile.apply(stok_onerisi, axis=1)

    profile = profile[[
        "profil",
        "toplam_uretilen_boy",
        "toplam_siparis_kalemi",
        "farkli_siparis_sayisi",
        UI_COLS["yillik_tuketim"],
        UI_COLS["onerilen_parti"],
        UI_COLS["kumulatif_pay"],
        UI_COLS["abc_sinifi"],
        UI_COLS["stok_karari"],
    ]]
    
    profile = profile.rename(columns={
        "profil": UI_COLS["profil"],
        "toplam_uretilen_boy": UI_COLS["toplam_uretim"],
        "toplam_siparis_kalemi": UI_COLS["toplam_kayit"],
        "farkli_siparis_sayisi": UI_COLS["farkli_siparis"],
    })
    
    return profile

def build_profit_simulation(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame()

    df = scope_df.groupby("profil", as_index=False).agg(
        toplam_adet=("adet", "sum"),
        siparis_sayisi=("siparis_no", pd.Series.nunique),
        satir_sayisi=("siparis_no", "size"),
    )

    # Ortalama sipariş büyüklüğü
    df["ort_boy"] = (
        df["toplam_adet"] / df["siparis_sayisi"].replace(0, pd.NA)
    ).round(2).fillna(0)
    
    df["yogunluk_skor"] = (
        df["satir_sayisi"] / df["toplam_adet"].replace(0, pd.NA)
    ).round(4).fillna(0)

    # Basit değer skoru
    df["deger_skor"] = (df["toplam_adet"] * df["ort_boy"]).round(2)

    # Normalize skor
    df["stratejik_skor"] = (
        df["deger_skor"] / (df["yogunluk_skor"] + 0.0001)
    ).round(2)

    df = df.sort_values("stratejik_skor", ascending=False)

    def karar(row):
        if row["stratejik_skor"] > df["stratejik_skor"].quantile(0.8):
            return "🔥 STRATEJİK (stok yapılır)"
        elif row["stratejik_skor"] > df["stratejik_skor"].quantile(0.5):
            return "⚖️ TAKİP (planlı üretim)"
        else:
            return "❌ ZAYIF (sipariş bazlı üret)"

    df["Karar"] = df.apply(karar, axis=1)

    df.columns = [
        "Profil",
        "Toplam Boy",
        "Sipariş Sayısı",
        "Satır Sayısı",
        "Ortalama Boy",
        "Yoğunluk Skoru",
        "Değer Skoru",
        "Stratejik Skor",
        "Karar"
    ]

    return df

# =========================
# 👤 CUSTOMER ENGINE
# =========================
def build_customer_detail(scope_df: pd.DataFrame, musteri_adi: str, secilen_boy: int):
    musteri_adi = str(musteri_adi).strip().upper()
    mapping = load_customer_mapping()
    grup_listesi = get_customer_group_list(mapping, musteri_adi)

    if not grup_listesi:
        return pd.DataFrame(), "Veri bulunamadı"

    df = scope_df[
        scope_df["musteri_siparis_no"].astype(str).str.upper().isin(grup_listesi)
    ].copy()

    if df.empty:
        return pd.DataFrame(), f"Seçilen müşteri grubu için kayıt bulunamadı.\n\nGrup: {musteri_adi}"

    toplam_satir = len(df)
    toplam_adet = int(df["adet"].sum())
    toplam_kg = float(df["kg"].fillna(0).sum())
    farkli_siparis = int(df["siparis_no"].nunique())
    farkli_profil = int(df["profil"].nunique())

    kucuk = df[df["adet"] <= secilen_boy].copy()
    buyuk = df[df["adet"] > secilen_boy].copy()

    kucuk_satir = len(kucuk)
    buyuk_satir = len(buyuk)

    kucuk_adet = int(kucuk["adet"].sum())
    buyuk_adet = int(buyuk["adet"].sum())

    kucuk_oran = (kucuk_satir / toplam_satir * 100) if toplam_satir else 0
    buyuk_oran = (buyuk_satir / toplam_satir * 100) if toplam_satir else 0

    kucuk_adet_oran = (kucuk_adet / toplam_adet * 100) if toplam_adet else 0
    buyuk_adet_oran = (buyuk_adet / toplam_adet * 100) if toplam_adet else 0

    summary_df = pd.DataFrame([
        ["İncelenen Müşteri Grubu", musteri_adi],
        ["Bu gruba bağlı alt müşteri/kod sayısı", len(grup_listesi)],
        ["Toplam Kayıt Sayısı", toplam_satir],
        ["Farklı Sipariş Sayısı", farkli_siparis],
        ["Farklı Profil Sayısı", farkli_profil],
        ["Toplam Üretim (Boy)", toplam_adet],
        ["Toplam Kg", round(toplam_kg, 2)],
        [f"{secilen_boy} Boy ve Altı Sipariş Kalemi", kucuk_satir],
        [f"{secilen_boy} Boy Üstü Sipariş Kalemi", buyuk_satir],
        [f"{secilen_boy} Boy ve Altı Üretim", kucuk_adet],
        [f"{secilen_boy} Boy Üstü Üretim", buyuk_adet],
        [f"{secilen_boy} Boy ve Altı Sipariş (%)", round(kucuk_oran, 1)],
        [f"{secilen_boy} Boy Üstü Sipariş (%)", round(buyuk_oran, 1)],
        [f"{secilen_boy} Boy ve Altı Üretim (%)", round(kucuk_adet_oran, 1)],
        [f"{secilen_boy} Boy Üstü Üretim (%)", round(buyuk_adet_oran, 1)],
    ], columns=["Metrik", "Değer"])

    if kucuk_oran > 50:
        yorum = f"❌ Bu müşteri grubunda {secilen_boy} boy ve altı sipariş yükü yüksek."
    elif kucuk_oran > 25:
        yorum = f"⚠️ Bu müşteri grubunda {secilen_boy} boy ve altı sipariş yoğunluğu dikkat çekiyor."
    else:
        yorum = f"✅ Bu müşteri grubu {secilen_boy} boy eşiğine göre daha dengeli çalışıyor."

    detay_text = f"""
    ## 👤 Müşteri Analizi: {musteri_adi}
    
    ### Grup Kapsamı
    {", ".join(grup_listesi[:50])}
    
    ### Özet
    - Toplam Kayıt Sayısı: **{toplam_satir}**
    - Farklı sipariş: **{farkli_siparis}**
    - Farklı profil: **{farkli_profil}**
    - Toplam üretim: **{toplam_adet} boy**
    - Toplam kg: **{toplam_kg:,.2f}**
    
    ### {secilen_boy} Boy ve Altı
    - **{kucuk_satir}** sipariş kalemi
    - **{kucuk_adet} boy**
    - Sipariş oranı: **%{kucuk_oran:.1f}**
    - Üretim oranı: **%{kucuk_adet_oran:.1f}**
    
    ### {secilen_boy} Boy Üstü
    - **{buyuk_satir}** sipariş kalemi
    - **{buyuk_adet} boy**
    - Sipariş oranı: **%{buyuk_oran:.1f}**
    - Üretim oranı: **%{buyuk_adet_oran:.1f}**
    
    ### 🎯 Yorum
    {yorum}
    """
    return summary_df, detay_text

# =========================
# 🚨 ROOT CAUSE ENGINE
# =========================
def build_root_cause(scope_df: pd.DataFrame, secilen_boy: int):
    kucuk = scope_df[scope_df["adet"] <= secilen_boy]

    if kucuk.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    kucuk_musteri = kucuk[
        kucuk["musteri_siparis_no"].astype(str).str.strip().ne("")
    ].copy()
    
    musteri = kucuk.groupby("musteri_siparis_no").agg(
        siparis=("siparis_no", "count"),
        toplam=("adet", "sum")
    ).sort_values("siparis", ascending=False).head(10)

    profil = kucuk.groupby("profil").agg(
        siparis=("siparis_no", "count"),
        toplam=("adet", "sum")
    ).sort_values("siparis", ascending=False).head(10)

    pres = kucuk.groupby("pres").agg(
        siparis=("siparis_no", "count"),
        toplam=("adet", "sum")
    ).sort_values("siparis", ascending=False)

    return musteri, profil, pres

# =========================
# 📈 FORECAST ENGINE
# =========================
def build_forecast_table(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=[
            "ay", "toplam_adet", "hareketli_ortalama_tahmin", "sapma"
        ])

    monthly = scope_df.copy()
    monthly["ay_dt"] = pd.to_datetime(monthly["ay"])
    
    monthly = monthly.groupby("ay_dt", as_index=False).agg(
        toplam_adet=("adet", "sum")
    ).sort_values("ay_dt")
    
    monthly["ay"] = monthly["ay_dt"].dt.strftime("%Y-%m")
    monthly = monthly.drop(columns="ay_dt")

    monthly["forecast_3_ay_ort"] = (
        monthly["toplam_adet"]
        .rolling(3, min_periods=1)
        .mean()
        .shift(1)
        .round(2)
    )

    monthly["forecast_3_ay_ort"] = monthly["forecast_3_ay_ort"].fillna(monthly["toplam_adet"])
    monthly["sapma"] = (monthly["toplam_adet"] - monthly["forecast_3_ay_ort"]).round(2)

    monthly = monthly.rename(columns={
        "ay": "Ay",
        "toplam_adet": "Gerçekleşen Üretim",
        "forecast_3_ay_ort": "3 Aylık Ortalama Tahmin",
        "sapma": "Tahmin Sapması"
    })
    return monthly

def forecast_chart(forecast_df: pd.DataFrame):
    if forecast_df.empty:
        return None

    fig = px.line(
        forecast_df,
        x="Ay",
        y=["Gerçekleşen Üretim", "3 Aylık Ortalama Tahmin"],
        markers=True,
        title="Aylık Gerçekleşen ve 3 Aylık Hareketli Ortalama",
    )
    fig.update_layout(height=420)
    return fig

# =========================
# 🔮 SCENARIO ENGINE
# =========================
def build_scenario_table(scope_df: pd.DataFrame, secilen_boy: int, hedef_kucuk_oran: float) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=["Senaryo", "Değer"])

    toplam_satir = len(scope_df)
    toplam_adet = int(scope_df["adet"].sum())

    kucuk = scope_df[scope_df["adet"] <= secilen_boy].copy()
    kucuk_satir = len(kucuk)
    kucuk_adet = int(kucuk["adet"].sum())

    mevcut_kucuk_oran = (kucuk_satir / toplam_satir * 100) if toplam_satir else 0
    hedef_oran = float(hedef_kucuk_oran)
    agresif_oran = max(1, min(hedef_oran - 3, hedef_oran))

    kalip_degisim_sayisi = kucuk.groupby("siparis_no")["profil"].nunique().sum()
    mevcut_kalip_suresi = float(kalip_degisim_sayisi) * (5 / 60)
    
    def scenario_calc(target_ratio):
        oran_iyilesme = max(mevcut_kucuk_oran - target_ratio, 0)
        saat_kazanci = round((oran_iyilesme / 100) * mevcut_kalip_suresi, 2)
        gun_kazanci = round(saat_kazanci / 24, 2)
        tahmini_yeni_kucuk_satir = int(round(toplam_satir * target_ratio / 100, 0))

        return {
            "oran_iyilesme": round(oran_iyilesme, 2),
            "saat_kazanci": saat_kazanci,
            "gun_kazanci": gun_kazanci,
            "tahmini_yeni_kucuk_satir": tahmini_yeni_kucuk_satir
        }

    hedef_sonuc = scenario_calc(hedef_oran)
    agresif_sonuc = scenario_calc(agresif_oran)

    rows = [
        {"Senaryo": "Toplam Sipariş Satırı", "Değer": f"{toplam_satir:,}"},
        {"Senaryo": "Toplam Üretim (Boy)", "Değer": f"{toplam_adet:,}"},
        {"Senaryo": f"Mevcut Küçük Sipariş Oranı (≤{secilen_boy})", "Değer": f"%{mevcut_kucuk_oran:.2f}"},
        {"Senaryo": "Mevcut Küçük Sipariş Satırı", "Değer": f"{kucuk_satir:,}"},
        {"Senaryo": "Mevcut Küçük Sipariş Üretimi", "Değer": f"{kucuk_adet:,}"},
        {"Senaryo": f"Hedef Senaryo Oranı", "Değer": f"%{hedef_oran:.2f}"},
        {"Senaryo": "Hedef Senaryo Tahmini Küçük Sipariş", "Değer": f"{hedef_sonuc['tahmini_yeni_kucuk_satir']:,}"},
        {"Senaryo": "Hedef Senaryo Saat Kazancı", "Değer": f"{hedef_sonuc['saat_kazanci']:,}"},
        {"Senaryo": "Hedef Senaryo Gün Kazancı", "Değer": f"{hedef_sonuc['gun_kazanci']:,}"},
        {"Senaryo": f"Agresif Senaryo Oranı", "Değer": f"%{agresif_oran:.2f}"},
        {"Senaryo": "Agresif Senaryo Tahmini Küçük Sipariş", "Değer": f"{agresif_sonuc['tahmini_yeni_kucuk_satir']:,}"},
        {"Senaryo": "Agresif Senaryo Saat Kazancı", "Değer": f"{agresif_sonuc['saat_kazanci']:,}"},
        {"Senaryo": "Agresif Senaryo Gün Kazancı", "Değer": f"{agresif_sonuc['gun_kazanci']:,}"},
    ]

    return pd.DataFrame(rows)


def scenario_summary_markdown(scope_df: pd.DataFrame, secilen_boy: int, hedef_kucuk_oran: float) -> str:
    if scope_df.empty:
        return "### Senaryo sonucu üretilemedi"

    toplam_satir = len(scope_df)
    kucuk = scope_df[scope_df["adet"] <= secilen_boy].copy()
    kucuk_satir = len(kucuk)

    mevcut_oran = (kucuk_satir / toplam_satir * 100) if toplam_satir else 0
    hedef_oran = float(hedef_kucuk_oran)
    agresif_oran = max(1, min(hedef_oran - 3, hedef_oran))

    kalip_degisim_sayisi = kucuk.groupby("siparis_no")["profil"].nunique().sum()

    def hesapla(oran):
        iyilesme = max(mevcut_oran - oran, 0)
        kazanc_saat = (iyilesme / 100) * kalip_degisim_sayisi
        kazanc_gun = kazanc_saat / 24
        return iyilesme, kazanc_saat, kazanc_gun

    iy1, ks1, kg1 = hesapla(hedef_oran)
    iy2, ks2, kg2 = hesapla(agresif_oran)

    lines = [
        "## 🔮 Scenario Engine",
        "",
        f"- Mevcut küçük sipariş oranı: **%{mevcut_oran:.1f}**",
        f"- Hedef oran: **%{hedef_oran:.1f}**",
        f"- Agresif oran: **%{agresif_oran:.1f}**",
        "",
        "### Senaryo 1: Hedef Oran",
        f"- İyileşme: **%{iy1:.1f}**",
        f"- Kazanç: **{ks1:.1f} saat** (~**{kg1:.1f} gün**)",
        "",
        "### Senaryo 2: Agresif Oran",
        f"- İyileşme: **%{iy2:.1f}**",
        f"- Kazanç: **{ks2:.1f} saat** (~**{kg2:.1f} gün**)",
        "",
        "### Yönetim Yorumu",
        "- Küçük sipariş oranı düştükçe kalıp değişim yükü azalır",
        "- Aynı profili birleştirerek üretmek hat ve planlama verimini artırır",
        "- Bu tablo karar toplantısında direkt kullanılabilir",
    ]
    return "\n".join(lines)
    
def abc_summary_markdown(abc_df: pd.DataFrame, scope_df: pd.DataFrame) -> str:
    if abc_df.empty:
        return "### Sonuç\nABC analizi için kayıt bulunamadı."

    a_sayisi = int((abc_df[UI_COLS["abc_sinifi"]] == "A").sum())
    b_sayisi = int((abc_df[UI_COLS["abc_sinifi"]] == "B").sum())
    c_sayisi = int((abc_df[UI_COLS["abc_sinifi"]] == "C").sum())

    a_toplam = int(
        abc_df[abc_df[UI_COLS["abc_sinifi"]] == "A"][UI_COLS["toplam_uretim"]].sum()
    )
    b_toplam = int(
        abc_df[abc_df[UI_COLS["abc_sinifi"]] == "B"][UI_COLS["toplam_uretim"]].sum()
    )
    c_toplam = int(
        abc_df[abc_df[UI_COLS["abc_sinifi"]] == "C"][UI_COLS["toplam_uretim"]].sum()
    )
    
    stok_adet = int((abc_df[UI_COLS["stok_karari"]] == "Evet").sum())
    
    # YENİ EKLENENLER
    toplam_kalip = int(scope_df["profil"].nunique())
    toplam_boy = int(scope_df["adet"].sum())
    toplam_kg = float(scope_df["kg"].fillna(0).sum())
    
    lines = [
        "## 📦 ABC Analizi ve Stok Önerisi",
        "",
        f"- A grubu profil sayısı: **{a_sayisi}**",
        f"- B grubu profil sayısı: **{b_sayisi}**",
        f"- C grubu profil sayısı: **{c_sayisi}**",
        "",
        f"- A grubu toplam üretim: **{a_toplam:,} boy**",
        f"- B grubu toplam üretim: **{b_toplam:,} boy**",
        f"- C grubu toplam üretim: **{c_toplam:,} boy**",
        "",
        f"- Doğrudan stok önerilen profil sayısı: **{stok_adet}**",
        "",
        "### 📊 Genel Üretim Özeti",
        f"- Toplam kullanılan kalıp (profil): **{toplam_kalip:,}**",
        f"- Toplam çekilen profil boy: **{toplam_boy:,} boy**",
        f"- Toplam üretilen kg: **{toplam_kg:,.2f} kg**",
        "",
        "### Yorum",
        "- **A grubu**: stok yapılmalı",
        "- **B grubu**: planlı / parti üretim yapılmalı",
        "- **C grubu**: sipariş gelmeden stok yapılmamalı",
    ]
    return "\n".join(lines)

def build_executive_summary(scope_df, abc_df, secilen_boy, hedef_kucuk_oran):
    if scope_df.empty:
        return "### Veri bulunamadı"

    toplam_adet = int(scope_df["adet"].sum())
    toplam_satir = len(scope_df)

    kucuk = scope_df[scope_df["adet"] <= secilen_boy]
    kucuk_adet = int(kucuk["adet"].sum())
    kucuk_satir = len(kucuk)

    kucuk_satir_yuzde = (kucuk_satir / toplam_satir * 100) if toplam_satir else 0
    kucuk_adet_yuzde = (kucuk_adet / toplam_adet * 100) if toplam_adet else 0
    verimsizlik_skoru = kucuk_satir_yuzde - kucuk_adet_yuzde
    # KALIP DEĞİŞİM ANALİZİ
    farkli_profil_sayisi = kucuk["profil"].nunique()
    
    # varsayım: her profil = 1 kalıp değişimi
    kalip_degisim_sayisi = kucuk.groupby("siparis_no")["profil"].nunique().sum()
    
    # ortalama 1 saat
    toplam_kalip_suresi_saat = kalip_degisim_sayisi * (5 / 60)
    toplam_kalip_suresi_gun = toplam_kalip_suresi_saat / 24
    # AKILLI YORUM MOTORU
    
    if verimsizlik_skoru > 15:
        yorum_text = "❌ Sistem ciddi verimsizlik üretiyor (çok sipariş, düşük üretim)"
    elif verimsizlik_skoru > 8:
        yorum_text = "⚠️ Operasyonel verimsizlik riski var"
    elif verimsizlik_skoru > 3:
        yorum_text = "ℹ️ Dikkat edilmeli"
    else:
        yorum_text = "✅ Sistem dengeli çalışıyor"

    # EN ÇOK ÜRETİM YAPILAN PROFİLLER
    top_profiles = (
        scope_df.groupby("profil")
        .agg(toplam=("adet", "sum"))
        .sort_values("toplam", ascending=False)
        .head(15)
    )

    top_profiles_total = int(top_profiles["toplam"].sum())
    top_profiles_yuzde = (top_profiles_total / toplam_adet * 100) if toplam_adet else 0

    # MÜŞTERİ ANALİZİ (musteri_siparis_no bazlı)
    musteri_df = scope_df.copy()
    
    musteri_df["musteri_siparis_no"] = musteri_df["musteri_siparis_no"].astype(str).str.strip()
    
    musteri_df = musteri_df[
        (musteri_df["musteri_siparis_no"] != "") &
        (musteri_df["musteri_siparis_no"] != "0") &
        (musteri_df["musteri_siparis_no"].notna())
    ]
    
    musteri = (
        musteri_df.groupby("musteri_siparis_no")
        .agg(
            siparis=("siparis_no", "count"),
            toplam=("adet", "sum"),
            ortalama=("adet", "mean")
        )
        .sort_values(["siparis", "toplam"], ascending=[False, False])
        .head(10)
    )

    # A SINIFI
    a_class = abc_df[abc_df[UI_COLS["abc_sinifi"]] == "A"].head(10)
    # SİMÜLASYON
    mevcut_oran = kucuk_satir_yuzde
    hedef_oran = hedef_kucuk_oran
    ideal_oran = max(hedef_oran - 3, 1)  # daha agresif senaryo
    
    def hesapla(oran):
        iyilesme = max(mevcut_oran - oran, 0)
        kazanc_saat = (iyilesme / 100) * toplam_kalip_suresi_saat
        kazanc_gun = kazanc_saat / 24
        return iyilesme, kazanc_saat, kazanc_gun
    
    iy1, ks1, kg1 = hesapla(hedef_oran)
    iy2, ks2, kg2 = hesapla(ideal_oran)

    lines = [
        "# 🚀 YÖNETİCİ ÖZETİ",
        "",
        "## 📊 Genel Durum",
        f"- Toplam üretim: **{toplam_adet:,} boy**",
        f"- Toplam sipariş: **{toplam_satir:,}**",
        "",
        "## ⚙️ Üretim Yükü",

    f"- Küçük sipariş oranı: **%{kucuk_satir_yuzde:.1f}**",
    f"- Üretime katkısı: **%{kucuk_adet_yuzde:.1f}**",
    
    "",
    "### 🔩 Operasyonel Etki",
    
    f"- Küçük siparişlerde farklı profil sayısı: **{farkli_profil_sayisi}**",
    f"- Tahmini kalıp değişim sayısı: **{kalip_degisim_sayisi}**",
    
    f"- Toplam kalıp değişim süresi: **{toplam_kalip_suresi_saat:,} saat** (~{toplam_kalip_suresi_gun:.1f} gün)",
    
    "",
    "### 📌 Akıllı Değerlendirme",

    yorum_text,
    "",
    f"- {secilen_boy} boy ve altı siparişler toplam siparişlerin **%{kucuk_satir_yuzde:.1f}**’ini oluşturuyor.",
    f"- Bu siparişlerin üretime katkısı sadece **%{kucuk_adet_yuzde:.1f}** seviyesinde.",
    
    f"- Toplam **{kalip_degisim_sayisi}** farklı profil nedeniyle yaklaşık **{toplam_kalip_suresi_saat:,} saat** kalıp değişim süresi oluşmuştur.",
    
    "",
    "### 🎯 Yorum Detayı",
    
    "- Sipariş sayısı yüksek ancak üretim katkısı düşükse → verimsizlik oluşur",
    "- Sipariş oranı düşükse → sistem dengeli çalışır",
    "- Kritik eşik: %20 üzeri sipariş yoğunluğu",
    
    "",
    "📎 Not: Kalıp değişim süresi ortalama **5 dakika** olarak varsayılmıştır.",
    "",
    "## 🔮 Simülasyon (What-if Analizi)",

    "",
    "### 📊 Senaryo 1: Mevcut → Hedef",
    
    f"- Mevcut oran: **%{mevcut_oran:.1f}**",
    f"- Hedef oran: **%{hedef_oran:.1f}**",
    f"- İyileşme: **%{iy1:.1f}**",
    f"- Kazanç: **{ks1:.0f} saat (~{kg1:.1f} gün)**",
    
    "",
    "### 🚀 Senaryo 2: Agresif Optimizasyon",
    
    f"- Hedef oran: **%{ideal_oran:.1f}**",
    f"- İyileşme: **%{iy2:.1f}**",
    f"- Kazanç: **{ks2:.0f} saat (~{kg2:.1f} gün)**",
        "",
        "## 🏆 Kritik Profiller",
        f"- İlk 15 profil üretimin **%{top_profiles_yuzde:.1f}**’ini oluşturuyor",
    ]

    for i, row in enumerate(top_profiles.reset_index().itertuples(), 1):
        lines.append(f"{i}. {row.profil} → {int(row.toplam):,} boy")

    lines.append("")
    lines.append("## 📦 A Sınıfı (Stok Önerilen)")
    
    for _, row in a_class.iterrows():
        lines.append(
            f"- {row[UI_COLS['profil']]} → aylık: {int(row[UI_COLS['yillik_tuketim']]/12):,} / yıllık: {int(row[UI_COLS['yillik_tuketim']]):,}"
        )

    forecast_df = build_forecast_table(scope_df)

    if not forecast_df.empty:
        son_gercek = forecast_df.iloc[-1]["Gerçekleşen Üretim"]
        son_tahmin = forecast_df.iloc[-1]["3 Aylık Ortalama Tahmin"]
        lines.append("")
        lines.append("## 📈 Kısa Vadeli Tahmin")
        lines.append(f"- Son gerçekleşen aylık üretim: **{int(son_gercek):,} boy**")
        lines.append(f"- 3 aylık ortalama tahmin seviyesi: **{int(son_tahmin):,} boy**")

    lines.append("")
    lines.append("## 👤 Müşteri Yük Analizi")

    for i, row in enumerate(musteri.reset_index().itertuples(), 1):
        yuzde = (row.toplam / toplam_adet * 100) if toplam_adet else 0
        lines.append(
            f"{i}. {row.musteri_siparis_no} → {int(row.siparis)} sipariş | "
            f"{int(row.toplam)} boy | %{yuzde:.1f} üretim"
        )

    lines.append("")
    lines.append("## 🔍 Kök Neden Analizi")

    for i, row in enumerate(musteri.reset_index().itertuples(), 1):
        if i > 5:
            break
        lines.append(
            f"{i}. {row.musteri_siparis_no} → {int(row.siparis)} sipariş (küçük sipariş kaynağı)"
        )

    lines.append("")
    lines.append("## 🎯 Aksiyon Önerileri")

    lines.append("- Küçük sipariş üreten müşterilere minimum sipariş kuralı getir")
    lines.append("- Aynı profil siparişlerini birleştir (batch üretim)")
    lines.append("- Düşük adetli profilleri stoktan üret")

    return "\n".join(lines)
    
def abc_chart(abc_df: pd.DataFrame, top_n_value: int):
    if abc_df.empty:
        return None

    top_n = abc_df.head(top_n_value).sort_values(UI_COLS["toplam_uretim"])

    fig = px.bar(
        top_n,
        x=UI_COLS["toplam_uretim"],
        y=UI_COLS["profil"],
        orientation="h",
        color=UI_COLS["abc_sinifi"],
        title=f"ABC Analizi - En Yüksek Tüketimli Profiller (ilk {top_n_value})",
        text=UI_COLS["toplam_uretim"],
        hover_data=[
            UI_COLS["toplam_kayit"],
            UI_COLS["yillik_tuketim"],
            UI_COLS["stok_karari"],
            UI_COLS["kumulatif_pay"],
        ]
    )
    fig.update_layout(height=max(500, top_n_value * 35), yaxis={"automargin": True})
    return fig


def build_year_summary(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(columns=["yil", "satir_sayisi", "benzersiz_siparis", "benzersiz_profil", "toplam_adet", "toplam_kg"])

    year = filtered.groupby("yil", as_index=False).agg(
        satir_sayisi=("siparis_no", "size"),
        benzersiz_siparis=("siparis_no", pd.Series.nunique),
        benzersiz_profil=("profil", pd.Series.nunique),
        toplam_adet=("adet", "sum"),
        toplam_kg=("kg", "sum"),
    )
    year["toplam_kg"] = year["toplam_kg"].round(2)

    year = year.rename(columns={
        "yil": "Yıl",
        "satir_sayisi": "Toplam Kayıt Sayısı",
        "benzersiz_siparis": "Farklı Sipariş Sayısı",
        "benzersiz_profil": "Farklı Profil Sayısı",
        "toplam_adet": "Toplam Üretim (Boy)",
        "toplam_kg": "Toplam Ağırlık (Kg)"
    })
    
    return year.sort_values("Yıl")


def top_profiles_chart(profile_summary: pd.DataFrame, top_n_value: int):
    if profile_summary.empty:
        return None

    top_n = profile_summary.head(top_n_value).sort_values("Toplam Kayıt Sayısı")
    grafik_yuksekligi = max(500, top_n_value * 35)

    fig = px.bar(
        top_n,
        x=UI_COLS["toplam_kayit"],
        y=UI_COLS["profil"],
        orientation="h",
        title=f"En sık geçen profiller (ilk {top_n_value})",
        text=UI_COLS["toplam_kayit"],
        hover_data=[
            UI_COLS["farkli_siparis"],
            UI_COLS["toplam_uretim"]
        ],
    )

    fig.update_layout(
        height=grafik_yuksekligi,
        yaxis={"automargin": True}
    )
    return fig

def high_volume_chart(profile_summary: pd.DataFrame, top_n_value: int):
    if profile_summary.empty:
        return None

    top_n = profile_summary.head(top_n_value).sort_values(UI_COLS["toplam_uretim"])

    grafik_yuksekligi = max(500, top_n_value * 35)

    fig = px.bar(
        top_n,
        x=UI_COLS["toplam_uretim"],
        y=UI_COLS["profil"],
        orientation="h",
        title=f"En çok üretime giren profiller (ilk {top_n_value})",
        text=UI_COLS["toplam_uretim"],
        hover_data=[
            UI_COLS["toplam_kayit"],
            UI_COLS["farkli_siparis"],
            UI_COLS["yillik_tuketim"],
        ]
    )

    fig.update_layout(
        height=grafik_yuksekligi,
        yaxis={"automargin": True}
    )

    return fig
    
def boy_breakdown_chart(boy_breakdown: pd.DataFrame):
    if boy_breakdown.empty:
        return None

    fig = px.bar(
        boy_breakdown.sort_values("Sipariş Boyu"),
        x="Sipariş Boyu",
        y="Toplam Kayıt Sayısı",
        title="Boylara Göre Sipariş Dağılımı",
        text="Toplam Kayıt Sayısı",
        hover_data=[
            "Bu Boydaki Farklı Sipariş Sayısı",
            "Bu Boydaki Farklı Profil Sayısı",
            "Bu Boydaki Toplam Üretim",
            "Bu Boydaki Toplam Ağırlık (Kg)"
        ],
    )

    fig.update_layout(height=400)
    return fig


def monthly_chart(filtered: pd.DataFrame):
    if filtered.empty:
        return None

    monthly = filtered.groupby("ay", as_index=False).agg(
        satir_sayisi=("siparis_no", "size"),
        toplam_adet=("adet", "sum"),
    ).sort_values("ay")

    fig = px.line(
        monthly,
        x="ay",
        y="satir_sayisi",
        markers=True,
        title="Aylık Sipariş Yoğunluğu",
        hover_data=["toplam_adet"],
    )

    fig.update_layout(height=400)
    return fig
    
def build_small_order_monthly(scope_df: pd.DataFrame, secilen_boy: int) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=[
            "ay",
            "toplam_satir",
            "kucuk_satir",
            "satir_oran",
            "toplam_adet",
            "kucuk_adet",
            "adet_oran"
        ])

    toplam = scope_df.groupby("ay", as_index=False).agg(
        toplam_satir=("siparis_no", "size"),
        toplam_adet=("adet", "sum"),
    )

    kucuk = scope_df[scope_df["adet"] <= secilen_boy].groupby("ay", as_index=False).agg(
        kucuk_satir=("siparis_no", "size"),
        kucuk_adet=("adet", "sum"),
    )

    merged = pd.merge(toplam, kucuk, on="ay", how="left").fillna(0)

    merged["kucuk_satir"] = merged["kucuk_satir"].astype(int)
    merged["kucuk_adet"] = merged["kucuk_adet"].astype(int)

    merged["satir_oran"] = (
        merged["kucuk_satir"] / merged["toplam_satir"] * 100
    ).round(1)

    merged["adet_oran"] = (
        merged["kucuk_adet"] / merged["toplam_adet"] * 100
    ).round(1)

    return merged.sort_values("ay")


def small_order_load_chart(monthly_load: pd.DataFrame, secilen_boy: int):
    if monthly_load.empty:
        return None

    fig = px.line(
        monthly_load,
        x="ay",
        y="adet_oran",
        markers=True,
        title=f"{secilen_boy} Boy ve Altı Siparişlerin Aylık Üretim Payı (%)",
        hover_data=["toplam_adet", "kucuk_adet", "satir_oran", "toplam_satir", "kucuk_satir"],
    )
    fig.update_layout(height=400)
    return fig

def dashboard_monthly_chart(monthly_df: pd.DataFrame):
    if monthly_df.empty:
        return None

    fig = px.line(
        monthly_df,
        x="ay",
        y="toplam_adet",
        markers=True,
        title="Günlük / Aylık Üretim Grafiği (Aylık Toplam Boy)",
        hover_data=["toplam_kg", "siparis_sayisi"],
    )
    fig.update_layout(height=420)
    return fig

def dashboard_pres_performance_chart(scope_df: pd.DataFrame):
    if scope_df.empty or "pres" not in scope_df.columns:
        return None

    pres_df = scope_df.groupby("pres", as_index=False).agg(
        toplam_adet=("adet", "sum"),
        toplam_kg=("kg", "sum"),
        siparis_sayisi=("siparis_no", pd.Series.nunique),
    )

    pres_df = pres_df.sort_values("toplam_adet", ascending=False)

    fig = px.bar(
        pres_df,
        x="pres",
        y="toplam_adet",
        title="Pres Bazlı Performans",
        text="toplam_adet",
        hover_data=["toplam_kg", "siparis_sayisi"],
    )

    fig.update_layout(height=420)
    return fig

def build_pres_efficiency(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty or "pres" not in scope_df.columns:
        return pd.DataFrame()

    pres_df = scope_df.groupby("pres", as_index=False).agg(
        toplam_adet=("adet", "sum"),
        toplam_kg=("kg", "sum"),
        siparis_sayisi=("siparis_no", pd.Series.nunique),
        satir_sayisi=("siparis_no", "size"),
    )

    pres_df["adet_per_siparis"] = (
        pres_df["toplam_adet"] / pres_df["siparis_sayisi"]
    ).round(2)

    pres_df["kg_per_siparis"] = (
        pres_df["toplam_kg"] / pres_df["siparis_sayisi"]
    ).round(2)

    pres_df["adet_per_satir"] = (
        pres_df["toplam_adet"] / pres_df["satir_sayisi"]
    ).round(2)

    pres_df = pres_df.sort_values("toplam_adet", ascending=False)

    pres_df.columns = [
        "Pres",
        "Toplam Üretim (Boy)",
        "Toplam Kg",
        "Sipariş Sayısı",
        "Satır Sayısı",
        "Sipariş Başına Boy",
        "Sipariş Başına Kg",
        "Satır Başına Boy"
    ]

    return pres_df
    
def dashboard_top_profiles_chart(top_profiles_df: pd.DataFrame):
    if top_profiles_df.empty:
        return None

    fig = px.bar(
        top_profiles_df.sort_values(UI_COLS["toplam_uretim"]),
        x=UI_COLS["toplam_uretim"],
        y="Profil Kodu",
        orientation="h",
        title="En Çok Üretilen Profil",
        text=UI_COLS["toplam_uretim"],
        hover_data=["Toplam Kg", "Sipariş Sayısı"],
    )
    fig.update_layout(height=420, yaxis={"automargin": True})
    return fig


def dashboard_termin_chart(termin_df: pd.DataFrame):
    if termin_df.empty or len(termin_df) < 3:
        return None

    try:
        zamaninda = int(str(termin_df.iloc[1]["Değer"]).replace(",", ""))
        geciken = int(str(termin_df.iloc[2]["Değer"]).replace(",", ""))
    except Exception as e:
        print("Termin chart error:", e)
        return None

    fig = px.pie(
        names=["Zamanında", "Geciken"],
        values=[zamaninda, geciken],
        title="Termin Performansı",
    )

    fig.update_layout(height=420)
    return fig

def load_customer_detail(musteri, excel_file, secilen_boy, mod, yillar, profil_ara):
    df = load_excel(excel_file)

    selected_years = [int(str(y)) for y in yillar] if yillar else sorted(df["yil"].unique().tolist())
    scope_df = filter_scope_data(df, selected_years, profil_ara)

    return build_customer_detail(scope_df, musteri, int(secilen_boy))

def summary_markdown(
    filtered: pd.DataFrame,
    scope_df: pd.DataFrame,
    secilen_boy: int,
    mod: str
) -> str:
    if filtered.empty:
        return "### Sonuç\nSeçilen filtrelere göre kayıt bulunamadı."

    exact = filtered[filtered["adet"] == secilen_boy]
    yil_min = int(scope_df["yil"].min())
    yil_max = int(scope_df["yil"].max())

    toplam_scope_satir = len(scope_df)
    toplam_scope_adet = int(scope_df["adet"].sum())

    kucuk_satir = len(filtered)
    kucuk_adet = int(filtered["adet"].sum())
    # 🔧 KALIP ANALİZİ (KALIP ÇEŞİTLİLİĞİ)
    toplam_kalip = scope_df["profil"].nunique()
    kucuk_kalip = filtered["profil"].nunique()
    kalip_oran = (kucuk_kalip / toplam_kalip * 100) if toplam_kalip > 0 else 0
    toplam_sure_saat = (kucuk_kalip * 5) / 60
    # 🔍 KALIP KIRILIM ANALİZİ
    tum_kaliplar = set(scope_df["profil"].unique())
    kucuk_kaliplar = set(filtered["profil"].unique())
    
    buyuk_df = scope_df[scope_df["adet"] > secilen_boy]
    buyuk_kaliplar = set(buyuk_df["profil"].unique())
    
    sadece_kucuk = len(kucuk_kaliplar - buyuk_kaliplar)
    sadece_buyuk = len(buyuk_kaliplar - kucuk_kaliplar)
    ortak = len(kucuk_kaliplar & buyuk_kaliplar)
    # KALIP DEĞİŞİM KODU ÜSTTEKİ 11 SATIR
    buyuk_satir = toplam_scope_satir - kucuk_satir
    buyuk_adet = toplam_scope_adet - kucuk_adet

    satir_yuzde = (kucuk_satir / toplam_scope_satir * 100) if toplam_scope_satir > 0 else 0
    adet_yuzde = (kucuk_adet / toplam_scope_adet * 100) if toplam_scope_adet > 0 else 0

    if satir_yuzde > 20 and adet_yuzde < 10:
        yorum = "⚠️ Çok sayıda küçük sipariş var ancak üretime katkısı düşük (verimsizlik riski)"
    elif satir_yuzde > 20:
        yorum = "📌 Küçük siparişler operasyonel yük oluşturuyor"
    elif adet_yuzde > 40:
        yorum = "📊 Küçük siparişler üretimde önemli paya sahip"
    else:
        yorum = "✅ Sipariş dağılımı dengeli"

    lines = [
        "### Özet Bilgi",
        f"- İncelenen sipariş tipi: **{mod}**",
        f"- Küçük sipariş eşiği: **{secilen_boy} boy**",
        f"- İncelenen tarih aralığı: **{yil_min} - {yil_max}**",
        f"- Toplam kayıt sayısı: **{len(filtered):,}**",
        f"- Farklı sipariş sayısı: **{filtered['siparis_no'].nunique():,}**",
        f"- Farklı profil kodu sayısı: **{filtered['profil'].nunique():,}**",
        f"- Toplam üretilen boy: **{int(filtered['adet'].sum()):,}**",
        f"- Toplam ağırlık: **{filtered['kg'].fillna(0).sum():,.2f} kg**",
        "",
        "### Genel Yük Değerlendirmesi",
        "",
        "### 📦 Sipariş Dağılımı",
        f"- Toplam sipariş: **{toplam_scope_satir:,}**",
        f"  - Küçük sipariş (≤{secilen_boy}): **{kucuk_satir:,}**",
        f"  - Büyük sipariş (>{secilen_boy}): **{buyuk_satir:,}**",
        "",
        "### 🔩 Üretim Dağılımı",
        f"- Toplam kayıt içinde küçük siparişlerin payı: **%{satir_yuzde:.1f}**",
        f"- Toplam üretim içinde küçük siparişlerin payı: **%{adet_yuzde:.1f}**",
        "",
        "### 🔧 Kalıp Analizi",
        f"- Toplam kalıp sayısı: **{toplam_kalip:,}**",
        f"- {secilen_boy} boy altı kalıp sayısı: **{kucuk_kalip:,}**",
        f"- Oranı: **%{kalip_oran:.1f}**",
        f"- Tahmini setup süresi: **{toplam_sure_saat:.1f} saat**",
        "",
        "### 📊 Kalıp Kullanım Dağılımı",
        f"- 🔴 Sadece küçük sipariş: **{sadece_kucuk:,}**",
        f"- 🟢 Sadece büyük sipariş: **{sadece_buyuk:,}**",
        f"- 🟡 Her iki sipariş türü: **{ortak:,}**",
        "",
        "### Genel Yorum Değerlendirme:",
        f"- Yorum: **{yorum}**",
    ]

    if mod == "Seçilen boy ve altı":
        lines += [
            "",
            f"**Tam {secilen_boy} boy olanlar**",
            f"- Sipariş sayısı: **{len(exact):,}**",
            f"- Sipariş no sayısı: **{exact['siparis_no'].nunique():,}**",
            f"- Profil sayısı: **{exact['profil'].nunique():,}**",
            f"- Toplam adet: **{int(exact['adet'].sum()):,}**",
        ]

    return "\n".join(lines)

def never_exceed_summary_markdown(
    never_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    secilen_boy: int
) -> str:
    if never_df.empty:
        return (
            f"### Sonuç\n"
            f"Seçilen filtrelere göre **{secilen_boy} boy üstüne hiç çıkmamış profil** bulunamadı."
        )

    yil_min = int(never_df["yil"].min())
    yil_max = int(never_df["yil"].max())

    toplam_satir = len(never_df)
    toplam_adet = int(never_df["adet"].sum())
    toplam_kg = float(never_df["kg"].fillna(0).sum())
    benzersiz_profil = int(never_df["profil"].nunique())
    # 🔧 KALIP ANALİZİ (Eşiği Aşmayan Profiller)
    toplam_kalip = scope_df["profil"].nunique()
    never_kalip = never_df["profil"].nunique()
    kalip_oran = (never_kalip / toplam_kalip * 100) if toplam_kalip > 0 else 0
    toplam_sure_saat = (never_kalip * 5) / 60
    # KALIP DEĞİŞİM KODU ÜSTTEKİ 4 SATIR
    benzersiz_siparis = int(never_df["siparis_no"].nunique())

    genel_satir = len(scope_df)
    genel_adet = int(scope_df["adet"].sum())

    satir_oran = (toplam_satir / genel_satir * 100) if genel_satir else 0
    adet_oran = (toplam_adet / genel_adet * 100) if genel_adet else 0

    max_boy = int(never_df["adet"].max()) if not never_df.empty else 0
    ort_boy = round(float(never_df["adet"].mean()), 2) if not never_df.empty else 0

    lines = [
        f"## 🚫 {secilen_boy} Boy Üstüne Hiç Çıkmamış Profiller",
        "",
        "### Tanım",
        f"- Bu sekmede sadece geçmişte **asla {secilen_boy} boy üstüne çıkmamış** profiller yer alır.",
        f"- Yani bu listede yer alan profiller, geçmişte bir kez bile **{secilen_boy} boyun üzerine çıkmamıştır**.",
        "",
        "### Özet",
        f"- Tarih aralığı: **{yil_min} - {yil_max}**",
        f"- Farklı profil sayısı: **{benzersiz_profil:,}**",
        "",
        "### 🔧 Kalıp Analizi",
        f"- Toplam kalıp sayısı: **{toplam_kalip:,}**",
        f"- Eşiği aşmayan kalıp sayısı: **{never_kalip:,}**",
        f"- Toplam içindeki oranı: **%{kalip_oran:.1f}**",
        f"- Diğer kalıplar: **{toplam_kalip - never_kalip:,}**",
        f"- Tahmini setup süresi: **{toplam_sure_saat:.1f} saat**",
        "",
        "### Toplam Sonuçlar",
        f"- Farklı sipariş sayısı: **{benzersiz_siparis:,}**",
        f"- Toplam kayıt sayısı: **{toplam_satir:,}**",
        f"- Toplam üretim: **{toplam_adet:,}**",
        f"- Toplam kg: **{toplam_kg:,.2f}**",
        f"- Bu grupta görülen en yüksek sipariş boyu: **{max_boy}**",
        f"- Ortalama sipariş boyu: **{ort_boy}**",
        "",
        "### Genel İçindeki Payı",
        f"- Toplam satır içindeki payı: **%{satir_oran:.1f}**",
        f"- Toplam üretim içindeki payı: **%{adet_oran:.1f}**",
        "",
        "### Yönetim Yorumu",
        f"- Bunlar gerçekten **{secilen_boy} boy üstüne hiç çıkmamış** profillerdir.",
        "- Bu yüzden aksiyon kararı için klasik küçük sipariş listesinden daha temiz bir segment sunar.",
    ]

    return "\n".join(lines)
    
def analyze(excel_file, secilen_boy, mod, yillar, profil_ara, hedef_uretim, top_n_sec, hedef_kucuk_oran):
    try:
        df = load_excel(excel_file)
        # 🚀 ANALİZ CACHE
        analysis_cache_path = os.path.join(CACHE_DIR, f"analysis_{hash(str([secilen_boy, mod, yillar, profil_ara, hedef_uretim]))}.pkl")
        
        if os.path.exists(analysis_cache_path):
            try:
                return pd.read_pickle(analysis_cache_path)
            except:
                pass
    except Exception as e:
        raise gr.Error(f"Excel yüklenemedi: {str(e)}")
    selected_years = [int(str(y)) for y in yillar] if yillar else sorted(df["yil"].unique().tolist())

    scope_df = filter_scope_data(df, selected_years, profil_ara)
    never_exceed_df = filter_never_exceed_profiles(scope_df, int(secilen_boy), profil_ara="")
    filtered = filter_data(scope_df, int(secilen_boy), mod, "")

    hedef_uretim = int(hedef_uretim)
    top_n_value = int(top_n_sec)

    summary_small_text = summary_markdown(filtered, scope_df, int(secilen_boy), mod)
    boy_df = build_boy_breakdown(filtered, int(secilen_boy))
    profile_df = build_profile_summary(filtered, hedef_uretim)
    year_df = build_year_summary(filtered)
    monthly_load_df = build_small_order_monthly(scope_df, int(secilen_boy))
    never_summary_text = never_exceed_summary_markdown(never_exceed_df, scope_df, int(secilen_boy))
    never_boy_df = build_boy_breakdown(never_exceed_df, int(secilen_boy))
    never_profile_df = build_profile_summary(never_exceed_df, hedef_uretim)
    never_year_df = build_year_summary(never_exceed_df)
    
    never_raw_cols = ["tarih", "firma_adi", "siparis_no", "musteri_siparis_no", "profil", "adet", "kg"]
    never_raw_df = never_exceed_df[never_raw_cols].sort_values("tarih", ascending=False).copy() if not never_exceed_df.empty else pd.DataFrame(columns=never_raw_cols)
    
    if not never_raw_df.empty:
        never_raw_df["tarih"] = never_raw_df["tarih"].dt.strftime("%Y-%m-%d")

    never_chart_boy = boy_breakdown_chart(never_boy_df)
    never_chart_profile = top_profiles_chart(never_profile_df, top_n_value)
    never_chart_monthly = monthly_chart(never_exceed_df)

    high_md = high_volume_summary_markdown(scope_df, int(secilen_boy))
    high_profile_df = build_high_volume_profile_summary(scope_df, int(secilen_boy), hedef_uretim)
    high_year_df = build_high_volume_year_summary(scope_df, int(secilen_boy))
    high_raw_df = build_high_volume_raw(scope_df, int(secilen_boy))

    abc_df = build_abc_analysis(scope_df, hedef_uretim)
    abc_md = abc_summary_markdown(abc_df, scope_df)
    profit_df = build_profit_simulation(scope_df)
    root_musteri_df, root_profil_df, root_pres_df = build_root_cause(scope_df, int(secilen_boy))
    dashboard_kpi_df = build_dashboard_kpis(scope_df)
    dashboard_monthly_df = build_dashboard_monthly(scope_df)
    dashboard_top_profiles_df = build_dashboard_top_profiles(scope_df, top_n=15)
    dashboard_termin_df = build_termin_dashboard(scope_df)    
    dashboard_pres_eff_df = build_pres_efficiency(scope_df)
    seasonality_df = build_seasonality_table(scope_df)
    year_month_pivot_df = build_year_month_pivot(scope_df)
    forecast_df = build_forecast_table(scope_df)
    scenario_df = build_scenario_table(scope_df, int(secilen_boy), hedef_kucuk_oran)
    scenario_md = scenario_summary_markdown(scope_df, int(secilen_boy), hedef_kucuk_oran)
    
    raw_cols = ["tarih", "firma_adi", "siparis_no", "musteri_siparis_no", "profil", "adet", "kg"]
    raw = filtered[raw_cols].sort_values("tarih", ascending=False).copy()
    raw["tarih"] = raw["tarih"].dt.strftime("%Y-%m-%d")

    profile_list = profile_df[UI_COLS["profil"]].tolist() if not profile_df.empty else []
    
    customer_mapping = load_customer_mapping()
    musteri_list = sorted(customer_mapping.keys())
    exec_summary = build_executive_summary(
        scope_df,
        abc_df,
        int(secilen_boy),
        hedef_kucuk_oran
    )

    result = (
        summary_small_text,
        boy_df,
        year_df,
        profile_df,
        raw.head(500),
        monthly_load_df,
        boy_breakdown_chart(boy_df),
        top_profiles_chart(profile_df, top_n_value),
        monthly_chart(filtered),
        small_order_load_chart(monthly_load_df, int(secilen_boy)),
    
        # 🔥 kritik eksik blok (şimdi eklendi)
        never_summary_text,
        never_boy_df,
        never_year_df,
        never_profile_df,
        never_raw_df.head(500),
        never_chart_boy,
        never_chart_profile,
        never_chart_monthly,
    
        high_md,
        high_year_df,
        high_profile_df,
        high_raw_df,
        high_volume_chart(high_profile_df, top_n_value),
    
        abc_md,
        abc_df,
        abc_chart(abc_df, top_n_value),
    
        gr.update(choices=profile_list, value=profile_list[0] if profile_list else ""),
        gr.update(choices=musteri_list, value=musteri_list[0] if musteri_list else ""),
    
        exec_summary,
        dashboard_kpi_df,
        dashboard_monthly_df,
        dashboard_top_profiles_df,
        dashboard_termin_df,
        dashboard_monthly_chart(dashboard_monthly_df),
        dashboard_pres_performance_chart(scope_df),
        dashboard_top_profiles_chart(dashboard_top_profiles_df),
        dashboard_termin_chart(dashboard_termin_df),
        dashboard_pres_eff_df,
        seasonality_df,
        year_month_pivot_df,
        seasonality_chart(seasonality_df),
        moving_average_chart(dashboard_monthly_df),
    
        profit_df,
        root_musteri_df,
        root_profil_df,
    
        forecast_df,
        forecast_chart(forecast_df),
    
        scenario_md,
        scenario_df
    )
    
    # 🚀 cache yaz
    try:
        pd.to_pickle(result, analysis_cache_path)
    except:
        pass
    
    return result
    
def load_profile_detail(profil, excel_file, secilen_boy, mod, yillar):
    df = load_excel(excel_file)
    selected_years = [int(str(y)) for y in yillar] if yillar else sorted(df["yil"].unique().tolist())
    filtered = filter_data(df, int(secilen_boy), mod, "")

    yearly, boy_dist, toplam, siparis = build_profile_detail(filtered, profil, selected_years)

    summary = f"""
    ### Profil Özeti
    
    - Toplam üretim: **{toplam} boy**
      - Bu profil için analiz döneminde gerçekleşen toplam üretim miktarıdır.
    - Farklı sipariş sayısı: **{siparis}**
      - Bu profilin kaç ayrı siparişte geçtiğini gösterir.
    """

    return yearly, boy_dist, summary


def years_from_file(excel_file):
    df = load_excel(excel_file)
    years = sorted(df["yil"].unique().tolist())
    return gr.update(choices=years, value=years)

# PDF KODLARININ BAŞLANĞIÇ YERİ AŞAĞISI
def clean_markdown_for_pdf(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    text = text.replace("•", "-")
    text = re.sub(r"^[-*]\s*", "• ", text, flags=re.MULTILINE)
    text = text.replace("\n", "<br/>")
    return text


def export_plotly_figure(fig, filename: str) -> str:
    if fig is None:
        return ""
    path = os.path.join(REPORT_DIR, filename)
    try:
        fig.write_image(path, width=1400, height=800, scale=2)
        return path
    except Exception:
        return ""


def build_pdf_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=18,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748B"),
        ),
    }


def dataframe_to_pdf_table(df: pd.DataFrame, max_rows: int = 20, col_widths=None):
    if df is None or df.empty:
        data = [["Bilgi", "Kayıt bulunamadı"]]
    else:
        limited = df.head(max_rows).copy()
        data = [list(limited.columns)] + limited.fillna("").astype(str).values.tolist()

    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def add_pdf_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.5)
    canvas.line(15 * mm, height - 12 * mm, width - 15 * mm, height - 12 * mm)
    canvas.line(15 * mm, 12 * mm, width - 15 * mm, 12 * mm)

    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(15 * mm, height - 9 * mm, "Üretim Analiz ve Karar Destek Platformu")

    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(
        width - 15 * mm,
        height - 9 * mm,
        datetime.now().strftime("Rapor tarihi: %d.%m.%Y %H:%M")
    )

    canvas.drawString(15 * mm, 7 * mm, "Oluşturulan çıktı: Yönetici PDF raporu")
    canvas.drawRightString(width - 15 * mm, 7 * mm, f"Sayfa {canvas.getPageNumber()}")
    canvas.restoreState()


def generate_professional_pdf(
    excel_file,
    secilen_boy,
    mod,
    yillar,
    profil_ara,
    hedef_uretim,
    top_n_sec,
    hedef_kucuk_oran
):
    if excel_file is None:
        raise gr.Error("Önce Excel dosyası yükleyin.")

    try:
        df = load_excel(excel_file)
    except Exception as e:
        raise gr.Error(f"PDF üretilemedi: {str(e)}")

    selected_years = [int(str(y)) for y in yillar] if yillar else sorted(df["yil"].unique().tolist())
    scope_df = filter_scope_data(df, selected_years, profil_ara)
    filtered = filter_data(scope_df, int(secilen_boy), mod, profil_ara)

    hedef_uretim = int(hedef_uretim)
    top_n_value = int(top_n_sec)

    summary_small_text = summary_markdown(filtered, scope_df, int(secilen_boy), mod)
    high_md = high_volume_summary_markdown(scope_df, int(secilen_boy))
    abc_df = build_abc_analysis(scope_df, hedef_uretim)
    abc_md = abc_summary_markdown(abc_df, scope_df)

    exec_summary = build_executive_summary(
        scope_df,
        abc_df,
        int(secilen_boy),
        hedef_kucuk_oran
    )

    dashboard_kpi_df = build_dashboard_kpis(scope_df)
    dashboard_monthly_df = build_dashboard_monthly(scope_df)
    dashboard_top_profiles_df = build_dashboard_top_profiles(scope_df, top_n=15)
    dashboard_termin_df = build_termin_dashboard(scope_df)
    forecast_df = build_forecast_table(scope_df)
    scenario_df = build_scenario_table(scope_df, int(secilen_boy), hedef_kucuk_oran)
    scenario_md = scenario_summary_markdown(scope_df, int(secilen_boy), hedef_kucuk_oran)
    profile_df = build_profile_summary(filtered, hedef_uretim)

    monthly_fig = dashboard_monthly_chart(dashboard_monthly_df)
    abc_fig = abc_chart(abc_df, top_n_value)
    small_order_fig = small_order_load_chart(
        build_small_order_monthly(scope_df, int(secilen_boy)),
        int(secilen_boy)
    )
    top_profiles_fig = dashboard_top_profiles_chart(dashboard_top_profiles_df)
    termin_fig = dashboard_termin_chart(dashboard_termin_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(REPORT_DIR, f"uretim_analiz_raporu_{ts}.pdf")
    styles = build_pdf_styles()

    chart_paths = [
        export_plotly_figure(monthly_fig, f"monthly_{ts}.png"),
        export_plotly_figure(abc_fig, f"abc_{ts}.png"),
        export_plotly_figure(small_order_fig, f"small_order_{ts}.png"),
        export_plotly_figure(top_profiles_fig, f"top_profiles_{ts}.png"),
        export_plotly_figure(termin_fig, f"termin_{ts}.png"),
    ]

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    yil_text = ", ".join(str(y) for y in selected_years) if selected_years else "Tüm yıllar"
    profil_text = profil_ara if str(profil_ara).strip() else "Tüm profiller"

    elements = []

    elements.append(Spacer(1, 18 * mm))
    elements.append(Paragraph("Profesyonel Üretim Analiz Raporu", styles["title"]))
    elements.append(Paragraph(
        "Küçük sipariş yükü, ABC sınıflandırması, yönetici özeti, tahmin ve senaryo analizi",
        styles["subtitle"]
    ))

    cover_info = pd.DataFrame([
        ["Rapor Kapsamı", "Operasyon + Yönetim Karar Desteği"],
        ["Seçilen Boy", str(secilen_boy)],
        ["Analiz Modu", str(mod)],
        ["Yıllar", yil_text],
        ["Profil Filtresi", profil_text],
        ["Yıllık Üretim Frekansı", str(hedef_uretim)],
        ["Hedef Küçük Sipariş Oranı", f"%{float(hedef_kucuk_oran):.0f}"],
    ], columns=["Parametre", "Değer"])

    elements.append(dataframe_to_pdf_table(
        cover_info,
        max_rows=20,
        col_widths=[55 * mm, 110 * mm]
    ))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(clean_markdown_for_pdf(exec_summary), styles["body"]))
    elements.append(PageBreak())

    elements.append(Paragraph("1. Yönetici Özeti", styles["section"]))
    elements.append(Paragraph(clean_markdown_for_pdf(exec_summary), styles["body"]))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph("2. KPI Özeti", styles["section"]))
    elements.append(dataframe_to_pdf_table(
        dashboard_kpi_df,
        max_rows=20,
        col_widths=[70 * mm, 40 * mm]
    ))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("3. Küçük Sipariş Özeti", styles["section"]))
    elements.append(Paragraph(clean_markdown_for_pdf(summary_small_text), styles["body"]))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("4. Büyük Sipariş Özeti", styles["section"]))
    elements.append(Paragraph(clean_markdown_for_pdf(high_md), styles["body"]))
    elements.append(PageBreak())

    elements.append(Paragraph("5. ABC Analizi ve Stok Önerisi", styles["section"]))
    elements.append(Paragraph(clean_markdown_for_pdf(abc_md), styles["body"]))
    elements.append(dataframe_to_pdf_table(abc_df, max_rows=15))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("6. Senaryo Özeti", styles["section"]))
    elements.append(Paragraph(clean_markdown_for_pdf(scenario_md), styles["body"]))
    elements.append(dataframe_to_pdf_table(
        scenario_df,
        max_rows=15,
        col_widths=[95 * mm, 55 * mm]
    ))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("7. Kritik Profil Özeti", styles["section"]))
    elements.append(dataframe_to_pdf_table(profile_df, max_rows=15))
    elements.append(PageBreak())

    elements.append(Paragraph("8. Grafikler", styles["section"]))
    chart_titles = [
        "Aylık Üretim Trendi",
        "ABC Analizi",
        "Küçük Siparişlerin Üretim Payı",
        "En Çok Üretilen Profiller",
        "Termin Performansı",
    ]

    for title, path in zip(chart_titles, chart_paths):
        if path and os.path.exists(path):
            elements.append(Paragraph(title, styles["section"]))
            elements.append(Image(path, width=175 * mm, height=95 * mm))
            elements.append(Spacer(1, 8))

    elements.append(PageBreak())
    elements.append(Paragraph("9. Dashboard Tabloları", styles["section"]))

    elements.append(Paragraph("Aylık Dashboard Verisi", styles["section"]))
    elements.append(dataframe_to_pdf_table(dashboard_monthly_df, max_rows=18))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Top Profil Listesi", styles["section"]))
    elements.append(dataframe_to_pdf_table(dashboard_top_profiles_df, max_rows=15))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Termin Özeti", styles["section"]))
    elements.append(dataframe_to_pdf_table(
        dashboard_termin_df,
        max_rows=10,
        col_widths=[80 * mm, 35 * mm]
    ))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Forecast Özeti", styles["section"]))
    elements.append(dataframe_to_pdf_table(forecast_df, max_rows=15))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Not", styles["section"]))
    elements.append(Paragraph(
        "Bu rapor uygulamadaki seçili filtrelere göre otomatik oluşturulmuştur. "
        "Grafiklerin görüntülenebilmesi için sunucuda kaleido kurulu olmalıdır.",
        styles["small"]
    ))

    try:
        doc.build(
            elements,
            onFirstPage=add_pdf_header_footer,
            onLaterPages=add_pdf_header_footer
        )
    except Exception as e:
        raise gr.Error(f"PDF oluşturulamadı: {str(e)}")

    return pdf_path
#PDF KODLARININ BİTİŞ YERİ KODLAR YUKARISI

with gr.Blocks(
    title="Alüminyum Sipariş Boy Analizi",
) as demo:

    # 🔥 STICKY PANEL CSS
    gr.HTML("""
    <style>
    #side-panel {
        position: sticky;
        top: 10px;
    }
    </style>
    """)

    # ✅ HEADER
    gr.HTML("""
    <div style="display:flex; align-items:center; gap:16px;">
        <img src="https://raw.githubusercontent.com/Khejah/analiz-app/main/ic_asistal_a_fab.png" style="height:100px;">
        
        <div>
            <h1 style="margin:0;">
                🏭 Üretim Analiz ve Karar Destek Platformu
            </h1>
            
            <p style="margin:4px 0; font-size:14px; color:#444;">
                Üretim verilerini anlamlı içgörülere dönüştürün, 
                operasyonel verimliliği artırın ve doğru stok kararlarını veriyle yönetin.
            </p>
            
            <p style="font-size:13px; color:#777; margin:4px 0;">
                ✔ Küçük Sipariş Analizi  
                ✔ ABC Stok Modeli  
                ✔ Talep Tahmini & Senaryo Analizi  
                ✔ Yönetici Dashboard
            </p>
        </div>
    </div>
    """)

    # 🔻 ANA LAYOUT (DÜZELTİLMİŞ)
    with gr.Row():

        # ✅ SOL → ANA CONTENT
        with gr.Column(scale=3):

            with gr.Tabs():

                with gr.Tab("📉 Küçük Sipariş Analizi (Operasyonel Yük)"):
                    summary_small = gr.Markdown()
                    gr.Markdown("""
                    ### 📌 Bu ekran neyi gösterir?
                    
                    Bu bölüm küçük siparişlerin üretim üzerindeki etkisini analiz eder.
                    
                    - Çok sayıda küçük sipariş → planlama yükü oluşturur  
                    - Düşük üretim katkısı → verimsizlik göstergesidir  
                    
                    Amaç: operasyonel yükü azaltmak
                    """)

                    with gr.Column():
                        chart1 = gr.Plot(label="📊 Sipariş Boy Dağılımı")
                        chart2 = gr.Plot(label="🏆 En Sık Sipariş Edilen Profiller")
                        chart3 = gr.Plot(label="📈 Sipariş Yoğunluk Trendi")
                        chart4 = gr.Plot(label="⚠️ Küçük Siparişlerin Üretime Etkisi")
                        
                    with gr.Tabs():
                        with gr.Tab("Boy Kırılımı"):
                            boy_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Yıl Özeti"):
                            year_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Profil Özeti"):
                            profile_table = gr.Dataframe(interactive=False, wrap=True)

                            gr.Markdown("## 🔍 Profil Detay")
                            profil_sec = gr.Dropdown(label="🔍 Detay İncelenecek Profil", choices=[])

                            detail_summary = gr.Markdown()
                            detail_year = gr.Dataframe(label="Yıllık Detay")
                            detail_boy = gr.Dataframe(label="Boy Dağılımı")

                        with gr.Tab("Aylık Yük Analizi"):
                            monthly_load_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Ham Kayıt Önizleme"):
                            raw_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("🚫 Eşiği Hiç Aşmamış Profiller"):
                    never_summary_md = gr.Markdown()
                
                    with gr.Row():
                        never_chart1 = gr.Plot(label="📊 Sipariş Boy Dağılımı")
                        never_chart2 = gr.Plot(label="🏆 Profil Yoğunluğu")
                        never_chart3 = gr.Plot(label="📈 Zaman Bazlı Trend")
                
                    with gr.Tabs():
                        with gr.Tab("Boy Kırılımı"):
                            never_boy_table = gr.Dataframe(interactive=False, wrap=True)
                
                        with gr.Tab("Yıl Özeti"):
                            never_year_table = gr.Dataframe(interactive=False, wrap=True)
                
                        with gr.Tab("Profil Özeti"):
                            never_profile_table = gr.Dataframe(interactive=False, wrap=True)
                
                        with gr.Tab("Ham Kayıt Önizleme"):
                            never_raw_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("📈 Büyük Sipariş Analizi (Core Üretim)"):
                    high_summary = gr.Markdown()
                    high_chart = gr.Plot(label="En Çok Üretime Giren Profiller")

                    with gr.Tabs():
                        with gr.Tab("Yıl Özeti"):
                            high_year_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Profil Özeti"):
                            high_profile_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Ham Kayıt Önizleme"):
                            high_raw_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("📦 ABC Analizi ve Stok Önerisi"):
                    gr.Markdown("""
                    ### 📌 ABC Analizi
                    
                    - 🟢 A → Kritik (stok yapılmalı)  
                    - 🟡 B → Planlı üretim  
                    - 🔴 C → Sipariş bazlı üretim  
                    
                    Amaç: stok ve üretim stratejisini optimize etmek
                    """)
                    abc_summary = gr.Markdown()
                    abc_plot = gr.Plot(label="ABC Analizi")
                    abc_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("🧠 Yönetici Özeti"):
                    gr.Markdown("""
                    ### 📌 Bu ekran ne sağlar?
                    
                    Tüm analizlerin özetini tek ekranda sunar.
                    
                    ✔ Verimlilik  
                    ✔ Kalıp değişim etkisi  
                    ✔ Kritik profiller  
                    ✔ Aksiyon önerileri  
                    """)
                    exec_summary_md = gr.Markdown()

                with gr.Tab("🔍 Kök Neden Analizi"):
                    root_musteri_table = gr.Dataframe(label="Müşteri Analizi")
                    root_profil_table = gr.Dataframe(label="Profil Analizi")

                with gr.Tab("👤 Müşteri Analizi"):
            
                    musteri_sec = gr.Dropdown(label="👤 Analiz Edilecek Müşteri", choices=[])
                
                    musteri_ozet = gr.Dataframe(label="Müşteri Özeti")
                    musteri_yorum = gr.Markdown()

                with gr.Tab("📊 Yönetim Dashboard"):
                    gr.Markdown("## 📊 Yönetim Dashboard")
                    gr.Markdown("""
                    ### 📌 Genel Durum
                    
                    Bu ekran üretimin genel performansını gösterir:
                    
                    - Üretim hacmi  
                    - Sipariş yoğunluğu  
                    - Pres performansı  
                    - Termin uyumu  
                    """)

                    dashboard_kpi_table = gr.Dataframe(label="KPI Özeti", interactive=False, wrap=True)

                    with gr.Row():
                        dashboard_chart_monthly = gr.Plot(label="Günlük / Aylık Üretim")
                        dashboard_chart_pres = gr.Plot(label="Pres Bazlı Performans")

                    with gr.Row():
                        dashboard_chart_profiles = gr.Plot(label="En Çok Üretilen Profil")
                        dashboard_chart_termin = gr.Plot(label="Termin Uyum Oranı")

                    with gr.Tabs():
                        with gr.Tab("Aylık Dashboard Verisi"):
                            dashboard_monthly_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Top Profil Listesi"):
                            dashboard_profiles_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Pres Verimlilik"):
                            dashboard_pres_eff_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Termin Özeti"):
                            dashboard_termin_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Sezonsallık"):
                            dashboard_seasonality_table = gr.Dataframe(interactive=False, wrap=True)
                            dashboard_seasonality_chart = gr.Plot(label="Sezonsallık Grafiği")

                        with gr.Tab("Yıl-Ay Kırılımı"):
                            dashboard_year_month_pivot_table = gr.Dataframe(interactive=False, wrap=True)
                            dashboard_moving_avg_chart = gr.Plot(label="Hareketli Ortalama")

                        with gr.Tab("Stratejik Karar"):
                            profit_table = gr.Dataframe(interactive=False, wrap=True)

                        with gr.Tab("Forecast Engine"):
                            forecast_table = gr.Dataframe(interactive=False, wrap=True)
                            forecast_plot = gr.Plot(label="Forecast Grafiği")
                        
                        with gr.Tab("Scenario Engine"):
                            scenario_md_box = gr.Markdown()
                            scenario_table = gr.Dataframe(interactive=False, wrap=True)

        # ✅ SAĞ → STICKY PANEL
        with gr.Column(scale=1, elem_id="side-panel"):

            gr.Markdown("""
            ### ⚙️ Analiz Parametreleri
            
            Lütfen analiz etmek istediğiniz veri ve kriterleri seçin.
            Sistem bu parametrelere göre otomatik içgörü üretir.
            """)

            excel_file = gr.File(label="📂 Veri Dosyası Yükle")
            
            secilen_boy = gr.Dropdown(
                label="🎯 Hedef Sipariş Boyu",
                choices=[str(i) for i in range(10, 0, -1)],
                value="10"
            )

            mod = gr.Dropdown(
                label="⚙️ Analiz Türü",
                choices=["Seçilen boy", "Seçilen boy ve altı"],
                value="Seçilen boy ve altı"
            )

            years = gr.CheckboxGroup(label="📅 Analiz Dönemi", choices=[])

            profil_ara = gr.Textbox(label="🔎 Profil Filtre", placeholder="Örn: TH62-01")

            hedef_uretim = gr.Dropdown(
                label="🏭 Yıllık Üretim Frekansı",
                choices=["4", "6", "12"],
                value="4"
            )

            top_n_sec = gr.Dropdown(
                label="📊 Grafik Detay Seviyesi",
                choices=["15", "50", "100"],
                value="15"
            )

            hedef_kucuk_oran = gr.Slider(
                label="🎯 Hedef Küçük Sipariş Oranı (%)",
                minimum=1,
                maximum=30,
                value=10,
                step=1
            )

            load_btn = gr.Button("📅 Yılları Yükle")
            analyze_btn = gr.Button("🚀 Analizi Başlat", variant="primary")
            pdf_btn = gr.Button("📄 Profesyonel PDF Oluştur")
            pdf_output = gr.File(label="Hazır PDF Raporu")

    load_btn.click(fn=years_from_file, inputs=excel_file, outputs=years)
    
    analyze_btn.click(
        fn=analyze,
        inputs=[excel_file, secilen_boy, mod, years, profil_ara, hedef_uretim, top_n_sec, hedef_kucuk_oran],
        outputs=[
            summary_small,
            boy_table,
            year_table,
            profile_table,
            raw_table,
            monthly_load_table,
            chart1,
            chart2,
            chart3,
            chart4,
            never_summary_md,
            never_boy_table,
            never_year_table,
            never_profile_table,
            never_raw_table,
            never_chart1,
            never_chart2,
            never_chart3,
            high_summary,
            high_year_table,
            high_profile_table,
            high_raw_table,
            high_chart,
            abc_summary,
            abc_table,
            abc_plot,
            profil_sec,
            musteri_sec,
            exec_summary_md,
            dashboard_kpi_table,
            dashboard_monthly_table,
            dashboard_profiles_table,
            dashboard_termin_table,
            dashboard_chart_monthly,
            dashboard_chart_pres,
            dashboard_chart_profiles,
            dashboard_chart_termin,
            dashboard_pres_eff_table,
            dashboard_seasonality_table,
            dashboard_year_month_pivot_table,
            dashboard_seasonality_chart,
            dashboard_moving_avg_chart,
            profit_table,
            root_musteri_table,
            root_profil_table,
            forecast_table,
            forecast_plot,
            scenario_md_box,
            scenario_table
        ],
    )

    pdf_btn.click(
        fn=generate_professional_pdf,
        inputs=[
            excel_file,
            secilen_boy,
            mod,
            years,
            profil_ara,
            hedef_uretim,
            top_n_sec,
            hedef_kucuk_oran
        ],
        outputs=pdf_output
    )

    profil_sec.change(
        fn=load_profile_detail,
        inputs=[profil_sec, excel_file, secilen_boy, mod, years],
        outputs=[detail_year, detail_boy, detail_summary]
    )
    
    musteri_sec.change(
        fn=load_customer_detail,
        inputs=[musteri_sec, excel_file, secilen_boy, mod, years, profil_ara],
        outputs=[musteri_ozet, musteri_yorum]
    )

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
        favicon_path="favicon.png",
        show_error=True
    )
