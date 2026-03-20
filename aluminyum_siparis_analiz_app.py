from typing import Iterable, Optional

import gradio as gr
import pandas as pd
import plotly.express as px

COLUMN_ALIASES = {
    "tarih": ["Tarih", "TARIH", "date"],
    "profil": ["Profil No", "Profil", "profil", "profil_kodu"],
    "siparis_no": ["Siparis No", "Sipariş No", "siparis_no", "order_no"],
    "musteri": ["Mus.Siparis No", "Müşteri Sipariş No", "Musteri", "Müşteri", "mus_siparis_no"],
    "adet": ["Adet", "adet", "Boy", "boy_adedi"],
    "kg": ["Kg", "kg"],
    "firma": ["Firma Adi", "Firma", "firma_adi"],
    "pres": ["Pres Adi", "Pres", "pres"],
    "termin": ["Termin"],
    "termin_hafta": ["Termin Hafta"],    
}


def normalize_col(s: str) -> str:
    return str(s).strip().lower().replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")


NORMALIZED_ALIAS_MAP = {
    key: {normalize_col(x) for x in vals} for key, vals in COLUMN_ALIASES.items()
}


def detect_header_row(excel_file, sheet_name: Optional[str] = None, max_rows: int = 15) -> int:
    preview = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=max_rows)
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
    for col in df.columns:
        if normalize_col(col) in aliases:
            return col
    return None


def load_excel(excel_file) -> pd.DataFrame:
    if excel_file is None:
        raise ValueError("Lütfen bir Excel dosyası yükleyin.")

    try:
        xls = pd.ExcelFile(excel_file)
    except Exception as e:
        raise ValueError(f"Excel okunamadı: {str(e)}")

    sheet_name = xls.sheet_names[0]
    header_row = detect_header_row(excel_file, sheet_name=sheet_name)
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
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
        "profil": df[profil_col].astype(str).str.strip(),
        "siparis_no": df[siparis_col].astype(str).str.strip(),
        "adet": pd.to_numeric(df[adet_col], errors="coerce"),
    })

    if musteri_col:
        work["musteri_siparis_no"] = df[musteri_col].astype(str).str.strip()
    else:
        work["musteri_siparis_no"] = ""

    if firma_col:
        work["firma_adi"] = df[firma_col].astype(str).str.strip()
    else:
        work["firma_adi"] = ""

    if kg_col:
        work["kg"] = pd.to_numeric(df[kg_col], errors="coerce")
    else:
        work["kg"] = 0
    # PRES
    if pres_col:
        work["pres"] = df[pres_col].astype(str).str.strip()
    else:
        work["pres"] = "Bilinmiyor"
    
    # TERMIN
    if termin_col:
        work["termin"] = pd.to_datetime(df[termin_col], errors="coerce", dayfirst=True)
    else:
        work["termin"] = pd.NaT
    
    # TERMIN HAFTA
    if termin_hafta_col:
        work["termin_hafta"] = df[termin_hafta_col]
    else:
        work["termin_hafta"] = None    
        
    work = work.dropna(subset=["tarih", "profil", "siparis_no", "adet"]).copy()
    work = work[(work["adet"] >= 1) & (work["adet"] <= 100000)]
    work["adet"] = work["adet"].astype(int)
    work["yil"] = work["tarih"].dt.year.astype(int)
    work["ay"] = work["tarih"].dt.to_period("M").astype(str)
    return work


def filter_data(df: pd.DataFrame, secilen_boy: int, mod: str, yillar: Iterable[int], profil_ara: str = "") -> pd.DataFrame:
    filtered = df.copy()
    if yillar:
        filtered = filtered[filtered["yil"].isin([int(y) for y in yillar])]

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
        filtered = filtered[filtered["yil"].isin([int(y) for y in yillar])]

    profil_ara = (profil_ara or "").strip().upper()
    if profil_ara:
        filtered = filtered[filtered["profil"].str.upper().str.contains(profil_ara, na=False)]

    return filtered


def build_boy_breakdown(filtered: pd.DataFrame, secilen_boy: int) -> pd.DataFrame:
    rows = []

    for boy in range(secilen_boy, 0, -1):
        part = filtered[filtered["adet"] == boy]

        rows.append({
            "Boy": boy,
            "Toplam Sipariş Kalemi": int(len(part)),
            "Farklı Sipariş Sayısı": int(part["siparis_no"].nunique()),
            "Farklı Profil Sayısı": int(part["profil"].nunique()),
            "Toplam Üretilen Boy": int(part["adet"].sum()) if not part.empty else 0,
            "Toplam Kg": float(round(part["kg"].fillna(0).sum(), 2)),
        })

    return pd.DataFrame(rows)


def build_profile_detail(df: pd.DataFrame, profil_kodu: str, yillar):
    data = df[df["profil"] == profil_kodu].copy()

    if yillar:
        data = data[data["yil"].isin([int(y) for y in yillar])]

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
            "Profil Kodu",
            "Toplam Sipariş Kalemi",
            "Farklı Sipariş Sayısı",
            "Toplam Üretilen Boy",
            "İlk Sipariş Tarihi",
            "Son Sipariş Tarihi",
            "Sipariş Başına Ortalama Boy",
            "Yıllık Tüketim",
            "Yeni Akıllı Öneri (Boy)"
        ])

    profile = filtered.groupby("profil", as_index=False).agg(
        toplam_siparis_kalemi=("siparis_no", "size"),
        farkli_siparis_sayisi=("siparis_no", pd.Series.nunique),
        toplam_uretilen_boy=("adet", "sum"),
        ilk_tarih=("tarih", "min"),
        son_tarih=("tarih", "max"),
    )

    profile["siparis_basina_ortalama_boy"] = (
        profile["toplam_uretilen_boy"] / profile["farkli_siparis_sayisi"]
    ).round(2)

    profile["ilk_tarih"] = profile["ilk_tarih"].dt.strftime("%Y-%m-%d")
    profile["son_tarih"] = profile["son_tarih"].dt.strftime("%Y-%m-%d")

    yil_sayisi = max(filtered["yil"].nunique(), 1)

    profile["Yıllık Tüketim"] = (
        profile["toplam_uretilen_boy"] / yil_sayisi
    ).round(0)

    profile["Yeni Akıllı Öneri (Boy)"] = (
        profile["Yıllık Tüketim"] / hedef_uretim
    ).round(0)

    profile.columns = [
        "Profil Kodu",
        "Toplam Sipariş Kalemi",
        "Farklı Sipariş Sayısı",
        "Toplam Üretilen Boy",
        "İlk Sipariş Tarihi",
        "Son Sipariş Tarihi",
        "Sipariş Başına Ortalama Boy",
        "Yıllık Tüketim",
        "Yeni Akıllı Öneri (Boy)"
    ]

    return profile.sort_values(
        ["Toplam Sipariş Kalemi", "Toplam Üretilen Boy"],
        ascending=[False, False]
    )


def build_high_volume_profile_summary(scope_df: pd.DataFrame, min_boy: int, hedef_uretim: int) -> pd.DataFrame:
    filtered = scope_df[scope_df["adet"] >= min_boy].copy()

    if filtered.empty:
        return pd.DataFrame(columns=[
            "Profil Kodu",
            "Toplam Sipariş Kalemi",
            "Farklı Sipariş Sayısı",
            "Toplam Üretilen Boy",
            "İlk Sipariş Tarihi",
            "Son Sipariş Tarihi",
            "Sipariş Başına Ortalama Boy",
            "Yıllık Tüketim",
            "Yeni Akıllı Öneri (Boy)"
        ])

    profile = filtered.groupby("profil", as_index=False).agg(
        toplam_siparis_kalemi=("siparis_no", "size"),
        farkli_siparis_sayisi=("siparis_no", pd.Series.nunique),
        toplam_uretilen_boy=("adet", "sum"),
        ilk_tarih=("tarih", "min"),
        son_tarih=("tarih", "max"),
    )

    profile["siparis_basina_ortalama_boy"] = (
        profile["toplam_uretilen_boy"] / profile["farkli_siparis_sayisi"]
    ).round(2)

    profile["ilk_tarih"] = profile["ilk_tarih"].dt.strftime("%Y-%m-%d")
    profile["son_tarih"] = profile["son_tarih"].dt.strftime("%Y-%m-%d")

    yil_sayisi = max(filtered["yil"].nunique(), 1)

    profile["Yıllık Tüketim"] = (
        profile["toplam_uretilen_boy"] / yil_sayisi
    ).round(0)

    profile["Yeni Akıllı Öneri (Boy)"] = (
        profile["Yıllık Tüketim"] / hedef_uretim
    ).round(0)

    profile.columns = [
        "Profil Kodu",
        "Toplam Sipariş Kalemi",
        "Farklı Sipariş Sayısı",
        "Toplam Üretilen Boy",
        "İlk Sipariş Tarihi",
        "Son Sipariş Tarihi",
        "Sipariş Başına Ortalama Boy",
        "Yıllık Tüketim",
        "Yeni Akıllı Öneri (Boy)"
    ]

    return profile.sort_values(
        ["Toplam Üretilen Boy", "Toplam Sipariş Kalemi"],
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
        {"KPI": "Toplam Sipariş Satırı", "Değer": f"{toplam_satir:,}"},
        {"KPI": "Benzersiz Sipariş", "Değer": f"{benzersiz_siparis:,}"},
        {"KPI": "Benzersiz Profil", "Değer": f"{benzersiz_profil:,}"},
        {"KPI": "Toplam Üretilen Boy", "Değer": f"{toplam_adet:,}"},
        {"KPI": "Toplam Kg", "Değer": f"{toplam_kg:,.2f}"},
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
        return pd.DataFrame(columns=["Profil Kodu", "Toplam Üretilen Boy", "Toplam Kg", "Sipariş Sayısı"])

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

    prof.columns = ["Profil Kodu", "Toplam Üretilen Boy", "Toplam Kg", "Sipariş Sayısı"]
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
        f"- Toplam sipariş satırı: **{toplam_satir:,}**",
        f"- Benzersiz sipariş sayısı: **{toplam_siparis:,}**",
        f"- Benzersiz profil sayısı: **{toplam_profil:,}**",
        f"- Toplam üretilen boy: **{toplam_adet:,}**",
        f"- Toplam kg: **{toplam_kg:,.2f}**",
        f"- Ortalama sipariş boyu: **{ort_boy:,}**",
        f"- En çok üretime giren profil: **{lider_profil}**",
        f"- Bu profilin toplam üretimi: **{lider_adet:,} boy**",
    ]

    return "\n".join(lines)


def build_abc_analysis(scope_df: pd.DataFrame, hedef_uretim: int) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame(columns=[
            "Profil Kodu",
            "Toplam Üretilen Boy",
            "Toplam Sipariş Kalemi",
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

    profile["Yıllık Tüketim"] = (profile["toplam_uretilen_boy"] / yil_sayisi).round(0)
    profile["Yeni Akıllı Öneri (Boy)"] = (profile["Yıllık Tüketim"] / hedef_uretim).round(0)

    if toplam_genel > 0:
        profile["pay"] = profile["toplam_uretilen_boy"] / toplam_genel * 100
        profile["Kümülatif Pay (%)"] = profile["pay"].cumsum().round(2)
    else:
        profile["pay"] = 0
        profile["Kümülatif Pay (%)"] = 0

    def abc_label(kum_pay):
        if kum_pay <= 80:
            return "A"
        elif kum_pay <= 95:
            return "B"
        return "C"

    profile["ABC Sınıfı"] = profile["Kümülatif Pay (%)"].apply(abc_label)

    def stok_onerisi(row):
        if row["ABC Sınıfı"] == "A":
            return "Evet"
        elif row["ABC Sınıfı"] == "B":
            return "Planlı Üret"
        return "Hayır"

    profile["Stok Önerisi"] = profile.apply(stok_onerisi, axis=1)

    profile = profile[[
        "profil",
        "toplam_uretilen_boy",
        "toplam_siparis_kalemi",
        "farkli_siparis_sayisi",
        "Yıllık Tüketim",
        "Yeni Akıllı Öneri (Boy)",
        "Kümülatif Pay (%)",
        "ABC Sınıfı",
        "Stok Önerisi"
    ]]

    profile.columns = [
        "Profil Kodu",
        "Toplam Üretilen Boy",
        "Toplam Sipariş Kalemi",
        "Farklı Sipariş Sayısı",
        "Yıllık Tüketim",
        "Yeni Akıllı Öneri (Boy)",
        "Kümülatif Pay (%)",
        "ABC Sınıfı",
        "Stok Önerisi"
    ]

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
    df["ort_boy"] = (df["toplam_adet"] / df["siparis_sayisi"]).round(2)

    # Yoğunluk skoru (çok sipariş ama küçükse kötü)
    df["yogunluk_skor"] = (df["satir_sayisi"] / df["toplam_adet"]).round(4)

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
    
def abc_summary_markdown(abc_df: pd.DataFrame) -> str:
    if abc_df.empty:
        return "### Sonuç\nABC analizi için kayıt bulunamadı."

    a_sayisi = int((abc_df["ABC Sınıfı"] == "A").sum())
    b_sayisi = int((abc_df["ABC Sınıfı"] == "B").sum())
    c_sayisi = int((abc_df["ABC Sınıfı"] == "C").sum())

    a_toplam = int(abc_df[abc_df["ABC Sınıfı"] == "A"]["Toplam Üretilen Boy"].sum())
    b_toplam = int(abc_df[abc_df["ABC Sınıfı"] == "B"]["Toplam Üretilen Boy"].sum())
    c_toplam = int(abc_df[abc_df["ABC Sınıfı"] == "C"]["Toplam Üretilen Boy"].sum())

    stok_adet = int((abc_df["Stok Önerisi"] == "Evet").sum())

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
        "### Yorum",
        "- **A grubu**: stok yapılmalı",
        "- **B grubu**: planlı / parti üretim yapılmalı",
        "- **C grubu**: sipariş gelmeden stok yapılmamalı",
    ]
    return "\n".join(lines)

def build_executive_summary(scope_df, filtered, abc_df, secilen_boy, hedef_uretim, hedef_kucuk_oran):
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
    toplam_kalip_suresi_saat = kalip_degisim_sayisi * 1
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
    
    # boş ve hatalı kayıtları temizle
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
    a_class = abc_df[abc_df["ABC Sınıfı"] == "A"].head(10)
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
    "📎 Not: Kalıp değişim süresi ortalama **1 saat** olarak varsayılmıştır.",
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

    for i, (idx, row) in enumerate(top_profiles.iterrows(), 1):
        lines.append(f"{i}. {idx} → {int(row['toplam']):,} boy")

    lines.append("")
    lines.append("## 📦 A Sınıfı (Stok Önerilen)")
    
    for _, row in a_class.iterrows():
        lines.append(
            f"- {row['Profil Kodu']} → aylık: {int(row['Yıllık Tüketim']/12):,} / yıllık: {int(row['Yıllık Tüketim']):,}"
        )

    lines.append("")
    lines.append("## 👤 Müşteri Yük Analizi")

    for i, (idx, row) in enumerate(musteri.iterrows(), 1):
        yuzde = (row["toplam"] / toplam_adet * 100) if toplam_adet else 0
        lines.append(
            f"{i}. {idx} → {int(row['siparis'])} sipariş | "
            f"{int(row['toplam'])} boy | %{yuzde:.1f} üretim"
        )

    return "\n".join(lines)
    
def abc_chart(abc_df: pd.DataFrame, top_n_value: int):
    if abc_df.empty:
        return None

    top_n = abc_df.head(top_n_value).sort_values("Toplam Üretilen Boy")

    fig = px.bar(
        top_n,
        x="Toplam Üretilen Boy",
        y="Profil Kodu",
        orientation="h",
        color="ABC Sınıfı",
        title=f"ABC Analizi - En Yüksek Tüketimli Profiller (ilk {top_n_value})",
        text="Toplam Üretilen Boy",
        hover_data=["Toplam Sipariş Kalemi", "Yıllık Tüketim", "Stok Önerisi", "Kümülatif Pay (%)"]
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
    return year.sort_values("yil")


def top_profiles_chart(profile_summary: pd.DataFrame, top_n_value: int):
    if profile_summary.empty:
        return None

    top_n = profile_summary.head(top_n_value).sort_values("Toplam Sipariş Kalemi")
    grafik_yuksekligi = max(500, top_n_value * 35)

    fig = px.bar(
        top_n,
        x="Toplam Sipariş Kalemi",
        y="Profil Kodu",
        orientation="h",
        title=f"En sık geçen profiller (ilk {top_n_value})",
        text="Toplam Sipariş Kalemi",
        hover_data=["Farklı Sipariş Sayısı", "Toplam Üretilen Boy"],
    )

    fig.update_layout(
        height=grafik_yuksekligi,
        yaxis={"automargin": True}
    )
    return fig

def high_volume_chart(profile_summary: pd.DataFrame, top_n_value: int):
    if profile_summary.empty:
        return None

    top_n = profile_summary.head(top_n_value).sort_values("Toplam Üretilen Boy")

    grafik_yuksekligi = max(500, top_n_value * 35)

    fig = px.bar(
        top_n,
        x="Toplam Üretilen Boy",
        y="Profil Kodu",
        orientation="h",
        title=f"En çok üretime giren profiller (ilk {top_n_value})",
        text="Toplam Üretilen Boy",
        hover_data=["Toplam Sipariş Kalemi", "Farklı Sipariş Sayısı", "Yıllık Tüketim"],
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
        boy_breakdown.sort_values("Boy"),
        x="Boy",
        y="Toplam Sipariş Kalemi",
        title="Boylara Göre Sipariş Dağılımı",
        text="Toplam Sipariş Kalemi",
        hover_data=[
            "Farklı Sipariş Sayısı",
            "Farklı Profil Sayısı",
            "Toplam Üretilen Boy",
            "Toplam Kg"
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
        top_profiles_df.sort_values("Toplam Üretilen Boy"),
        x="Toplam Üretilen Boy",
        y="Profil Kodu",
        orientation="h",
        title="En Çok Üretilen Profil",
        text="Toplam Üretilen Boy",
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
        f"- Filtre modu: **{mod}**",
        f"- Seçilen üst sınır / hedef boy: **{secilen_boy}**",
        f"- Tarih aralığı: **{yil_min} - {yil_max}**",
        f"- Toplam sipariş satırı: **{len(filtered):,}**",
        f"- Benzersiz sipariş no: **{filtered['siparis_no'].nunique():,}**",
        f"- Benzersiz profil: **{filtered['profil'].nunique():,}**",
        f"- Toplam adet: **{int(filtered['adet'].sum()):,}**",
        f"- Toplam kg: **{filtered['kg'].fillna(0).sum():,.2f}**",
        "",
        "### Genel Yük Değerlendirmesi",
        "",
        "### 📦 Sipariş Dağılımı",
        f"- Toplam satır: **{toplam_scope_satir:,}**",
        f"  - Küçük sipariş (≤{secilen_boy}): **{kucuk_satir:,}**",
        f"  - Büyük sipariş (>{secilen_boy}): **{buyuk_satir:,}**",
        "",
        "### 🔩 Üretim Dağılımı",
        f"- Toplam üretim: **{toplam_scope_adet:,}**",
        f"  - Küçük sipariş üretimi: **{kucuk_adet:,}**",
        f"  - Büyük sipariş üretimi: **{buyuk_adet:,}**",
        f"- Kapsamdaki toplam sipariş satırı: **{toplam_scope_satir:,}**",
        f"- {secilen_boy} boy ve altı satırlar, toplam listenin **%{satir_yuzde:.1f}**'ini oluşturuyor",
        f"- Kapsamdaki toplam üretim adedi: **{toplam_scope_adet:,}**",
        f"- {secilen_boy} boy ve altı ürünler, toplam üretimin **%{adet_yuzde:.1f}**'ini oluşturuyor",
        f"- Yorum: **{yorum}**",
    ]

    if mod == "Seçilen boy ve altı":
        lines += [
            "",
            f"**Tam {secilen_boy} boy olanlar**",
            f"- Satır sayısı: **{len(exact):,}**",
            f"- Sipariş no sayısı: **{exact['siparis_no'].nunique():,}**",
            f"- Profil sayısı: **{exact['profil'].nunique():,}**",
            f"- Toplam adet: **{int(exact['adet'].sum()):,}**",
        ]

    return "\n".join(lines)


def analyze(excel_file, secilen_boy, mod, yillar, profil_ara, hedef_uretim, top_n_sec, hedef_kucuk_oran):
    df = load_excel(excel_file)
    selected_years = [int(y) for y in yillar] if yillar else sorted(df["yil"].unique().tolist())

    scope_df = filter_scope_data(df, selected_years, profil_ara)
    filtered = filter_data(df, int(secilen_boy), mod, selected_years, profil_ara)

    hedef_uretim = int(hedef_uretim)
    top_n_value = min(int(top_n_sec), 50)

    summary_small_text = summary_markdown(filtered, scope_df, int(secilen_boy), mod)
    boy_df = build_boy_breakdown(filtered, int(secilen_boy))
    profile_df = build_profile_summary(filtered, hedef_uretim)
    year_df = build_year_summary(filtered)
    monthly_load_df = build_small_order_monthly(scope_df, int(secilen_boy))

    high_md = high_volume_summary_markdown(scope_df, int(secilen_boy))
    high_profile_df = build_high_volume_profile_summary(scope_df, int(secilen_boy), hedef_uretim)
    high_year_df = build_high_volume_year_summary(scope_df, int(secilen_boy))
    high_raw_df = build_high_volume_raw(scope_df, int(secilen_boy))

    abc_df = build_abc_analysis(scope_df, hedef_uretim)
    abc_md = abc_summary_markdown(abc_df)
    profit_df = build_profit_simulation(scope_df)
    dashboard_kpi_df = build_dashboard_kpis(scope_df)
    dashboard_monthly_df = build_dashboard_monthly(scope_df)
    dashboard_top_profiles_df = build_dashboard_top_profiles(scope_df, top_n=15)
    dashboard_termin_df = build_termin_dashboard(scope_df)    
    dashboard_pres_eff_df = build_pres_efficiency(scope_df)
    seasonality_df = build_seasonality_table(scope_df)
    year_month_pivot_df = build_year_month_pivot(scope_df)
    
    raw_cols = ["tarih", "firma_adi", "siparis_no", "musteri_siparis_no", "profil", "adet", "kg"]
    raw = filtered[raw_cols].sort_values("tarih", ascending=False).copy()
    raw["tarih"] = raw["tarih"].dt.strftime("%Y-%m-%d")

    profile_list = profile_df["Profil Kodu"].tolist() if not profile_df.empty else []
    exec_summary = build_executive_summary(
        scope_df,
        filtered,
        abc_df,
        int(secilen_boy),
        hedef_uretim,
        hedef_kucuk_oran
    )
    
    return (
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
        high_md,
        high_year_df,
        high_profile_df,
        high_raw_df,
        high_volume_chart(high_profile_df, top_n_value),
        abc_md,
        abc_df,
        abc_chart(abc_df, top_n_value),
        gr.update(choices=profile_list, value=profile_list[0] if profile_list else None),
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
        profit_df
    )


def load_profile_detail(profil, excel_file, secilen_boy, mod, yillar):
    df = load_excel(excel_file)
    selected_years = [int(y) for y in yillar] if yillar else sorted(df["yil"].unique().tolist())
    filtered = filter_data(df, int(secilen_boy), mod, selected_years, "")

    yearly, boy_dist, toplam, siparis = build_profile_detail(filtered, profil, selected_years)

    summary = f"""
### Profil Özeti

- Toplam tüketim: **{toplam} boy**
- Sipariş sayısı: **{siparis}**
"""

    return yearly, boy_dist, summary


def years_from_file(excel_file):
    df = load_excel(excel_file)
    years = sorted(df["yil"].unique().tolist())
    return gr.update(choices=years, value=years)


with gr.Blocks(title="Alüminyum Sipariş Boy Analizi") as demo:
    gr.Markdown("# Alüminyum Sipariş Boy Analizi")
    gr.Markdown(
        "Excel dosyasını yükleyin. Ardından küçük siparişleri, büyük siparişleri ve ABC stok önerisini birlikte analiz edin."
    )

    with gr.Row():
        excel_file = gr.File(label="Excel dosyası", file_types=[".xlsx", ".xls"])
        secilen_boy = gr.Dropdown(label="Boy seç", choices=[str(i) for i in range(10, 0, -1)], value="10")
        mod = gr.Dropdown(label="Filtre modu", choices=["Seçilen boy", "Seçilen boy ve altı"], value="Seçilen boy ve altı")
        years = gr.CheckboxGroup(label="Yıllar (önce yükle)", choices=[])
        profil_ara = gr.Textbox(label="Profil ara (opsiyonel)", placeholder="Örn: TH62-01")
        hedef_uretim = gr.Dropdown(
            label="Yılda Kaç Kez Üretim Yapılsın?",
            choices=["4", "6", "12"],
            value="6"
        )
        top_n_sec = gr.Dropdown(
            label="Grafiklerde kaç profil gösterilsin?",
            choices=["15", "50", "100"],
            value="15"
        )
        
        hedef_kucuk_oran = gr.Slider(
            label="Simülasyon: Küçük sipariş oranı (%)",
            minimum=1,
            maximum=30,
            value=10,
            step=1
        )

    with gr.Row():
        load_btn = gr.Button("Yılları yükle")
        analyze_btn = gr.Button("Analizi çalıştır", variant="primary")

    with gr.Tabs():
        with gr.Tab("En Az Üretime Giren Ürünlerin Listesi"):
            summary_small = gr.Markdown()

            with gr.Row():
                chart1 = gr.Plot(label="Boy dağılımı")
                chart2 = gr.Plot(label="Top profiller")

            chart3 = gr.Plot(label="Aylık trend")
            chart4 = gr.Plot(label="Küçük Boy Sipariş Yükü")

            with gr.Tabs():
                with gr.Tab("Boy Kırılımı"):
                    boy_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("Yıl Özeti"):
                    year_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("Profil Özeti"):
                    profile_table = gr.Dataframe(interactive=False, wrap=True)

                    gr.Markdown("## 🔍 Profil Detay")
                    profil_sec = gr.Dropdown(label="Profil seç", choices=[])

                    detail_summary = gr.Markdown()
                    detail_year = gr.Dataframe(label="Yıllık Detay")
                    detail_boy = gr.Dataframe(label="Boy Dağılımı")

                with gr.Tab("Aylık Yük Analizi"):
                    monthly_load_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("Ham Kayıt Önizleme"):
                    raw_table = gr.Dataframe(interactive=False, wrap=True)

        with gr.Tab("En Çok Üretime Giren Ürünlerin Listesi"):
            high_summary = gr.Markdown()
            high_chart = gr.Plot(label="En Çok Üretime Giren Profiller")

            with gr.Tabs():
                with gr.Tab("Yıl Özeti"):
                    high_year_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("Profil Özeti"):
                    high_profile_table = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("Ham Kayıt Önizleme"):
                    high_raw_table = gr.Dataframe(interactive=False, wrap=True)

        with gr.Tab("ABC Analizi ve Stok Önerisi"):
            abc_summary = gr.Markdown()
            abc_plot = gr.Plot(label="ABC Analizi")
            abc_table = gr.Dataframe(interactive=False, wrap=True)

        with gr.Tab("Yönetici Özeti"):
            exec_summary_md = gr.Markdown()
        with gr.Tab("Yönetim Dashboard"):
            gr.Markdown("## 📊 Yönetim Dashboard")

            dashboard_kpi_table = gr.Dataframe(
                label="KPI Özeti",
                interactive=False,
                wrap=True
            )

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
            high_summary,
            high_year_table,
            high_profile_table,
            high_raw_table,
            high_chart,
            abc_summary,
            abc_table,
            abc_plot,
            profil_sec,
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
            profit_table
        ],
    )

    profil_sec.change(
        fn=load_profile_detail,
        inputs=[profil_sec, excel_file, secilen_boy, mod, years],
        outputs=[detail_year, detail_boy, detail_summary]
    )

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        theme=gr.themes.Soft()
    )
