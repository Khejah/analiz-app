import math
from pathlib import Path
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
}


def normalize_col(s: str) -> str:
    return str(s).strip().lower().replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")


NORMALIZED_ALIAS_MAP = {
    key: {normalize_col(x) for x in vals} for key, vals in COLUMN_ALIASES.items()
}


def detect_header_row(excel_path: str, sheet_name: Optional[str] = None, max_rows: int = 15) -> int:
    preview = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, nrows=max_rows)
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

    excel_path = excel_file.name if hasattr(excel_file, "name") else str(excel_file)
    xls = pd.ExcelFile(excel_path)
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

    # Yıllık özet
    yearly = data.groupby("yil").agg(
        toplam_boy=("adet", "sum"),
        siparis_sayisi=("siparis_no", "nunique")
    ).reset_index()

    # Boy dağılımı
    boy_dist = data.groupby("adet").agg(
        kac_siparis=("siparis_no", "count"),
        toplam_boy=("adet", "sum")
    ).reset_index().sort_values("adet", ascending=False)

    # Genel özet
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
            "Sipariş Başına Ortalama Boy",
            "İlk Sipariş Tarihi",
            "Son Sipariş Tarihi"
        ])

    profile = filtered.groupby("profil", as_index=False).agg(
        toplam_siparis_kalemi=("siparis_no", "size"),
        farkli_siparis_sayisi=("siparis_no", pd.Series.nunique),
        toplam_uretilen_boy=("adet", "sum"),
        ilk_tarih=("tarih", "min"),
        son_tarih=("tarih", "max"),
    )

    # Ortalama Hesaplama
    profile["siparis_basina_ortalama_boy"] = (
        profile["toplam_uretilen_boy"] / profile["farkli_siparis_sayisi"]
    ).round(2)

    # Tarih formatı
    profile["ilk_tarih"] = profile["ilk_tarih"].dt.strftime("%Y-%m-%d")
    profile["son_tarih"] = profile["son_tarih"].dt.strftime("%Y-%m-%d")
    
    # yıl sayısı
    yil_sayisi = max(filtered["yil"].nunique(), 1)

    # yıllık tüketim
    profile["Yıllık Tüketim"] = (
        profile["toplam_uretilen_boy"] / yil_sayisi
    ).round(0)

    # kullanıcıya göre akıllı lot
    profile["Yeni Akıllı Öneri (Boy)"] = (
        profile["Yıllık Tüketim"] / hedef_uretim
    ).round(0)
    
    # Kolon isimlerini kullanıcı dili yap
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



def summary_markdown(filtered: pd.DataFrame, secilen_boy: int, mod: str) -> str:
    if filtered.empty:
        return "### Sonuç\nSeçilen filtrelere göre kayıt bulunamadı."

    exact = filtered[filtered["adet"] == secilen_boy]
    yil_min = int(filtered["yil"].min())
    yil_max = int(filtered["yil"].max())

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



def analyze(excel_file, secilen_boy, mod, yillar, profil_ara, hedef_uretim, top_n_sec):
    df = load_excel(excel_file)
    selected_years = [int(y) for y in yillar] if yillar else sorted(df["yil"].unique().tolist())
    filtered = filter_data(df, int(secilen_boy), mod, selected_years, profil_ara)

    md = summary_markdown(filtered, int(secilen_boy), mod)
    boy_df = build_boy_breakdown(filtered, int(secilen_boy))
    hedef_uretim = int(hedef_uretim)
    top_n_value = int(top_n_sec)
    profile_df = build_profile_summary(filtered, hedef_uretim)
    year_df = build_year_summary(filtered)
    
    raw_cols = ["tarih", "firma_adi", "siparis_no", "musteri_siparis_no", "profil", "adet", "kg"]
    raw = filtered[raw_cols].sort_values("tarih", ascending=False).copy()
    raw["tarih"] = raw["tarih"].dt.strftime("%Y-%m-%d")

    profile_list = profile_df["Profil Kodu"].tolist()

    return (
        md,
        boy_df,
        year_df,
        profile_df,
        raw.head(500),
        boy_breakdown_chart(boy_df),
        top_profiles_chart(profile_df, top_n_value),
        monthly_chart(filtered),
        gr.update(choices=profile_list, value=profile_list[0] if profile_list else None)
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


with gr.Blocks(title="Alüminyum Sipariş Boy Analizi", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Alüminyum Sipariş Boy Analizi")
    gr.Markdown(
        "Excel dosyasını yükleyin. Ardından 10'dan 1'e kadar boy seçip, "
        "yalnızca o boyu ya da o boy ve altındaki tüm siparişleri analiz edin."
    )

    with gr.Row():
        excel_file = gr.File(label="Excel dosyası", file_types=[".xlsx", ".xls"])
        secilen_boy = gr.Dropdown(label="Boy seç", choices=[str(i) for i in range(10, 0, -1)], value="10")
        mod = gr.Dropdown(label="Filtre modu", choices=["Seçilen boy", "Seçilen boy ve altı"], value="Seçilen boy ve altı")
        years = gr.CheckboxGroup(label="Yıllar", choices=[])
        profil_ara = gr.Textbox(label="Profil ara (opsiyonel)", placeholder="Örn: LS60 veya TH62")
        hedef_uretim = gr.Dropdown(
            label="Yılda Kaç Kez Üretim Yapılsın?",
            choices=["4", "6", "12"],
            value="6"
        )
        top_n_sec = gr.Dropdown(
            label="Top profiller grafiğinde kaç profil gösterilsin?",
            choices=["15", "50", "100"],
            value="15"
        )
    
    with gr.Row():
        load_btn = gr.Button("Yılları yükle")
        analyze_btn = gr.Button("Analizi çalıştır", variant="primary")

    summary = gr.Markdown()
    with gr.Row():
        chart1 = gr.Plot(label="Boy dağılımı")
        chart2 = gr.Plot(label="Top profiller")
    chart3 = gr.Plot(label="Aylık trend")

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
        
        with gr.Tab("Ham Kayıt Önizleme"):
            raw_table = gr.Dataframe(interactive=False, wrap=True)

    load_btn.click(fn=years_from_file, inputs=excel_file, outputs=years)
    analyze_btn.click(
        fn=analyze,
        inputs=[excel_file, secilen_boy, mod, years, profil_ara, hedef_uretim, top_n_sec],
        outputs=[summary, boy_table, year_table, profile_table, raw_table, chart1, chart2, chart3, profil_sec],
    )
    
    profil_sec.change(
        fn=load_profile_detail,
        inputs=[profil_sec, excel_file, secilen_boy, mod, years],
        outputs=[detail_year, detail_boy, detail_summary]
    )

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
