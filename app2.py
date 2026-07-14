import os
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.title("Kalkulator Gizi Makanan")
# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

NUTRISI_COLS = [
    'kalori_kkal', 'protein_g', 'lemak_g', 'karbo_g',
    'serat_g', 'vitamin_c_mg', 'kalsium_mg',
    'zat_besi_mg', 'vitamin_a_mcg', 'sodium_mg',
]

LABEL_NUTRISI = {
    'kalori_kkal'  : 'Kalori (kkal)',
    'protein_g'    : 'Protein (g)',
    'lemak_g'      : 'Lemak (g)',
    'karbo_g'      : 'Karbohidrat (g)',
    'serat_g'      : 'Serat (g)',
    'vitamin_c_mg' : 'Vitamin C (mg)',
    'kalsium_mg'   : 'Kalsium (mg)',
    'zat_besi_mg'  : 'Zat Besi (mg)',
    'vitamin_a_mcg': 'Vitamin A (mcg)',
    'sodium_mg'    : 'Sodium (mg)',
}


@st.cache_data(show_spinner="Memuat database makanan...")
def load_all_data() -> pd.DataFrame:
    """Load dan merge 4 file CSV USDA menjadi 1 dataframe lengkap."""
    food_df = pd.read_csv(
        os.path.join(DATA_DIR, 'food.csv'),
        usecols=['fdc_id', 'description', 'food_category_id']
    )
    category_df = pd.read_csv(
        os.path.join(DATA_DIR, 'food_category.csv'),
        usecols=['id', 'description']
    ).rename(columns={'id': 'food_category_id', 'description': 'category_name'})

    nutrient_df = pd.read_csv(
        os.path.join(DATA_DIR, 'nutrient.csv'),
        usecols=['id', 'name', 'unit_name']
    ).rename(columns={'id': 'nutrient_id'})

    food_nutrient_df = pd.read_csv(
        os.path.join(DATA_DIR, 'food_nutrient.csv'),
        usecols=['fdc_id', 'nutrient_id', 'amount']
    )

    food_merged    = food_df.merge(category_df, on='food_category_id', how='left')
    nutrient_merged = food_nutrient_df.merge(nutrient_df, on='nutrient_id', how='left')

    NUTRISI_UTAMA = [
        'Energy', 'Protein', 'Total lipid (fat)',
        'Carbohydrate, by difference', 'Fiber, total dietary',
        'Sugars, total including NLEA', 'Calcium, Ca', 'Iron, Fe',
        'Sodium, Na', 'Vitamin C, total ascorbic acid',
        'Vitamin A, RAE', 'Fatty acids, total saturated',
    ]
    nutrient_filtered = nutrient_merged[nutrient_merged['name'].isin(NUTRISI_UTAMA)]

    pivot = nutrient_filtered.pivot_table(
        index='fdc_id', columns='name', values='amount', aggfunc='mean'
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={
        'Energy'                        : 'kalori_kkal',
        'Protein'                       : 'protein_g',
        'Total lipid (fat)'             : 'lemak_g',
        'Carbohydrate, by difference'   : 'karbo_g',
        'Fiber, total dietary'          : 'serat_g',
        'Sugars, total including NLEA'  : 'gula_g',
        'Calcium, Ca'                   : 'kalsium_mg',
        'Iron, Fe'                      : 'zat_besi_mg',
        'Sodium, Na'                    : 'sodium_mg',
        'Vitamin C, total ascorbic acid': 'vitamin_c_mg',
        'Vitamin A, RAE'                : 'vitamin_a_mcg',
        'Fatty acids, total saturated'  : 'lemak_jenuh_g',
    }, inplace=True)

    final_df = food_merged.merge(pivot, on='fdc_id', how='inner')
    cols_fill = [c for c in NUTRISI_COLS + ['gula_g', 'lemak_jenuh_g'] if c in final_df.columns]
    final_df[cols_fill] = final_df[cols_fill].fillna(0)
    return final_df


def search_food(df: pd.DataFrame, keyword: str, top_n: int = 20) -> pd.DataFrame:
    if not keyword.strip():
        return pd.DataFrame()
    mask = df['description'].str.contains(keyword.strip(), case=False, na=False)
    return df[mask].head(top_n).copy()


# ══════════════════════════════════════════════════════════════════
# BAGIAN 2 — CALCULATOR
# ══════════════════════════════════════════════════════════════════
FAKTOR_AKTIVITAS = {
    "Jarang gerak/duduk terus"                          : 1.2,
    "Aktivitas ringan (olahraga 1-3 kali/minggu)"       : 1.375,
    "Aktivitas sedang (olahraga 3-5 kali/minggu)"       : 1.55,
    "Aktivitas berat (olahraga 6-7 kali/minggu)"        : 1.725,
    "Aktivitas sangat berat (olahraga intens 2x sehari)": 1.9,
}


def hitung_bmr(jenis_kelamin: str, berat: float, tinggi: float, umur: int) -> float:
    base = (10 * berat) + (6.25 * tinggi) - (5 * umur)
    return base + 5 if jenis_kelamin == "Pria" else base - 161


def hitung_tdee(bmr: float, aktivitas: str) -> float:
    return bmr * FAKTOR_AKTIVITAS.get(aktivitas, 1.2)


def hitung_target_kalori(tdee: float, tujuan: str) -> float:
    adj = {"Turun berat badan": -500, "Pertahankan berat badan": 0, "Naikkan berat badan": 300}
    return tdee + adj.get(tujuan, 0)


def hitung_kebutuhan_nutrisi(target_kalori: float, jenis_kelamin: str, umur: int) -> dict:
    karbo_g   = (target_kalori * 0.50) / 4
    protein_g = (target_kalori * 0.20) / 4
    lemak_g   = (target_kalori * 0.30) / 9

    if jenis_kelamin == "Pria":
        return {
            "kalori_kkal"  : round(target_kalori, 1),
            "protein_g"    : round(protein_g, 1),
            "lemak_g"      : round(lemak_g, 1),
            "karbo_g"      : round(karbo_g, 1),
            "serat_g"      : 38 if umur <= 50 else 30,
            "vitamin_c_mg" : 90,
            "kalsium_mg"   : 1000,
            "zat_besi_mg"  : 9 if umur >= 18 else 15,
            "vitamin_a_mcg": 600,
            "sodium_mg"    : 1500,
        }
    else:
        return {
            "kalori_kkal"  : round(target_kalori, 1),
            "protein_g"    : round(protein_g, 1),
            "lemak_g"      : round(lemak_g, 1),
            "karbo_g"      : round(karbo_g, 1),
            "serat_g"      : 25 if umur <= 50 else 21,
            "vitamin_c_mg" : 75,
            "kalsium_mg"   : 1000,
            "zat_besi_mg"  : 18 if 18 <= umur <= 50 else 8,
            "vitamin_a_mcg": 500,
            "sodium_mg"    : 1500,
        }


def hitung_total_konsumsi(daftar_makanan: list) -> dict:
    total = {k: 0.0 for k in NUTRISI_COLS}
    for item in daftar_makanan:
        scale = item.get('porsi_gram', 100) / 100
        for key in NUTRISI_COLS:
            total[key] += item.get(key, 0) * scale
    return {k: round(v, 2) for k, v in total.items()}


def hitung_gap(kebutuhan: dict, konsumsi: dict) -> dict:
    return {k: round(kebutuhan[k] - konsumsi.get(k, 0), 2) for k in kebutuhan}


# ══════════════════════════════════════════════════════════════════
# BAGIAN 3 — RECOMMENDER (AI)
# ══════════════════════════════════════════════════════════════════
def rekomendasikan_cosine_similarity(
    df: pd.DataFrame,
    kebutuhan: dict,
    konsumsi: dict,
    exclude_fdc_ids: list = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rekomendasikan makanan menggunakan cosine similarity terhadap gap nutrisi."""
    if exclude_fdc_ids is None:
        exclude_fdc_ids = []

    df_filtered = df[~df['fdc_id'].isin(exclude_fdc_ids)].copy()
    available_cols = [c for c in NUTRISI_COLS if c in df_filtered.columns]

    target = np.array([
        max(kebutuhan.get(k, 0) - konsumsi.get(k, 0), 0)
        for k in available_cols
    ]).reshape(1, -1)

    if target.sum() == 0:
        # Semua nutrisi sudah terpenuhi — kembalikan makanan paling seimbang
        nutrisi_matrix = df_filtered[available_cols].fillna(0).values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(nutrisi_matrix)
        skor = 1 / (1 + np.std(scaled, axis=1))
    else:
        nutrisi_matrix = df_filtered[available_cols].fillna(0).values
        scaler = MinMaxScaler()
        nutrisi_scaled = scaler.fit_transform(nutrisi_matrix)
        target_scaled  = scaler.transform(target)
        skor = cosine_similarity(target_scaled, nutrisi_scaled)[0]

    df_filtered = df_filtered.copy()
    df_filtered['skor_ai'] = skor.round(4)

    return (
        df_filtered
        .sort_values('skor_ai', ascending=False)
        .head(top_n)
        [['fdc_id', 'description', 'category_name'] + available_cols + ['skor_ai']]
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════
# BAGIAN 4 — UI UTAMA
# ══════════════════════════════════════════════════════════════════
# Inisialisasi session state untuk keranjang makanan
if 'keranjang' not in st.session_state:
    st.session_state.keranjang = []

# ── Tab navigasi ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👤 Profil & BMR/TDEE", "🔍 Cari & Tambah Makanan", "📊 Hasil & Rekomendasi AI"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PROFIL USER
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Data Diri")

    jenis_kelamin = st.selectbox("Pilih Jenis Kelamin", ["---", "Pria", "Wanita"])

    if jenis_kelamin == "---":
        st.info("Silahkan pilih jenis kelamin untuk memulai.")
    else:
        st.subheader("Masukkan Informasi Singkat Mengenai Anda")

        col1, col2, col3 = st.columns(3)
        with col1:
            umur = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=25)
        with col2:
            tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170)
        with col3:
            berat_badan = st.number_input("Berat Badan (kg)", min_value=10, max_value=300, value=65)

        bmr  = hitung_bmr(jenis_kelamin, berat_badan, tinggi_badan, umur)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("BMR Anda", f"{bmr:.1f} kkal/hari",
                      help="Kalori yang dibutuhkan tubuh saat istirahat total")

        tingkat_aktivitas = st.selectbox(
            "Tingkat Aktivitas",
            ["---"] + list(FAKTOR_AKTIVITAS.keys())
        )

        tujuan = st.selectbox(
            "Tujuan",
            ["Turun berat badan", "Pertahankan berat badan", "Naikkan berat badan"]
        )

        if tingkat_aktivitas != "---":
            tdee           = hitung_tdee(bmr, tingkat_aktivitas)
            target_kalori  = hitung_target_kalori(tdee, tujuan)
            kebutuhan      = hitung_kebutuhan_nutrisi(target_kalori, jenis_kelamin, umur)

            with col_b:
                st.metric("TDEE Anda", f"{tdee:.1f} kkal/hari",
                          help="Total kalori yang dibutuhkan sesuai aktivitas")

            st.metric("🎯 Target Kalori Harian", f"{target_kalori:.1f} kkal/hari")

            st.divider()
            st.subheader("📋 Kebutuhan Nutrisi Harian Anda")
            k_cols = st.columns(5)
            for i, (key, label) in enumerate(LABEL_NUTRISI.items()):
                with k_cols[i % 5]:
                    st.metric(label, kebutuhan.get(key, 0))

            # Simpan ke session state
            st.session_state['profil'] = {
                'jenis_kelamin' : jenis_kelamin,
                'umur'          : umur,
                'tinggi_badan'  : tinggi_badan,
                'berat_badan'   : berat_badan,
                'aktivitas'     : tingkat_aktivitas,
                'tujuan'        : tujuan,
                'bmr'           : bmr,
                'tdee'          : tdee,
                'target_kalori' : target_kalori,
                'kebutuhan'     : kebutuhan,
            }
            st.success("✅ Profil tersimpan! Lanjut ke tab **Cari & Tambah Makanan**.")
        else:
            st.info("Pilih tingkat aktivitas untuk melihat TDEE dan kebutuhan nutrisi.")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — CARI & TAMBAH MAKANAN
# ══════════════════════════════════════════════════════════════════
with tab2:
    if 'profil' not in st.session_state:
        st.warning("⚠️ Lengkapi profil di tab **Profil & BMR/TDEE** terlebih dahulu.")
    else:
        # Load database
        try:
            df_food = load_all_data()
        except FileNotFoundError as e:
            st.error(f"❌ File CSV tidak ditemukan: {e}\nPastikan folder `data/` berisi keempat file CSV.")
            st.stop()

        st.subheader("🔍 Cari Makanan")
        keyword = st.text_input("Ketik nama makanan (dalam bahasa Inggris)", placeholder="contoh: rice, chicken, egg...")

        if keyword:
            hasil_cari = search_food(df_food, keyword)
            if hasil_cari.empty:
                st.warning("Makanan tidak ditemukan. Coba kata kunci lain.")
            else:
                st.write(f"Ditemukan **{len(hasil_cari)}** makanan:")
                pilihan = st.selectbox(
                    "Pilih makanan",
                    options=hasil_cari['fdc_id'].tolist(),
                    format_func=lambda x: hasil_cari[hasil_cari['fdc_id'] == x]['description'].values[0]
                )

                makanan_terpilih = hasil_cari[hasil_cari['fdc_id'] == pilihan].iloc[0]

                # Tampilkan info nutrisi makanan terpilih
                with st.expander("📊 Detail Nutrisi (per 100g)", expanded=True):
                    d_cols = st.columns(5)
                    for i, (key, label) in enumerate(LABEL_NUTRISI.items()):
                        with d_cols[i % 5]:
                            st.metric(label, f"{makanan_terpilih.get(key, 0):.1f}")

                porsi = st.number_input("Porsi yang dimakan (gram)", min_value=1, max_value=2000, value=100)

                if st.button("➕ Tambahkan ke Daftar Makan"):
                    item = makanan_terpilih.to_dict()
                    item['porsi_gram'] = porsi
                    st.session_state.keranjang.append(item)
                    st.success(f"✅ **{makanan_terpilih['description']}** ({porsi}g) ditambahkan!")

        # Tampilkan keranjang
        st.divider()
        st.subheader("🛒 Daftar Makanan Hari Ini")
        if not st.session_state.keranjang:
            st.info("Belum ada makanan yang ditambahkan.")
        else:
            for i, item in enumerate(st.session_state.keranjang):
                col_nama, col_porsi, col_hapus = st.columns([4, 2, 1])
                with col_nama:
                    st.write(f"🍽️ {item['description']}")
                with col_porsi:
                    st.write(f"{item['porsi_gram']} g")
                with col_hapus:
                    if st.button("🗑️", key=f"hapus_{i}"):
                        st.session_state.keranjang.pop(i)
                        st.rerun()

            if st.button("🗑️ Kosongkan Semua"):
                st.session_state.keranjang = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════
# TAB 3 — HASIL & REKOMENDASI AI
# ══════════════════════════════════════════════════════════════════
with tab3:
    if 'profil' not in st.session_state:
        st.warning("⚠️ Lengkapi profil di tab **Profil & BMR/TDEE** terlebih dahulu.")
    elif not st.session_state.keranjang:
        st.warning("⚠️ Tambahkan makanan di tab **Cari & Tambah Makanan** terlebih dahulu.")
    else:
        kebutuhan = st.session_state['profil']['kebutuhan']
        konsumsi  = hitung_total_konsumsi(st.session_state.keranjang)
        gap       = hitung_gap(kebutuhan, konsumsi)

        # ── Ringkasan konsumsi vs kebutuhan ──────────────────────
        st.subheader("📊 Konsumsi vs Kebutuhan Harian")

        rows = []
        for key, label in LABEL_NUTRISI.items():
            kb  = kebutuhan.get(key, 0)
            kons = konsumsi.get(key, 0)
            pct  = (kons / kb * 100) if kb > 0 else 0
            if gap.get(key, 0) > 0:
                status = "Kurang ⚠️"
            elif gap.get(key, 0) < 0:
                status = "Berlebih ❌"
            else:
                status = "Cukup ✅"
            rows.append({
                "Nutrisi"       : label,
                "Konsumsi"      : f"{kons:.1f}",
                "Kebutuhan"     : f"{kb:.1f}",
                "Persentase (%)" : f"{pct:.1f}%",
                "Status"        : status,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Progress bar per nutrisi
        st.subheader("📈 Progress Nutrisi")
        for key, label in LABEL_NUTRISI.items():
            kb   = kebutuhan.get(key, 0)
            kons = konsumsi.get(key, 0)
            pct  = min(kons / kb, 1.0) if kb > 0 else 0
            color = "normal" if pct >= 0.9 else ("off" if pct > 1.0 else "normal")
            col_l, col_bar = st.columns([2, 5])
            with col_l:
                st.write(label)
            with col_bar:
                st.progress(pct, text=f"{kons:.1f} / {kb:.1f}")

        # ── Rekomendasi AI ────────────────────────────────────────
        st.divider()
        st.subheader("🤖 Rekomendasi Makanan AI")
        st.caption("Menggunakan Cosine Similarity untuk mencari makanan yang paling tepat menutup kekurangan nutrisi Anda.")

        try:
            df_food = load_all_data()
            exclude = [item['fdc_id'] for item in st.session_state.keranjang]
            rekomendasi = rekomendasikan_cosine_similarity(
                df_food, kebutuhan, konsumsi,
                exclude_fdc_ids=exclude, top_n=10
            )

            for _, row in rekomendasi.iterrows():
                with st.expander(f"🍽️ {row['description']}  |  Kategori: {row.get('category_name', '-')}  |  Skor AI: {row['skor_ai']:.4f}"):
                    r_cols = st.columns(5)
                    for i, (key, label) in enumerate(LABEL_NUTRISI.items()):
                        with r_cols[i % 5]:
                            st.metric(label, f"{row.get(key, 0):.1f}")

        except FileNotFoundError:
            st.error("❌ Database makanan tidak ditemukan. Pastikan folder `data/` sudah ada.")

        # ── Nutrisi yang perlu diperhatikan ──────────────────────
        st.divider()
        st.subheader("⚠️ Nutrisi yang Masih Kurang")
        kurang = {k: v for k, v in gap.items() if v > 0}
        if not kurang:
            st.success("🎉 Semua kebutuhan nutrisi sudah terpenuhi hari ini!")
        else:
            for key, val in kurang.items():
                label = LABEL_NUTRISI.get(key, key)
                st.warning(f"**{label}** masih kurang **{val:.1f}** dari target harian.")