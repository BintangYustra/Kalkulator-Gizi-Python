"""
app.py — Kalkulator Gizi Makanan
Jalankan: streamlit run app.py
Pastikan folder data/ berisi: food.csv, food_category.csv, nutrient.csv, food_nutrient.csv
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Kalkulator Gizi", page_icon="🥗", layout="wide")

# ══════════════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

FAKTOR_AKTIVITAS = {
    "Jarang gerak/duduk terus"                          : 1.2,
    "Aktivitas ringan (olahraga 1-3 kali/minggu)"       : 1.375,
    "Aktivitas sedang (olahraga 3-5 kali/minggu)"       : 1.55,
    "Aktivitas berat (olahraga 6-7 kali/minggu)"        : 1.725,
    "Aktivitas sangat berat (olahraga intens 2x sehari)": 1.9,
}

NUTRISI = {
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

# ══════════════════════════════════════════════════════════════════
# FUNGSI DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Memuat database makanan...")
def load_data():
    food     = pd.read_csv(f'{DATA_DIR}/food.csv', usecols=['fdc_id', 'description', 'food_category_id'])
    category = pd.read_csv(f'{DATA_DIR}/food_category.csv', usecols=['id', 'description'])
    nutrient = pd.read_csv(f'{DATA_DIR}/nutrient.csv', usecols=['id', 'name'])
    fn       = pd.read_csv(f'{DATA_DIR}/food_nutrient.csv', usecols=['fdc_id', 'nutrient_id', 'amount'])

    # Rename agar bisa di-merge
    category.rename(columns={'id': 'food_category_id', 'description': 'category_name'}, inplace=True)
    nutrient.rename(columns={'id': 'nutrient_id'}, inplace=True)

    # Pivot nutrisi jadi kolom
    NAMA_NUTRISI = {
        'Energy'                        : 'kalori_kkal',
        'Protein'                       : 'protein_g',
        'Total lipid (fat)'             : 'lemak_g',
        'Carbohydrate, by difference'   : 'karbo_g',
        'Fiber, total dietary'          : 'serat_g',
        'Calcium, Ca'                   : 'kalsium_mg',
        'Iron, Fe'                      : 'zat_besi_mg',
        'Sodium, Na'                    : 'sodium_mg',
        'Vitamin C, total ascorbic acid': 'vitamin_c_mg',
        'Vitamin A, RAE'                : 'vitamin_a_mcg',
    }
    fn_merged = fn.merge(nutrient, on='nutrient_id')
    fn_merged = fn_merged[fn_merged['name'].isin(NAMA_NUTRISI)]
    fn_merged['name'] = fn_merged['name'].map(NAMA_NUTRISI)

    pivot = fn_merged.pivot_table(index='fdc_id', columns='name', values='amount', aggfunc='mean').reset_index()
    pivot.columns.name = None

    # Gabung semua
    df = food.merge(category, on='food_category_id', how='left').merge(pivot, on='fdc_id', how='inner')
    df[list(NUTRISI.keys())] = df[list(NUTRISI.keys())].fillna(0)
    return df


# ══════════════════════════════════════════════════════════════════
# FUNGSI KALKULASI
# ══════════════════════════════════════════════════════════════════
def hitung_bmr(gender, berat, tinggi, umur):
    base = (10 * berat) + (6.25 * tinggi) - (5 * umur)
    return base + 5 if gender == "Pria" else base - 161

def hitung_kebutuhan(kalori, gender, umur):
    """Hitung kebutuhan makro & mikro harian."""
    kebutuhan = {
        'kalori_kkal'  : round(kalori),
        'protein_g'    : round(kalori * 0.20 / 4),
        'lemak_g'      : round(kalori * 0.30 / 9),
        'karbo_g'      : round(kalori * 0.50 / 4),
        'serat_g'      : 38 if gender == "Pria" and umur <= 50 else 25,
        'vitamin_c_mg' : 90 if gender == "Pria" else 75,
        'kalsium_mg'   : 1000,
        'zat_besi_mg'  : 9 if gender == "Pria" else 18,
        'vitamin_a_mcg': 600 if gender == "Pria" else 500,
        'sodium_mg'    : 1500,
    }
    return kebutuhan

def total_konsumsi(keranjang):
    """Jumlahkan semua nutrisi dari makanan yang dipilih."""
    total = {k: 0.0 for k in NUTRISI}
    for item in keranjang:
        scale = item.get('porsi_gram', 100) / 100
        for k in NUTRISI:
            total[k] += item.get(k, 0) * scale
    return {k: round(v, 1) for k, v in total.items()}

def rekomendasikan(df, kebutuhan, konsumsi, exclude_ids):
    """Rekomendasi makanan menggunakan Cosine Similarity."""
    df_f = df[~df['fdc_id'].isin(exclude_ids)].copy()
    cols = list(NUTRISI.keys())

    # Vektor target = kekurangan nutrisi
    target = np.array([max(kebutuhan[k] - konsumsi.get(k, 0), 0) for k in cols]).reshape(1, -1)

    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(df_f[cols].fillna(0))
    target_scaled = scaler.transform(target)

    df_f = df_f.copy()
    df_f['skor'] = cosine_similarity(target_scaled, matrix)[0].round(4)
    return df_f.sort_values('skor', ascending=False).head(10).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
st.title("🥗 Kalkulator Gizi Makanan")

if 'keranjang' not in st.session_state:
    st.session_state.keranjang = []

tab1, tab2, tab3 = st.tabs(["👤 Profil", "🍽️ Pilih Makanan", "📊 Hasil & Rekomendasi"])

# ── TAB 1: PROFIL ─────────────────────────────────────────────────
with tab1:
    gender = st.selectbox("Jenis Kelamin", ["---", "Pria", "Wanita"])

    if gender == "---":
        st.info("Pilih jenis kelamin untuk memulai.")
    else:
        col1, col2, col3 = st.columns(3)
        umur   = col1.number_input("Umur (tahun)", 1, 120, 25)
        tinggi = col2.number_input("Tinggi Badan (cm)", 50, 250, 170)
        berat  = col3.number_input("Berat Badan (kg)", 10, 300, 65)

        bmr = hitung_bmr(gender, berat, tinggi, umur)
        st.metric("BMR", f"{bmr:.1f} kkal/hari")

        aktivitas = st.selectbox("Tingkat Aktivitas", ["---"] + list(FAKTOR_AKTIVITAS))
        tujuan    = st.selectbox("Tujuan", ["Turun berat badan", "Pertahankan berat badan", "Naikkan berat badan"])

        if aktivitas != "---":
            tdee   = bmr * FAKTOR_AKTIVITAS[aktivitas]
            adj    = {"Turun berat badan": -500, "Pertahankan berat badan": 0, "Naikkan berat badan": 300}
            kalori = tdee + adj[tujuan]

            col_a, col_b = st.columns(2)
            col_a.metric("TDEE", f"{tdee:.1f} kkal/hari")
            col_b.metric("🎯 Target Kalori", f"{kalori:.1f} kkal/hari")

            kebutuhan = hitung_kebutuhan(kalori, gender, umur)
            st.subheader("Kebutuhan Nutrisi Harian")
            cols = st.columns(5)
            for i, (k, label) in enumerate(NUTRISI.items()):
                cols[i % 5].metric(label, kebutuhan[k])

            st.session_state['kebutuhan'] = kebutuhan
            st.success("✅ Profil tersimpan! Lanjut ke tab **Pilih Makanan**.")

# ── TAB 2: PILIH MAKANAN ──────────────────────────────────────────
with tab2:
    if 'kebutuhan' not in st.session_state:
        st.warning("Lengkapi profil terlebih dahulu.")
    else:
        try:
            df = load_data()
        except FileNotFoundError:
            st.error("File CSV tidak ditemukan. Pastikan folder `data/` sudah ada.")
            st.stop()

        keyword = st.text_input("Cari makanan (bahasa Inggris)", placeholder="contoh: rice, chicken, egg...")

        if keyword:
            hasil = df[df['description'].str.contains(keyword, case=False, na=False)].head(20)
            if hasil.empty:
                st.warning("Tidak ditemukan. Coba kata kunci lain.")
            else:
                pilihan_id = st.selectbox(
                    "Pilih makanan",
                    hasil['fdc_id'].tolist(),
                    format_func=lambda x: hasil[hasil['fdc_id'] == x]['description'].values[0]
                )
                makanan = hasil[hasil['fdc_id'] == pilihan_id].iloc[0]

                with st.expander("Detail Nutrisi per 100g"):
                    cols = st.columns(5)
                    for i, (k, label) in enumerate(NUTRISI.items()):
                        cols[i % 5].metric(label, f"{makanan.get(k, 0):.1f}")

                porsi = st.number_input("Porsi (gram)", 1, 2000, 100)

                if st.button("➕ Tambahkan"):
                    item = makanan.to_dict()
                    item['porsi_gram'] = porsi
                    st.session_state.keranjang.append(item)
                    st.success(f"✅ {makanan['description']} ({porsi}g) ditambahkan!")

        # Keranjang
        st.divider()
        st.subheader("🛒 Makanan Hari Ini")
        if not st.session_state.keranjang:
            st.info("Belum ada makanan yang ditambahkan.")
        else:
            for i, item in enumerate(st.session_state.keranjang):
                c1, c2, c3 = st.columns([4, 2, 1])
                c1.write(f"🍽️ {item['description']}")
                c2.write(f"{item['porsi_gram']} g")
                if c3.button("🗑️", key=f"hapus_{i}"):
                    st.session_state.keranjang.pop(i)
                    st.rerun()

            if st.button("Kosongkan Semua"):
                st.session_state.keranjang = []
                st.rerun()

# ── TAB 3: HASIL & REKOMENDASI ────────────────────────────────────
with tab3:
    if 'kebutuhan' not in st.session_state:
        st.warning("Lengkapi profil terlebih dahulu.")
    elif not st.session_state.keranjang:
        st.warning("Tambahkan makanan terlebih dahulu.")
    else:
        kebutuhan = st.session_state['kebutuhan']
        konsumsi  = total_konsumsi(st.session_state.keranjang)

        # Tabel ringkasan
        st.subheader("📊 Konsumsi vs Kebutuhan")
        rows = []
        for k, label in NUTRISI.items():
            kb, kons = kebutuhan[k], konsumsi[k]
            gap = kb - kons
            status = "Cukup ✅" if abs(gap) < kb * 0.1 else ("Kurang ⚠️" if gap > 0 else "Berlebih ❌")
            rows.append({"Nutrisi": label, "Konsumsi": kons, "Kebutuhan": kb,
                         "Sisa": round(gap, 1), "Status": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Progress bar
        st.subheader("📈 Progress")
        for k, label in NUTRISI.items():
            kb   = kebutuhan[k]
            kons = konsumsi[k]
            pct  = min(kons / kb, 1.0) if kb > 0 else 0
            c1, c2 = st.columns([2, 5])
            c1.write(label)
            c2.progress(pct, text=f"{kons} / {kb}")

        # Rekomendasi AI
        st.divider()
        st.subheader("🤖 Rekomendasi AI (Cosine Similarity)")
        try:
            df = load_data()
            exclude = [item['fdc_id'] for item in st.session_state.keranjang]
            rekomendasi = rekomendasikan(df, kebutuhan, konsumsi, exclude)

            for _, row in rekomendasi.iterrows():
                with st.expander(f"🍽️ {row['description']}  —  Skor: {row['skor']:.4f}"):
                    cols = st.columns(5)
                    for i, (k, label) in enumerate(NUTRISI.items()):
                        cols[i % 5].metric(label, f"{row.get(k, 0):.1f}")
        except FileNotFoundError:
            st.error("Database makanan tidak ditemukan.")

        # Peringatan kekurangan
        st.divider()
        kurang = {k: kebutuhan[k] - konsumsi[k] for k in NUTRISI if kebutuhan[k] - konsumsi[k] > 0}
        if not kurang:
            st.success("🎉 Semua nutrisi sudah terpenuhi!")
        else:
            st.subheader("⚠️ Nutrisi yang Masih Kurang")
            for k, val in kurang.items():
                st.warning(f"**{NUTRISI[k]}** masih kurang **{val:.1f}**")