import streamlit as st
import pandas as pd
import sklearn as sk






st.title("Kalkulator Gizi Makanan")
jenis_kelamin = st.selectbox("Pilih Jenis Kelamin", ["---", "Pria", "Wanita"])

if jenis_kelamin == "---":
    st.write("Pilih Jenis Kelamin")
else:
    st.subheader("Masukkan Informasi Singkat Mengenai Anda")
    umur = st.number_input("Umur = ", 0)
    tinggi_badan = st.number_input("Tinggi Badan(CM) = ", 0)
    berat_badan = st.number_input("Berat Badan(KG) = ", 0)

    if jenis_kelamin == "Pria":
        bmr = (10 * berat_badan) + (6.25 * tinggi_badan) - (5 * umur) + 5

    else:
        bmr = (10 * berat_badan) + (6.25 * tinggi_badan) - (5 * umur) - 161

    st.metric(label="BMR Anda", value=f"{bmr} kkal/hari")

    tingkat_aktivitas = st.selectbox("Tipe Aktivitas = ", 
                                    ["---",
                                    "Jarang gerak/duduk terus",
                                    "Aktivitas ringan (olahraga 1-3 kali/minggu)",
                                    "Aktivitas sedang (olahraga 3-5 kali/minggu)",
                                    "Aktivitas berat (olahraga 6-7 kali/minggu)",
                                    "Aktivitas sangat berat (olahraga intens 2x sehari)"
                                    ])
    if tingkat_aktivitas == "---":
        st.write("---")

    elif tingkat_aktivitas == "Jarang gerak/duduk terus":
        tdee = bmr * 1.2
        st.metric(label="TDEE Anda", value=f"{tdee} Kalori per Hari")

    elif tingkat_aktivitas == "Aktivitas ringan (olahraga 1-3 kali/minggu)":
        tdee = bmr * 1.375
        st.metric(label="TDEE Anda", value=f"{tdee} Kalori per Hari")

    elif tingkat_aktivitas == "Aktivitas sedang (olahraga 3-5 kali/minggu)":
        tdee = bmr * 1.55
        st.metric(label="TDEE Anda", value=f"{tdee} Kalori per Hari")

    elif tingkat_aktivitas == "Aktivitas berat (olahraga 6-7 kali/minggu)":
        tdee = bmr * 1.725
        st.metric(label="TDEE Anda", value=f"{tdee} Kalori per Hari")

    elif tingkat_aktivitas == "Aktivitas sangat berat (olahraga intens 2x sehari)":
        tdee = bmr * 1.9
        st.metric(label="TDEE Anda", value=f"{tdee} Kalori per Hari")
