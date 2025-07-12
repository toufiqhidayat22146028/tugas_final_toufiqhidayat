import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Data", layout="wide", page_icon="📊")
st.markdown("<h2 style='text-align:center;'>📊 Aplikasi Analisis & Klasterisasi Data</h2>", unsafe_allow_html=True)

# Navigasi sidebar
menu = st.sidebar.radio("📌 Pilih Menu", [
    "Klasifikasi Diabetes",
    "Pengelompokan Data",
    "Clustering Lokasi Gerai Kopi",
    "Visualisasi DBSCAN",
    "Input Lokasi Baru"
])

# Load model klasifikasi
try:
    with open("model_knn.pkl", "rb") as f:
        model_knn = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        diabetes_scaler = pickle.load(f)
except FileNotFoundError:
    model_knn = None
    diabetes_scaler = None

# Load dataset gerai kopi
df_kopi = pd.read_csv("lokasi_gerai_kopi_clean.csv")
X_kopi = df_kopi[["x", "y"]].values
kopi_scaler = StandardScaler().fit(X_kopi)
X_scaled_kopi = kopi_scaler.transform(X_kopi)

kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
agglo = AgglomerativeClustering(n_clusters=5)
dbscan = DBSCAN(eps=0.5, min_samples=5)

df_kopi["kmeans_label"] = kmeans.fit_predict(X_scaled_kopi)
df_kopi["agglo_label"] = agglo.fit_predict(X_scaled_kopi)
df_kopi["dbscan_label"] = dbscan.fit_predict(X_scaled_kopi)

# 1. Klasifikasi Diabetes
if menu == "Klasifikasi Diabetes":
    st.header("🔬 Klasifikasi Diabetes (KNN)")
    if model_knn is None or diabetes_scaler is None:
        st.warning("❌ Model atau scaler belum tersedia.")
    else:
        st.subheader("📥 Input Data Pasien")
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Kehamilan", 0, 20)
            glucose = st.number_input("Glukosa", 0, 200)
            blood_pressure = st.number_input("Tekanan Darah", 0, 150)
            skin_thickness = st.number_input("Ketebalan Kulit", 0, 100)
        with col2:
            insulin = st.number_input("Insulin", 0, 900)
            bmi = st.number_input("BMI", 0.0, 70.0)
            dpf = st.number_input("DPF", 0.0, 3.0)
            age = st.number_input("Usia", 1, 120)

        if st.button("🔎 Prediksi"):
            data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]])
            hasil = model_knn.predict(diabetes_scaler.transform(data))[0]
            st.success(f"Hasil: {'🟢 Tidak Diabetes' if hasil==0 else '🔴 Diabetes'}")

# 2. Pengelompokan Data (Clustering)
elif menu == "Pengelompokan Data":
    st.header("📊 Clustering Data Diabetes")
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
    df.columns = ['Preg', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']
    scaled = StandardScaler().fit_transform(df.drop("Outcome", axis=1))
    df["Cluster"] = KMeans(n_clusters=2, random_state=42, n_init='auto').fit_predict(scaled)

    st.subheader("📈 Visualisasi Glukosa vs BMI")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Glucose", y="BMI", hue="Cluster", data=df, palette="Set2", s=100, edgecolor='black', ax=ax)
    ax.set_title("Pengelompokan Berdasarkan Glukosa dan BMI")
    st.pyplot(fig)

# 3. Clustering Lokasi Kopi
elif menu == "Clustering Lokasi Gerai Kopi":
    st.header("📍 Clustering Lokasi Gerai Kopi")

    st.subheader("🔹 KMeans")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="x", y="y", hue="kmeans_label", data=df_kopi, palette="Set1", s=100, edgecolor='black', ax=ax1)
    ax1.set_title("KMeans Clustering")
    st.pyplot(fig1)

    st.subheader("🔸 Agglomerative")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="x", y="y", hue="agglo_label", data=df_kopi, palette="cool", s=100, edgecolor='black', ax=ax2)
    ax2.set_title("Agglomerative Clustering")
    st.pyplot(fig2)

# 4. Visualisasi DBSCAN
elif menu == "Visualisasi DBSCAN":
    st.header("🌌 DBSCAN Clustering Lokasi Gerai Kopi")
    fig3, ax3 = plt.subplots()
    for label in set(df_kopi["dbscan_label"]):
        data = df_kopi[df_kopi["dbscan_label"] == label]
        color = "gray" if label == -1 else None
        ax3.scatter(data["x"], data["y"], label=f"Klaster {label}" if label != -1 else "Noise", s=100, edgecolors="black", c=color)
    ax3.set_title("DBSCAN Clustering")
    ax3.legend()
    st.pyplot(fig3)

# 5. Input Lokasi Baru
elif menu == "Input Lokasi Baru":
    st.header("📍 Prediksi Klaster Lokasi Baru")
    x = st.number_input("Koordinat X", value=0.0)
    y = st.number_input("Koordinat Y", value=0.0)

    if st.button("🔍 Prediksi Klaster"):
        point = np.array([[x, y]])
        scaled_point = kopi_scaler.transform(point)
        km_label = kmeans.predict(scaled_point)[0]
        agglo_label = agglo.fit_predict(np.vstack([X_scaled_kopi, scaled_point]))[-1]
        dbscan_label = dbscan.fit_predict(np.vstack([X_scaled_kopi, scaled_point]))[-1]

        st.success(f"KMeans: Klaster {km_label}")
        st.info(f"Agglomerative: Klaster {agglo_label}")
        st.warning(f"DBSCAN: {'Noise' if dbscan_label == -1 else f'Klaster {dbscan_label}'}")
