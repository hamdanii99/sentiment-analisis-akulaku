import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model_nb.pkl")

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analisis Sentimen Ulasan Akulaku")
st.write("Sistem analisis sentimen (positif, netral, negatif) berbasis Naïve Bayes")

# =========================
# INPUT TEKS MANUAL
# =========================
st.subheader("✍️ Analisis 1 Kalimat")
input_text = st.text_area("Masukkan ulasan")

if st.button("Prediksi Kalimat"):
    if input_text.strip() != "":
        X = tfidf.transform([input_text])
        pred = model.predict(X)[0]
        st.success(f"Sentimen: **{pred.upper()}**")
    else:
        st.warning("Teks tidak boleh kosong")

st.divider()

# =========================
# UPLOAD CSV
# =========================
st.subheader("📂 Analisis File CSV")
uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(f"Berhasil upload {len(df)} data")
    st.dataframe(df.head())

    text_col = st.selectbox(
        "Pilih kolom teks ulasan",
        df.columns
    )

    if st.button("🔍 Analisis CSV"):
        texts = df[text_col].astype(str)
        X = tfidf.transform(texts)
        df["sentimen"] = model.predict(X)

        st.subheader("📊 Distribusi Sentimen")
        st.bar_chart(df["sentimen"].value_counts())

        st.subheader("🔎 Filter Sentimen")
        filter_sentimen = st.multiselect(
            "Pilih sentimen",
            ["positif", "netral", "negatif"],
            default=["positif", "netral", "negatif"]
        )

        filtered_df = df[df["sentimen"].isin(filter_sentimen)]
        st.dataframe(filtered_df.head(100))

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Hasil CSV",
            csv,
            "hasil_sentimen_akulaku.csv",
            "text/csv"
        )

# =========================
# FOOTER
# =========================
st.caption("© Sistem Analisis Sentimen Akulaku | Naïve Bayes")
