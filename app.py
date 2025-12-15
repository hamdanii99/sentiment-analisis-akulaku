import streamlit as st
import pandas as pd
import joblib
import re

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analisis Sentimen Ulasan Akulaku")
st.caption("Naïve Bayes | Positif • Netral • Negatif")

# =========================
# LOAD MODEL (ANTI CRASH)
# =========================
try:
    tfidf = joblib.load("tfidf.pkl")
    model = joblib.load("model_nb.pkl")
except Exception as e:
    st.error("❌ Model gagal dimuat")
    st.code(str(e))
    st.stop()

# =========================
# PREPROCESSING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# SESSION STATE
# =========================
if "result_df" not in st.session_state:
    st.session_state.result_df = None

# =========================
# INPUT MANUAL
# =========================
st.subheader("✍️ Analisis 1 Kalimat")
input_text = st.text_area("Masukkan ulasan")

if st.button("Prediksi Kalimat"):
    if input_text.strip():
        clean = clean_text(input_text)
        X = tfidf.transform([clean])
        proba = model.predict_proba(X)[0]
        labels = model.classes_

        sentimen = labels[proba.argmax()]

        st.success(f"Sentimen: **{sentimen.upper()}**")
        st.write("Probabilitas:", dict(zip(labels, proba)))
    else:
        st.warning("Teks tidak boleh kosong")

st.divider()

# =========================
# UPLOAD CSV
# =========================
st.subheader("📂 Analisis File CSV")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Berhasil upload {len(df)} data")
    st.dataframe(df.head())

    text_col = st.selectbox("Pilih kolom teks ulasan", df.columns)

    if st.button("🔍 Analisis CSV"):
        df["clean_text"] = df[text_col].astype(str).apply(clean_text)
        X = tfidf.transform(df["clean_text"])
        proba = model.predict_proba(X)
        labels = model.classes_

        df["sentimen"] = [labels[p.argmax()] for p in proba]
        st.session_state.result_df = df

# =========================
# HASIL ANALISIS
# =========================
if st.session_state.result_df is not None:
    result = st.session_state.result_df

    st.subheader("📊 Distribusi Sentimen")
    st.bar_chart(result["sentimen"].value_counts())

    st.subheader("🔎 Filter Sentimen")
    filter_sentimen = st.multiselect(
        "Pilih sentimen",
        ["positif", "netral", "negatif"],
        default=["positif", "netral", "negatif"]
    )

    filtered = result[result["sentimen"].isin(filter_sentimen)]
    st.dataframe(filtered.head(100))

    csv = filtered.to_csv(index=False).encode("utf-8")
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
