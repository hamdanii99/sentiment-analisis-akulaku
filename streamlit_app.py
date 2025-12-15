import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analisis Sentimen Ulasan Akulaku")
st.caption("Versi sederhana & stabil (tanpa error)")

st.write("Upload file CSV, lalu sistem akan memberi label sentimen sederhana.")

# =========================
# RULE-BASED SENTIMENT
# =========================
def rule_sentiment(text):
    text = str(text).lower()

    positif = ["bagus", "mantap", "baik", "membantu", "cepat", "mudah"]
    negatif = ["jelek", "error", "lambat", "buruk", "kecewa", "ribet"]

    if any(k in text for k in positif):
        return "positif"
    elif any(k in text for k in negatif):
        return "negatif"
    else:
        return "netral"

# =========================
# UPLOAD CSV
# =========================
uploaded_file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(f"Berhasil upload {len(df)} data")
    st.dataframe(df.head())

    text_col = st.selectbox(
        "Pilih kolom teks ulasan",
        df.columns
    )

    if st.button("🔍 Analisis Sentimen"):
        df["sentimen"] = df[text_col].apply(rule_sentiment)

        st.subheader("📊 Distribusi Sentimen")
        st.bar_chart(df["sentimen"].value_counts())

        st.subheader("📄 Hasil Analisis")
        st.dataframe(df.head(100))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Hasil CSV",
            csv,
            "hasil_sentimen.csv",
            "text/csv"
        )

st.caption("© Analisis Sentimen Akulaku – Streamlit")
