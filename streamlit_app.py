import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Analisis Sentimen Akulaku")
st.write("Versi stabil (anti error Streamlit Cloud)")

def rule_sentiment(text):
    text = str(text).lower()

    if any(k in text for k in ["bagus", "baik", "mantap", "cepat", "puas"]):
        return "positif"
    if any(k in text for k in ["jelek", "error", "lambat", "kecewa"]):
        return "negatif"
    return "netral"

uploaded_file = st.file_uploader("📂 Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview data:")
    st.dataframe(df.head())

    col = st.selectbox("Pilih kolom teks", df.columns)

    df["sentimen"] = df[col].apply(rule_sentiment)

    st.write("Hasil analisis:")
    st.dataframe(df)

    st.write("Jumlah sentimen:")
    st.write(df["sentimen"].value_counts())
