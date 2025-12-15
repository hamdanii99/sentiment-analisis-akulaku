import streamlit as st
import os
import sys
import traceback

st.set_page_config(page_title="DEBUG MODE", layout="wide")
st.title("🛠️ DEBUG STREAMLIT STARTUP")

st.write("Jika halaman ini muncul, Streamlit BERJALAN.")
st.write("Sekarang cek file & model satu per satu.")

st.divider()

# =========================
# CEK FILE DI DIREKTORI
# =========================
st.subheader("📁 File di root repo:")
try:
    files = os.listdir(".")
    st.write(files)
except Exception as e:
    st.error("Gagal membaca direktori")
    st.code(str(e))
    st.stop()

st.divider()

# =========================
# TES IMPORT LIBRARY
# =========================
st.subheader("📦 Tes import library")

try:
    import joblib
    import pandas
    import sklearn
    st.success("Import library BERHASIL")
    st.write("Versi sklearn:", sklearn.__version__)
except Exception as e:
    st.error("Import library GAGAL")
    st.code(traceback.format_exc())
    st.stop()

st.divider()

# =========================
# TES LOAD TFIDF
# =========================
st.subheader("🧪 Tes load tfidf.pkl")

try:
    tfidf = joblib.load("tfidf.pkl")
    st.success("tfidf.pkl BERHASIL dimuat")
except Exception as e:
    st.error("GAGAL load tfidf.pkl")
    st.code(traceback.format_exc())
    st.stop()

st.divider()

# =========================
# TES LOAD MODEL
# =========================
st.subheader("🧪 Tes load model_nb.pkl")

try:
    model = joblib.load("model_nb.pkl")
    st.success("model_nb.pkl BERHASIL dimuat")
    st.write("Model type:", type(model))
except Exception as e:
    st.error("GAGAL load model_nb.pkl")
    st.code(traceback.format_exc())
    st.stop()

st.success("🎉 SEMUA TES LULUS — MODEL AMAN")
