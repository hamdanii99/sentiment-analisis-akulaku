import streamlit as st
import os

st.set_page_config(page_title="DEBUG", layout="wide")
st.title("🔍 DEBUG STREAMLIT")

st.write("Python berjalan normal ✅")

st.subheader("📁 Isi Direktori:")
files = os.listdir(".")
st.code(files)

if "tfidf.pkl" not in files:
    st.error("❌ tfidf.pkl TIDAK ditemukan")
else:
    st.success("✅ tfidf.pkl ditemukan")

if "model_nb.pkl" not in files:
    st.error("❌ model_nb.pkl TIDAK ditemukan")
else:
    st.success("✅ model_nb.pkl ditemukan")

st.success("Jika halaman ini muncul → Streamlit AMAN")
