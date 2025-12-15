import streamlit as st
import joblib
import pandas as pd
import streamlit as st
import pandas as pd
import joblib


tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model_nb.pkl")

st.set_page_config(
    page_title="Analisis Sentimen Akulaku",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analisis Sentimen Ulasan Akulaku")
st.write("Klasifikasi sentimen: **Positif – Netral – Negatif**")
st.header("📂 Upload Data CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV ulasan Akulaku",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Berhasil upload {len(df)} data")

    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    text_col = st.selectbox(
        "Pilih kolom teks ulasan",
        df.columns
    )

text = st.text_area(
    "Masukkan ulasan pengguna:",
    height=150,
    placeholder="Contoh: aplikasinya cukup membantu tapi kadang error"
)

    if st.button("🔍 Analisis Sentimen"):
        texts = df[text_col].astype(str)

        X = tfidf.transform(texts)
        preds = model.predict(X)

        df["sentimen"] = preds

if st.button("Analisis Sentimen"):
    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if pred == "positif":
            st.success("😊 Sentimen: POSITIF")
        elif pred == "netral":
            st.info("😐 Sentimen: NETRAL")
        else:
            st.error("😡 Sentimen: NEGATIF")

        df_prob = pd.DataFrame({
            "Sentimen": model.classes_,
            "Probabilitas": prob
        })

        st.bar_chart(df_prob.set_index("Sentimen"))
        st.subheader("📌 Hasil Analisis")
        st.dataframe(df[[text_col, "sentimen"]].head(50))
        st.subheader("📈 Distribusi Sentimen")

        sentiment_count = df["sentimen"].value_counts()
        st.bar_chart(sentiment_count)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Hasil CSV",
            csv,
            "hasil_sentimen_akulaku.csv",
            "text/csv"
        )
