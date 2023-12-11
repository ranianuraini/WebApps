import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Muat data yang telah diproses
url = "https://raw.githubusercontent.com/ranianuraini/PencarianPenambanganWeb/main/DataOlah_Antara.csv"
data = pd.read_csv(url)

# Muat vektorizer TF-IDF
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Muat klasifier Naive Bayes
with open('multinomialNB_model.pkl', 'rb') as model_file:
    nb_classifier = pickle.load(model_file)

# Aplikasi Streamlit
st.title("Prediksi Kategori Berita Antara News")

# Input pengguna
user_input = st.text_area("Masukkan artikel berita:", "Artikel berita Anda di sini...")

# Lakukan prediksi
if st.button("Prediksi Kategori"):
    # Terapkan preprocessing yang sama pada input pengguna
    user_input_tokens = user_input.lower().split(' ')
    user_input_tfidf = vectorizer.transform([' '.join(user_input_tokens)])

    # Lakukan prediksi menggunakan klasifier Naive Bayes yang sudah dilatih
    prediksi = nb_classifier.predict(user_input_tfidf)

    # Tampilkan hasil prediksi
    st.success(f"Kategori yang Diprediksi: {prediksi[0]}")

# Tampilkan dataset
if st.checkbox("Tampilkan Dataset"):
    st.dataframe(data)

# Tampilkan beberapa statistik
st.subheader("Statistik Dataset:")
st.write(f"Jumlah Sampel: {len(data)}")
st.write(f"Jumlah Kategori: {len(data['Label'].unique())}")
st.write(f"Kategori: {', '.join(data['Label'].unique())}")
