import streamlit as st
import pandas as pd

st.title("HomeValue-Analytics 🏡📊")
st.subheader("Import własnego zestawu danych")

# Przesyłanie pliku
uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")

if uploaded_file is not None:
    try:
        # Wczytanie danych
        df = pd.read_csv(uploaded_file)

        # Podgląd pierwszych kilku wierszy
        st.write("Podgląd danych:")
        st.dataframe(df.head())

        # Informacje o datasetcie
        st.write("Podstawowe informacje:")
        st.write(df.info())

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
