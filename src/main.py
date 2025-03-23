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

        # Podgląd pierwszych wierszy
        st.write("Podgląd danych:")
        st.dataframe(df.head())

        # Rozpoznawanie typów kolumn
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Wybór kolumn do analizy
        st.subheader("Wybierz atrybuty do analizy")
        
        selected_numeric_cols = st.multiselect("Kolumny numeryczne", numeric_cols, default=numeric_cols)
        selected_categorical_cols = st.multiselect("Kolumny kategoryczne", categorical_cols, default=categorical_cols)

        # Walidacja wyboru
        if not selected_numeric_cols and not selected_categorical_cols:
            st.warning("Wybierz przynajmniej jedną kolumnę do analizy.")

        else:
            st.success("Wybrano kolumny do analizy!")
            st.write(f"**Numeryczne:** {selected_numeric_cols}")
            st.write(f"**Kategoryczne:** {selected_categorical_cols}")

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
