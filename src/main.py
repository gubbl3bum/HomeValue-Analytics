import streamlit as st
import pandas as pd
from data_stats import compute_basic_statistics
from data_loader import load_csv_file, preview_data, get_numeric_columns

st.title("HomeValue-Analytics 🏡📊")

# Przesyłanie pliku CSV
uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")

if uploaded_file:
    # Ładowanie danych przy użyciu funkcji z modułu data_loader
    df = load_csv_file(uploaded_file)
    
    if df is not None:
        # Wyświetlanie podglądu danych
        preview_data(df)

        # Wybór kolumn do analizy
        numeric_cols = get_numeric_columns(df)
        selected_numeric_cols = st.multiselect("Wybierz kolumny numeryczne do analizy", numeric_cols, default=numeric_cols)

        # **Statystyki podstawowe**
        if st.button("Oblicz statystyki"):
            stats = compute_basic_statistics(df, selected_numeric_cols)
            if stats is not None:
                st.subheader("Podstawowe statystyki")
                st.dataframe(stats)
            else:
                st.warning("Nie wybrano kolumn numerycznych do analizy.")