import streamlit as st
from data_stats import display_statistics_ui
from data_loader import load_csv_file, preview_data
from data_visuals import display_chart_ui
from data_ml import display_ml_ui
from data_filter import display_subtable_ui

# Konfiguracja strony
st.set_page_config(page_title="HomeValue-Analytics", page_icon="🏡", layout="wide")

st.title("HomeValue-Analytics 🏡📊")

# Przesyłanie pliku CSV
uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")

if uploaded_file:
    # Ładowanie danych przy użyciu funkcji z modułu data_loader
    df = load_csv_file(uploaded_file)

    if df is not None:
        # Wyświetlanie i edycja podglądu danych
        df = preview_data(df)

        # Sekcja ekstrakcji podtablic
        st.header("Ekstrakcja podtablic")
        extracted_df, extraction_applied = display_subtable_ui(df)

        # Informacja o wyekstrahowanych danych
        if extraction_applied:
            dataset_info = "Używane dane: WYEKSTRAHOWANA PODTABLICA"
            working_df = extracted_df
        else:
            dataset_info = "Używane dane: ORYGINALNE"
            working_df = df

        st.info(dataset_info)

        # Tworzymy zakładki dla różnych funkcjonalności
        tab1, tab2, tab3 = st.tabs(["Statystyki", "Wizualizacje", "Machine Learning"])

        with tab1:
            # Używamy odpowiedniego zestawu danych
            display_statistics_ui(working_df)

        with tab2:
            # Sekcja wizualizacji danych
            display_chart_ui(working_df)

        with tab3:
            # Sekcja analizy machine learning
            display_ml_ui(working_df)
