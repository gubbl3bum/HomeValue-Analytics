import streamlit as st
from data_stats import compute_basic_statistics
from data_loader import load_csv_file, preview_data, get_numeric_columns
from data_visuals import display_chart_ui
from data_ml import display_ml_ui
from data_filter import display_filter_ui

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
        df = preview_data(df)  # Przypisujemy zaktualizowany DataFrame

        # Sekcja filtrowania
        st.header("Filtrowanie danych")
        filtered_df, filters_applied = display_filter_ui(df)

        # Informacja o przefiltrowanych danych
        if filters_applied:
            dataset_info = "Używane dane: PRZEFILTROWANE"
            working_df = filtered_df  # Używamy przefiltrowanych danych
        else:
            dataset_info = "Używane dane: ORYGINALNE (bez filtrów)"
            working_df = df  # Używamy oryginalnych danych

        st.info(dataset_info)

        # Tworzymy zakładki dla różnych funkcjonalności
        tab1, tab2, tab3 = st.tabs(["Statystyki", "Wizualizacje", "Machine Learning"])

        with tab1:
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

        with tab2:
            # Sekcja wizualizacji danych
            display_chart_ui(df)

        with tab3:
            # Sekcja analizy machine learning
            display_ml_ui(df)
