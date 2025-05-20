import streamlit as st
from data_stats import display_statistics_ui, compute_correlation_matrix
from data_loader import (
    load_csv_file, preview_data, scale_numeric_data, encode_categorical_columns
)
from data_visuals import display_chart_ui, create_violin_plot, create_pair_plot
from data_ml import display_ml_ui
from data_filter import display_subtable_ui
from data_processing import extract_subtable, replace_values, fill_missing_values
from data_encoding import one_hot_encode, target_encode

# Konfiguracja strony
st.set_page_config(page_title="HomeValue-Analytics", page_icon="🏡", layout="wide")

st.title("HomeValue-Analytics 🏡📊")

# Przesyłanie pliku CSV
uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")

if uploaded_file:
    # Ładowanie danych przy użyciu funkcji z modułu data_loader
    df, row_count = load_csv_file(uploaded_file)

    if df is not None:
        st.success(f"Plik został wczytany pomyślnie! Liczba wierszy: {row_count}")

        # Wyświetlanie i edycja podglądu danych
        st.header("Podgląd danych")
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
            # Tworzymy kolumny dla statystyk opisowych i kategorycznych
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Statystyki opisowe")
                display_statistics_ui(working_df, analysis_type="descriptive")

            with col2:
                st.subheader("Analiza kategoryczna")
                display_statistics_ui(working_df, analysis_type="categorical")

            # Analiza korelacji w oddzielnej sekcji
            st.subheader("Analiza korelacji")
            numeric_columns = working_df.select_dtypes(include=['number']).columns
            selected_columns = st.multiselect("Wybierz kolumny do analizy korelacji", numeric_columns)
            method = st.selectbox("Metoda korelacji", ["pearson", "kendall", "spearman"])
            if st.button("Oblicz korelację"):
                corr_matrix = compute_correlation_matrix(working_df, selected_columns, method)
                st.write(corr_matrix)

        with tab2:
            # Sekcja wizualizacji danych
            display_chart_ui(working_df)

            # Wizualizacje
            st.subheader("Dodatkowe wykresy")
            chart_type = st.selectbox("Typ wykresu", ["Violin Plot", "Pair Plot"])
            if chart_type == "Violin Plot":
                x_column = st.selectbox("Kolumna X", working_df.columns)
                y_column = st.selectbox("Kolumna Y", working_df.columns)
                if st.button("Generuj Violin Plot"):
                    fig = create_violin_plot(working_df, x_column, y_column)
                    st.plotly_chart(fig)
            elif chart_type == "Pair Plot":
                selected_columns = st.multiselect("Wybierz kolumny do Pair Plot", working_df.columns)
                if st.button("Generuj Pair Plot"):
                    fig = create_pair_plot(working_df, selected_columns)
                    st.pyplot(fig)

        with tab3:
            # Sekcja analizy machine learning
            display_ml_ui(working_df)

            # Kodowanie
            st.subheader("Kodowanie kolumn symbolicznych")
            column = st.selectbox("Kolumna do zakodowania", working_df.columns)
            encoding_method = st.selectbox("Metoda kodowania", ["One-Hot", "Target"])
            if encoding_method == "One-Hot":
                if st.button("Zakoduj One-Hot"):
                    working_df = one_hot_encode(working_df, column)
                    st.write(working_df)
            elif encoding_method == "Target":
                target = st.selectbox("Kolumna docelowa", working_df.columns)
                if st.button("Zakoduj Target"):
                    working_df = target_encode(working_df, column, target)
                    st.write(working_df)

            # Zamiana wartości
            st.subheader("Zamiana wartości")
            column = st.selectbox("Kolumna do zamiany", working_df.columns, key="replace_column")
            to_replace = st.text_input("Wartość do zamiany")
            value = st.text_input("Nowa wartość")
            if st.button("Zamień wartości"):
                working_df = replace_values(working_df, column, to_replace, value)
                st.write(working_df)

            # Ekstrakcja podtablic
            st.subheader("Ekstrakcja podtablic")
            mode = st.selectbox("Tryb ekstrakcji", ["remove", "keep"])
            columns = st.multiselect("Kolumny do ekstrakcji", working_df.columns, key="extract_columns")
            if st.button("Ekstraktuj podtablicę"):
                working_df = extract_subtable(working_df, columns, mode)
                st.write(working_df)
