import streamlit as st
from data_stats import compute_basic_statistics
from data_loader import load_csv_file, preview_data, get_numeric_columns
from data_visuals import display_chart_ui
from data_ml import display_ml_ui
from data_filter import display_filter_ui

# Page configuration
st.set_page_config(page_title="HomeValue-Analytics", page_icon="🏡", layout="wide")

st.title("HomeValue-Analytics 🏡📊")

# Uploading a CSV file
uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")

if uploaded_file:
    # Loading data using the data_loader module function
    df = load_csv_file(uploaded_file)

    if df is not None:
        # Viewing and editing data preview
        df = preview_data(df)  # Assigning updated DataFrame

        # Filtering section
        st.header("Filtrowanie danych")
        filtered_df, filters_applied = display_filter_ui(df)

        # Information about filtered data
        if filters_applied:
            dataset_info = "Używane dane: PRZEFILTROWANE"
            working_df = filtered_df  # Using filtered data
        else:
            dataset_info = "Używane dane: ORYGINALNE (bez filtrów)"
            working_df = df  # Using original data

        st.info(dataset_info)

        # Creating a tab for different functionalitiesi
        tab1, tab2, tab3 = st.tabs(["Statystyki", "Wizualizacje", "Machine Learning"])

        with tab1:
            # Selecting columns for analysis
            numeric_cols = get_numeric_columns(df)
            selected_numeric_cols = st.multiselect("Wybierz kolumny numeryczne do analizy", numeric_cols, default=numeric_cols)

            # **Basic Statistics**
            if st.button("Oblicz statystyki"):
                stats = compute_basic_statistics(df, selected_numeric_cols)
                if stats is not None:
                    st.subheader("Podstawowe statystyki")
                    st.dataframe(stats)
                else:
                    st.warning("Nie wybrano kolumn numerycznych do analizy.")

        with tab2:
            # Data visualization section
            display_chart_ui(df)

        with tab3:
            # Machine learning analysis section
            display_ml_ui(df)
