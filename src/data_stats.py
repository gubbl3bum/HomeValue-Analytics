import streamlit as st

def compute_basic_statistics(df, columns):
    """
    Oblicza podstawowe statystyki dla wybranych kolumn numerycznych.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    columns : list
        Lista kolumn do analizy
    
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame ze statystykami lub None jeśli brak kolumn do analizy
    """
    if not columns:
        return None

    stats = df[columns].agg([
        'count',
        'mean',
        'std',
        'min',
        'max',
        'median',
        'skew',  # używamy nazw funkcji zamiast lambda
        'kurt'   # używamy nazw funkcji zamiast lambda
    ]).round(2)

    # Tłumaczenie nazw statystyk na polski
    stats = stats.rename({
        'count': 'Liczba obserwacji',
        'mean': 'Średnia',
        'std': 'Odchylenie standardowe',
        'min': 'Minimum',
        'max': 'Maksimum',
        'median': 'Mediana',
        'skew': 'Skośność',
        'kurt': 'Kurtoza'
    })

    return stats

def compute_correlation_matrix(df, selected_columns, method='pearson'):
    """
    Oblicza macierz korelacji między wybranymi kolumnami.
    :param df: DataFrame z danymi
    :param selected_columns: Lista kolumn do analizy korelacji
    :param method: Metoda korelacji ('pearson', 'kendall', 'spearman')
    :return: DataFrame z macierzą korelacji
    """
    if not selected_columns or len(selected_columns) < 2:
        return None

    # Filtruj tylko kolumny numeryczne
    numeric_columns = df[selected_columns].select_dtypes(include=['number']).columns
    if len(numeric_columns) < 2:
        raise ValueError("Do obliczenia korelacji wymagane są co najmniej dwie kolumny numeryczne.")

    return df[numeric_columns].corr(method=method)

def analyze_categorical_columns(df, categorical_columns):
    """
    Analizuje kolumny kategoryczne - oblicza statystyki opisowe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    categorical_columns : list
        Lista kolumn kategorycznych do analizy
    
    Returns:
    --------
    dict
        Słownik zawierający dla każdej kolumny:
        - value_counts: częstotliwość występowania wartości
        - unique_count: liczba unikalnych wartości
        - mode: dominanta
        - null_count: liczba wartości null
        - null_percentage: procent wartości null
    """
    if not categorical_columns:
        return None

    results = {}

    for col in categorical_columns:
        # Podstawowe statystyki
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Liczba']
        value_counts['Procent'] = (value_counts['Liczba'] / len(df)) * 100

        # Dodatkowe statystyki
        summary = {
            'value_counts': value_counts,
            'unique_count': df[col].nunique(),
            'mode': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100
        }

        results[col] = summary

    return results

def analyze_price_per_sqm(df, price_column, area_column):
    """
    Oblicza cenę za metr kwadratowy i podstawowe statystyki.
    
    :param df: DataFrame z danymi
    :param price_column: Nazwa kolumny z ceną
    :param area_column: Nazwa kolumny z powierzchnią
    :return: DataFrame ze statystykami ceny za metr kwadratowy
    """
    if price_column not in df.columns or area_column not in df.columns:
        return None

    # Unikamy dzielenia przez zero
    valid_data = df[(df[area_column] > 0) & df[price_column].notna()]

    if len(valid_data) == 0:
        return None

    valid_data['price_per_sqm'] = valid_data[price_column] / valid_data[area_column]

    stats = valid_data['price_per_sqm'].describe().to_frame().T
    stats['missing_values'] = len(df) - len(valid_data)
    stats['missing_percentage'] = (stats['missing_values'] / len(df)) * 100

    return stats


def display_statistics_ui(df, analysis_type="descriptive"):
    """
    Wyświetla interfejs użytkownika do analizy statystycznej w zależności od typu analizy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    analysis_type : str
        Typ analizy: "descriptive", "correlation", "categorical"
    """
    if df is None or df.empty:
        st.warning("Brak danych do analizy.")
        return

    if analysis_type == "descriptive":
        # Analiza statystyk opisowych
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        st.subheader("Analiza kolumn numerycznych 📊")
        if numeric_cols:
            selected_numeric_cols = st.multiselect(
                "Wybierz kolumny numeryczne do analizy",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            if st.button("Oblicz statystyki numeryczne", key="descriptive_stats"):
                stats = compute_basic_statistics(df, selected_numeric_cols)
                if stats is not None:
                    st.dataframe(stats)
        else:
            st.info("Brak kolumn numerycznych w zbiorze danych.")

    elif analysis_type == "categorical":
        # Analiza kolumn kategorycznych
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.subheader("Analiza kolumn kategorycznych 📑")
        if categorical_cols:
            selected_cat_cols = st.multiselect(
                "Wybierz kolumny kategoryczne do analizy",
                categorical_cols,
                default=categorical_cols[:min(3, len(categorical_cols))]
            )
            if st.button("Oblicz statystyki kategoryczne", key="categorical_stats"):
                cat_stats = analyze_categorical_columns(df, selected_cat_cols)
                if cat_stats:
                    for col, stats in cat_stats.items():
                        st.write(f"\n### {col}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Liczba unikalnych wartości", stats['unique_count'])
                        with col2:
                            st.metric("Liczba braków danych", stats['null_count'])
                        with col3:
                            st.metric("Najczęstsza wartość", stats['mode'])
                        st.write("Rozkład wartości:")
                        st.dataframe(stats['value_counts'])
        else:
            st.info("Brak kolumn kategorycznych w zbiorze danych.")
