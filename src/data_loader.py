import pandas as pd
import streamlit as st

def load_csv_file(uploaded_file):
    """
    Wczytuje dane z pliku CSV.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Plik CSV przesłany przez użytkownika
    
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame z załadowanymi danymi lub None w przypadku błędu
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku: {e}")
        return None

def preview_data(df):
    """
    Wyświetla edytowalny podgląd danych z odpowiednim formatowaniem liczb.
    """
    if df is not None:
        st.write("Podgląd danych (kliknij w komórkę aby edytować):")

        # Przygotowanie konfiguracji kolumn
        column_config = {}
        for col in df.columns:
            # Sprawdzenie czy nazwa kolumny zawiera "year" (bez względu na wielkość liter)
            is_year_column = "year" in col.lower()

            if pd.api.types.is_integer_dtype(df[col]) or is_year_column:
                # Dla liczb całkowitych i kolumn z rokiem - bez separatorów tysięcy
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%d",
                    step=1
                )
                # Jeśli to kolumna z rokiem, konwertujemy wartości na int
                if is_year_column and not pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif pd.api.types.is_float_dtype(df[col]):
                # Dla liczb zmiennoprzecinkowych - 2 miejsca po przecinku
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%.2f",
                    step=0.01
                )
        # Używamy data_editor z konfiguracją kolumn
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            column_config=column_config
        )

        # Jeśli dane zostały zmienione, aktualizujemy DataFrame
        if not edited_df.equals(df):
            st.success("Dane zostały zaktualizowane!")
            return edited_df

        return df
    return None

def get_numeric_columns(df):
    """
    Zwraca listę kolumn numerycznych z DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    
    Returns:
    --------
    list
        Lista nazw kolumn numerycznych
    """
    if df is not None:
        return df.select_dtypes(include=["number"]).columns.tolist()
    return []

def get_categorical_columns(df):
    """
    Zwraca listę kolumn kategorycznych z DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    
    Returns:
    --------
    list
        Lista nazw kolumn kategorycznych
    """
    if df is not None:
        return df.select_dtypes(include=["object", "category"]).columns.tolist()
    return []

def get_datetime_columns(df):
    """
    Zwraca listę kolumn z datami z DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    
    Returns:
    --------
    list
        Lista nazw kolumn z datami
    """
    if df is not None:
        return df.select_dtypes(include=["datetime"]).columns.tolist()
    return []

def get_column_types(df):
    """
    Zwraca słownik z typami kolumn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    
    Returns:
    --------
    dict
        Słownik z typami kolumn
    """
    if df is not None:
        return {
            'numeric': get_numeric_columns(df),
            'categorical': get_categorical_columns(df),
            'datetime': get_datetime_columns(df)
        }
    return {'numeric': [], 'categorical': [], 'datetime': []}
