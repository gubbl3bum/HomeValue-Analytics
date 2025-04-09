import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def load_csv_file(uploaded_file):
    """
    Wczytuje dane z pliku CSV z opcjami przygotowania danych.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Sekcja przygotowania danych
        st.subheader("Przygotowanie danych")
        
        # Obsługa brakujących wartości
        handle_missing = st.checkbox(
            "Usuń wiersze z brakującymi wartościami",
            help="Usuwa wszystkie wiersze, które zawierają brakujące wartości"
        )
        
        if handle_missing:
            df = handle_missing_values(df)
        
        # Standaryzacja danych
        scale_data = st.checkbox(
            "Standaryzuj dane numeryczne",
            help="Standaryzacja (odjęcie średniej i podzielenie przez odchylenie standardowe) pomoże w analizie ML i wizualizacjach"
        )
        
        if scale_data:
            df = scale_numeric_data(df)
            st.success("Dane zostały standaryzowane (średnia=0, odchylenie standardowe=1)")
            
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

def scale_numeric_data(df):
    """
    Standaryzuje kolumny numeryczne w DataFrame (odejmuje średnią i dzieli przez odchylenie standardowe).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame ze standaryzowanymi danymi numerycznymi
    """
    try:
        # Kopiowanie DataFrame
        scaled_df = df.copy()
        
        # Identyfikacja kolumn numerycznych (z wyjątkiem lat)
        numeric_cols = df.select_dtypes(include=['number']).columns
        year_cols = [col for col in numeric_cols if 'year' in col.lower()]
        cols_to_scale = [col for col in numeric_cols if col not in year_cols]
        
        if not cols_to_scale:
            return df
            
        # Standaryzacja danych
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[cols_to_scale])
        scaled_df[cols_to_scale] = scaled_values
        
        return scaled_df
        
    except Exception as e:
        st.error(f"Błąd podczas standaryzacji danych: {e}")
        return df

def handle_missing_values(df):
    """
    Usuwa wiersze z brakującymi wartościami z DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame bez wierszy z brakującymi wartościami
    """
    try:
        # Kopiowanie DataFrame
        cleaned_df = df.copy()
        
        # Sprawdzenie liczby wierszy przed czyszczeniem
        original_rows = len(cleaned_df)
        
        # Usuwanie wierszy z brakującymi wartościami
        cleaned_df = cleaned_df.dropna()
        
        # Liczba usuniętych wierszy
        removed_rows = original_rows - len(cleaned_df)
        
        if removed_rows > 0:
            st.success(f"Usunięto {removed_rows} wierszy z brakującymi wartościami. Pozostało {len(cleaned_df)} wierszy.")
        else:
            st.info("Brak brakujących wartości w danych.")
            
        return cleaned_df
        
    except Exception as e:
        st.error(f"Błąd podczas usuwania brakujących wartości: {e}")

        return df
