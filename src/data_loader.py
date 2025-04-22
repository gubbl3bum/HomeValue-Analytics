import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelBinarizer

def load_csv_file(uploaded_file):
    """
    Wczytuje dane z pliku CSV z opcjami przygotowania danych.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Sekcja przygotowania danych
        st.subheader("Przygotowanie danych")
        
        # 1. Najpierw operacje czyszczące dane
        # Usuwanie duplikatów
        remove_duplicates_opt = st.checkbox(
            "Usuń powtarzające się wiersze",
            help="Usuwa wszystkie zduplikowane wiersze z danych"
        )
        
        if remove_duplicates_opt:
            df = remove_duplicates(df)
        
        # Obsługa brakujących wartości
        handle_missing = st.checkbox(
            "Usuń wiersze z brakującymi wartościami",
            help="Usuwa wszystkie wiersze, które zawierają brakujące wartości"
        )
        
        if handle_missing:
            df = handle_missing_values(df)
            
        # 2. Opcje transformacji danych
        st.write("Wybierz opcje transformacji danych:")
        
        standardize_numeric = st.checkbox(
            "Standaryzacja danych numerycznych",
            help="Przekształca dane numeryczne (średnia=0, odchylenie standardowe=1)"
        )
        
        encode_categorical = st.checkbox(
            "Kodowanie kolumn kategorycznych",
            help="Zamienia kolumny kategoryczne na binarne (0/1), one-hot lub ordinalne"
        )
        
        # Wykonanie transformacji w odpowiedniej kolejności
        if standardize_numeric:
            df = scale_numeric_data(df)
            st.success("Dane numeryczne zostały standaryzowane")
            
        if encode_categorical:
            df = encode_categorical_columns(df)
            st.success("Kolumny kategoryczne zostały zakodowane")
            
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
    """
    try:
        # Kopiowanie DataFrame
        scaled_df = df.copy()
        
        # Identyfikacja kolumn numerycznych (z wyjątkiem lat i kolumn binarnych)
        numeric_cols = df.select_dtypes(include=['number']).columns
        # Wykluczamy kolumny z latami i kolumny binarne (0/1)
        cols_to_exclude = [col for col in numeric_cols if 'year' in col.lower()]
        cols_to_exclude.extend([col for col in numeric_cols if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1})])
        cols_to_scale = [col for col in numeric_cols if col not in cols_to_exclude]
        
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

def remove_duplicates(df):
    """
    Usuwa powtarzające się wiersze z DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame bez powtarzających się wierszy
    """
    try:
        # Kopiowanie DataFrame
        cleaned_df = df.copy()
        
        # Liczba wierszy przed usunięciem duplikatów
        original_rows = len(cleaned_df)
        
        # Usuwanie duplikatów
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Liczba usuniętych duplikatów
        removed_rows = original_rows - len(cleaned_df)
        
        if removed_rows > 0:
            st.success(f"Usunięto {removed_rows} powtarzających się wierszy. Pozostało {len(cleaned_df)} wierszy.")
        else:
            st.info("Brak powtarzających się wierszy w danych.")
            
        return cleaned_df
        
    except Exception as e:
        st.error(f"Błąd podczas usuwania duplikatów: {e}")
        return df

def encode_categorical_columns(df):
    """
    Koduje kolumny kategoryczne na wartości binarne tam, gdzie to możliwe.
    Dla kolumn z więcej niż dwiema wartościami używa prostego kodowania ordynalnego.
    """
    try:
        # Kopiowanie DataFrame
        encoded_df = df.copy()
        
        # Identyfikacja wszystkich kolumn kategorycznych
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filtrujemy kolumny, które już zostały zakodowane lub są ID
        categorical_cols = [col for col in categorical_cols 
                           if not any(c.startswith(f"{col}_") for c in df.columns) 
                           and 'id' not in col.lower()]
        
        if categorical_cols:
            from sklearn.preprocessing import LabelEncoder
            import pandas as pd
            import numpy as np
            
            # Wyświetlamy informację o znalezionych kolumnach kategorycznych
            st.info(f"Znaleziono {len(categorical_cols)} kolumn kategorycznych do zakodowania: {', '.join(categorical_cols)}")
            
            for col in categorical_cols:
                try:
                    # Konwertujemy wartości na stringi przed kodowaniem (pomijając NaN)
                    values_to_encode = df[col].astype(str)
                    values_to_encode = values_to_encode.replace('nan', np.nan)
                    
                    unique_values = values_to_encode.dropna().nunique()
                    
                    # Sprawdzamy czy kolumna ma tylko 2 unikalne wartości (binarna)
                    if unique_values == 2:
                        # Pobieramy dwie unikalne wartości
                        values_list = values_to_encode.dropna().unique().tolist()                        
                        # Kodujemy pierwszą wartość jako 0, drugą jako 1
                        first_value, second_value = values_list
                        encoded_df[col] = values_to_encode.map(
                            lambda x: 0 if x == first_value else (1 if x == second_value else np.nan)
                        ).astype('Int64')  # Używamy Int64 aby obsłużyć NaN

                    else:
                        # Dla kolumn z więcej niż 2 wartościami stosujemy proste kodowanie numeryczne
                        # Używamy LabelEncoder dla prostego kodowania numerycznego
                        encoder = LabelEncoder()
                        
                        # Przygotowujemy dane do kodowania (tylko niepuste wartości)
                        temp_values = values_to_encode.copy()
                        non_null_mask = temp_values.notna()
                        
                        if non_null_mask.any():
                            # Kodujemy tylko niepuste wartości
                            encoded_values = pd.Series(index=temp_values.index, data=np.nan)
                            encoded_values.loc[non_null_mask] = encoder.fit_transform(temp_values[non_null_mask])
                            
                            # Zapisujemy zakodowane wartości
                            encoded_df[col] = encoded_values.astype('Int64')
                            
                            # Wyświetlamy mapowanie wartości
                            mapping = {i: label for i, label in enumerate(encoder.classes_)}                                                
                except Exception as e:
                    st.warning(f"Nie udało się zakodować kolumny {col}: {str(e)}")
                    continue
            
        else:
            st.info("Nie znaleziono kolumn kategorycznych do zakodowania")
            
        return encoded_df
        
    except Exception as e:
        st.error(f"Błąd podczas kodowania kolumn kategorycznych: {e}")
        return df
