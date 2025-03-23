import streamlit as st
import pandas as pd
import numpy as np

def filter_numeric_column(df, column, min_value=None, max_value=None):
    """
    Filtruje DataFrame według wartości minimalnej i maksymalnej dla kolumny numerycznej.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    column : str
        Nazwa kolumny numerycznej do filtrowania
    min_value : float, optional
        Minimalna wartość (włącznie)
    max_value : float, optional
        Maksymalna wartość (włącznie)
        
    Returns:
    --------
    pandas.DataFrame
        Przefiltrowany DataFrame
    """
    filtered_df = df.copy()
    
    if min_value is not None:
        filtered_df = filtered_df[filtered_df[column] >= min_value]
        
    if max_value is not None:
        filtered_df = filtered_df[filtered_df[column] <= max_value]
        
    return filtered_df

def filter_categorical_column(df, column, selected_values):
    """
    Filtruje DataFrame według wybranych wartości dla kolumny kategorycznej.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    column : str
        Nazwa kolumny kategorycznej do filtrowania
    selected_values : list
        Lista wybranych wartości
        
    Returns:
    --------
    pandas.DataFrame
        Przefiltrowany DataFrame
    """
    if not selected_values:
        return df
        
    return df[df[column].isin(selected_values)]

def filter_by_missing_values(df, columns, include_missing=True):
    """
    Filtruje DataFrame według brakujących wartości w wybranych kolumnach.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    columns : list
        Lista nazw kolumn do sprawdzenia
    include_missing : bool
        Jeśli True, zachowuje wiersze z brakującymi wartościami,
        jeśli False, usuwa wiersze z brakującymi wartościami
        
    Returns:
    --------
    pandas.DataFrame
        Przefiltrowany DataFrame
    """
    if not columns:
        return df
        
    if include_missing:
        return df
    else:
        return df.dropna(subset=columns)

def extract_subtable(df, rows=None, columns=None):
    """
    Ekstrahuje podtablicę na podstawie wybranych numerów/nazw wierszy i kolumn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame źródłowy
    rows : list, optional
        Lista indeksów lub nazw wierszy do wyekstrahowania
    columns : list, optional
        Lista nazw kolumn do wyekstrahowania
        
    Returns:
    --------
    pandas.DataFrame
        Wyekstrahowana podtablica
    """
    try:
        if rows is not None and columns is not None:
            return df.loc[rows, columns]
        elif rows is not None:
            return df.loc[rows, :]
        elif columns is not None:
            return df.loc[:, columns]
        return df
    except Exception as e:
        st.error(f"Błąd podczas ekstrakcji podtablicy: {e}")
        return None

def display_filter_ui(df):
    """
    Wyświetla interfejs użytkownika do filtrowania danych.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
        
    Returns:
    --------
    tuple
        (przefiltrowany_dataframe, czy_filtrowano)
    """
    st.subheader("Filtrowanie danych 🔍")
    
    if df is None or df.empty:
        st.warning("Brak danych do filtrowania. Najpierw wczytaj plik CSV.")
        return df, False
    
    # Inicjalizacja filtrowanego dataframe w session state
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = df.copy()
        st.session_state.filters_applied = False
    
    # Pobieranie typów kolumn
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Tworzenie kontenerów dla różnych typów filtrów
    with st.expander("Filtry dla kolumn numerycznych", expanded=False):  # Zmiana na False
        numeric_filters = {}
        
        for col in numeric_cols:
            st.subheader(f"Filtruj kolumnę: {col}")
            
            # Określenie min i max wartości
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            # Pola do wprowadzenia zakresu
            col1, col2 = st.columns(2)
            with col1:
                use_min = st.checkbox(f"Użyj min. wartości dla {col}", key=f"use_min_{col}")
                min_input = st.number_input(f"Min. wartość dla {col}", 
                                           value=min_val,
                                           min_value=min_val,
                                           max_value=max_val,
                                           key=f"min_{col}")
            
            with col2:
                use_max = st.checkbox(f"Użyj max. wartości dla {col}", key=f"use_max_{col}")
                max_input = st.number_input(f"Max. wartość dla {col}", 
                                           value=max_val,
                                           min_value=min_val,
                                           max_value=max_val,
                                           key=f"max_{col}")
            
            # Zapisanie filtrów
            numeric_filters[col] = {
                'use_min': use_min,
                'min_val': min_input if use_min else None,
                'use_max': use_max,
                'max_val': max_input if use_max else None
            }
    
    with st.expander("Filtry dla kolumn kategorycznych", expanded=False):  # Zmiana na False
        categorical_filters = {}
        
        for col in categorical_cols:
            st.subheader(f"Filtruj kolumnę: {col}")
            
            # Pobranie unikalnych wartości
            unique_values = df[col].dropna().unique().tolist()
            
            # Utworzenie wielokrotnego wyboru
            selected = st.multiselect(
                f"Wybierz wartości dla {col} (puste = wszystkie)",
                options=unique_values,
                default=[],
                key=f"cat_{col}"
            )
            
            # Zapisanie filtrów
            categorical_filters[col] = selected

    
    # Dodaj nowy expander dla ekstrakcji podtablic
    with st.expander("Ekstrakcja podtablicy", expanded=False):  # Zmiana na False
        st.subheader("Wybierz wiersze i kolumny do ekstrakcji")
        
        # Wybór kolumn
        selected_columns = st.multiselect(
            "Wybierz kolumny do ekstrakcji (puste = wszystkie)",
            options=df.columns.tolist(),
            default=[]
        )
        
        # Wybór wierszy przez zakres indeksów
        st.subheader("Wybierz zakres wierszy")
        use_row_range = st.checkbox("Użyj zakresu wierszy")
        
        if use_row_range:
            row_range = st.slider(
                "Zakres wierszy",
                min_value=0,
                max_value=len(df)-1,
                value=(0, min(10, len(df)-1))
            )
            selected_rows = list(range(row_range[0], row_range[1] + 1))
        else:
            selected_rows = None
            
        # Przycisk do ekstrakcji
        if st.button("Ekstrahuj podtablicę"):
            extracted_df = extract_subtable(
                df, 
                rows=selected_rows,
                columns=selected_columns if selected_columns else None
            )
            
            if extracted_df is not None:
                st.success(f"Wyekstrahowano podtablicę o wymiarach: {extracted_df.shape}")
                st.dataframe(extracted_df)
                
                # Opcja pobrania wyekstrahowanej podtablicy
                csv = extracted_df.to_csv(index=False)
                st.download_button(
                    label="Pobierz wyekstrahowaną podtablicę (CSV)",
                    data=csv,
                    file_name="extracted_subtable.csv",
                    mime="text/csv"
                )
    
    # Przycisk do stosowania filtrów - użycie kolumn i kolorowego przycisku
    col1, col2 = st.columns(2)
    with col1:
        filter_button = st.button("🔍 Filtruj dane", key="filter_button", use_container_width=True)
    with col2:
        reset_button = st.button("🔄 Resetuj filtry", key="reset_button", type="tertiary", use_container_width=True)
    
    # Logika filtrowania po naciśnięciu przycisku
    if filter_button:
        filtered_data = df.copy()
        filters_applied = False
        
        # Stosowanie filtrów numerycznych
        for col, filter_settings in numeric_filters.items():
            if filter_settings['use_min'] or filter_settings['use_max']:
                filtered_data = filter_numeric_column(
                    filtered_data, 
                    col, 
                    filter_settings['min_val'], 
                    filter_settings['max_val']
                )
                filters_applied = True
        
        # Stosowanie filtrów kategorycznych
        for col, selected_values in categorical_filters.items():
            if selected_values:
                filtered_data = filter_categorical_column(
                    filtered_data,
                    col,
                    selected_values
                )
                filters_applied = True
        
        # Zapisanie przefiltrowanego dataframe w session state
        st.session_state.filtered_df = filtered_data
        st.session_state.filters_applied = filters_applied
        
        # Wyświetlenie informacji o filtrach
        if filters_applied:
            st.success(f"Dane przefiltrowane. Pozostało {len(filtered_data)} z {len(df)} wierszy ({(len(filtered_data)/len(df)*100):.1f}%).")
        else:
            st.info("Nie zastosowano żadnych filtrów.")
    
    # Resetowanie filtrów
    if reset_button:
        st.session_state.filtered_df = df.copy()
        st.session_state.filters_applied = False
        st.success("Filtry zresetowane.")
    
    # Wyświetlenie podglądu przefiltrowanych danych
    if st.session_state.filters_applied:
        st.subheader("Podgląd przefiltrowanych danych")
        st.dataframe(st.session_state.filtered_df.head())
    
    # Zwrócenie przefiltrowanego dataframe i informacji, czy zastosowano filtry
    return st.session_state.filtered_df, st.session_state.filters_applied
