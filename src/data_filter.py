import streamlit as st
import pandas as pd
import numpy as np

def filter_numeric_column(df, column, min_value=None, max_value=None):
    """
    Filters the DataFrame by the minimum and maximum values ​​for a numeric column.

    Parameters:
    -----------
    df : pandas.DataFrame
    DataFrame with data
    column : str
    Name of the numeric column to filter
    min_value : float, optional
    Minimum value (inclusive)
    max_value : float, optional
    Maximum value (inclusive)

    Returns:
    --------
    pandas.DataFrame
    Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if min_value is not None:
        filtered_df = filtered_df[filtered_df[column] >= min_value]
        
    if max_value is not None:
        filtered_df = filtered_df[filtered_df[column] <= max_value]
        
    return filtered_df

def filter_categorical_column(df, column, selected_values):
    """
    Filters the DataFrame by selected values ​​for a categorical column.

    Parameters:
    -----------
    df : pandas.DataFrame
    DataFrame with data
    column : str
    Name of the categorical column to filter
    selected_values ​​: list
    List of selected values

    Returns:
    --------
    pandas.DataFrame
    Filtered DataFrame
    """
    if not selected_values:
        return df
        
    return df[df[column].isin(selected_values)]

def filter_by_missing_values(df, columns, include_missing=True):
    """
    Filters DataFrame by missing values ​​in selected columns.

    Parameters:
    -----------
    df : pandas.DataFrame
    DataFrame with data
    columns : list
    List of column names to check
    include_missing : bool
    If True, keeps rows with missing values,
    if False, removes rows with missing values

    Returns:
    --------
    pandas.DataFrame
    Filtered DataFrame
    """
    if not columns:
        return df

    if include_missing:
        return df
    else:
        return df.dropna(subset=columns)

def extract_subtable(df, rows=None, columns=None):
    """
    Extracts a subarray based on the selected row and column numbers/names.

    Parameters:
    -----------
    df : pandas.DataFrame
    Source DataFrame
    rows : list, optional
    List of row indexes or names to extract
    columns : list, optional
    List of column names to extract

    Returns:
    --------
    pandas.DataFrame
    Extracted subarray
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
    Displays a user interface for filtering data.

    Parameters:
    -----------
    df : pandas.DataFrame
    A DataFrame with data

    Returns:
    --------
    tuple
    (filtered_dataframe, is_filtered)
    """
    st.subheader("Filtrowanie danych 🔍")
    
    if df is None or df.empty:
        st.warning("Brak danych do filtrowania. Najpierw wczytaj plik CSV.")
        return df, False
    
    # Initialize filtered dataframe in session state
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = df.copy()
        st.session_state.filters_applied = False
    
    #Getting column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Creating containers for different types of filters
    with st.expander("Filtry dla kolumn numerycznych", expanded=False):  # Change to False
        numeric_filters = {}
        
        for col in numeric_cols:
            st.subheader(f"Filtruj kolumnę: {col}")
            
            # Determine min and max values
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            # Fields to enter the range
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
            
            # Saving filters
            numeric_filters[col] = {
                'use_min': use_min,
                'min_val': min_input if use_min else None,
                'use_max': use_max,
                'max_val': max_input if use_max else None
            }
    
    with st.expander("Filtry dla kolumn kategorycznych", expanded=False):  # Change to False
        categorical_filters = {}
        
        for col in categorical_cols:
            st.subheader(f"Filtruj kolumnę: {col}")
            
            # Getting unique values
            unique_values = df[col].dropna().unique().tolist()
            
            # Create multiple selection
            selected = st.multiselect(
                f"Wybierz wartości dla {col} (puste = wszystkie)",
                options=unique_values,
                default=[],
                key=f"cat_{col}"
            )
            
            # Saving filters
            categorical_filters[col] = selected

    
    # Add new expander for subarray extraction
    with st.expander("Ekstrakcja podtablicy", expanded=False):  # Change to False
        st.subheader("Wybierz wiersze i kolumny do ekstrakcji")
        
        # Column selection
        selected_columns = st.multiselect(
            "Wybierz kolumny do ekstrakcji (puste = wszystkie)",
            options=df.columns.tolist(),
            default=[]
        )
        
        # Selecting rows by index range
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
            
        # Extraction button
        if st.button("Ekstrahuj podtablicę"):
            extracted_df = extract_subtable(
                df, 
                rows=selected_rows,
                columns=selected_columns if selected_columns else None
            )
            
            if extracted_df is not None:
                st.success(f"Wyekstrahowano podtablicę o wymiarach: {extracted_df.shape}")
                st.dataframe(extracted_df)
                
                # Option to download extracted subarray
                csv = extracted_df.to_csv(index=False)
                st.download_button(
                    label="Pobierz wyekstrahowaną podtablicę (CSV)",
                    data=csv,
                    file_name="extracted_subtable.csv",
                    mime="text/csv"
                )
    
    # Button for applying filters - using columns and a colored button
    col1, col2 = st.columns(2)
    with col1:
        filter_button = st.button("🔍 Filtruj dane", key="filter_button", use_container_width=True)
    with col2:
        reset_button = st.button("🔄 Resetuj filtry", key="reset_button", type="tertiary", use_container_width=True)
    
    # Filter logic at the push of a button
    if filter_button:
        filtered_data = df.copy()
        filters_applied = False
        
        # Using numerical filters
        for col, filter_settings in numeric_filters.items():
            if filter_settings['use_min'] or filter_settings['use_max']:
                filtered_data = filter_numeric_column(
                    filtered_data, 
                    col, 
                    filter_settings['min_val'], 
                    filter_settings['max_val']
                )
                filters_applied = True
        
        # Using categorical filters
        for col, selected_values in categorical_filters.items():
            if selected_values:
                filtered_data = filter_categorical_column(
                    filtered_data,
                    col,
                    selected_values
                )
                filters_applied = True
        
        # Saving the filtered dataframe in session state
        st.session_state.filtered_df = filtered_data
        st.session_state.filters_applied = filters_applied
        
        # Displaying information about filters
        if filters_applied:
            st.success(f"Dane przefiltrowane. Pozostało {len(filtered_data)} z {len(df)} wierszy ({(len(filtered_data)/len(df)*100):.1f}%).")
        else:
            st.info("Nie zastosowano żadnych filtrów.")
    
    # Resetting filters
    if reset_button:
        st.session_state.filtered_df = df.copy()
        st.session_state.filters_applied = False
        st.success("Filtry zresetowane.")
    
    # Preview of filtered data
    if st.session_state.filters_applied:
        st.subheader("Podgląd przefiltrowanych danych")
        st.dataframe(st.session_state.filtered_df.head())
    
    # Returns the filtered dataframe and whether any filters are applied
    return st.session_state.filtered_df, st.session_state.filters_applied
