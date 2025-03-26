import pandas as pd
import streamlit as st

def load_csv_file(uploaded_file):
    """
    Loads data from a CSV file.

    Parameters:
    -----------
    uploaded_file : UploadedFile
    CSV file uploaded by user

    Returns:
    --------
    pandas.DataFrame or None
    DataFrame with loaded data or None on error
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku: {e}")
        return None

def preview_data(df):
    """
    Displays an editable preview of the data.
    """
    if df is not None:
        st.write("Podgląd danych (kliknij w komórkę aby edytować):")
        
        # Use data_editor instead of dataframe
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False
        )
        
        # If the data has changed, we update the DataFrame
        if not edited_df.equals(df):
            st.success("Dane zostały zaktualizowane!")
            return edited_df
        
        return df
    return None

def get_numeric_columns(df):
    """
    Returns a list of numeric columns from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
    A DataFrame with data

    Returns:
    --------
    list
    A list of numeric column names
    """
    if df is not None:
        return df.select_dtypes(include=["number"]).columns.tolist()
    return []

def get_categorical_columns(df):
    """
    Returns a list of categorical columns from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
    DataFrame with data

    Returns:
    --------
    list
    List of categorical column names
    """
    if df is not None:
        return df.select_dtypes(include=["object", "category"]).columns.tolist()
    return []

def get_datetime_columns(df):
    """
    Returns a list of date columns from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
    A DataFrame with data

    Returns:
    --------
    list
    A list of date column names
    """
    if df is not None:
        return df.select_dtypes(include=["datetime"]).columns.tolist()
    return []

def get_column_types(df):
    """
    Returns a dictionary of column types.

    Parameters:
    -----------
    df : pandas.DataFrame
    DataFrame with data

    Returns:
    --------
    dict
    A dictionary of column types
    """
    if df is not None:
        return {
            'numeric': get_numeric_columns(df),
            'categorical': get_categorical_columns(df),
            'datetime': get_datetime_columns(df)
        }
    return {'numeric': [], 'categorical': [], 'datetime': []}