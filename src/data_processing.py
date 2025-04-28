import pandas as pd  # Add this import

def extract_subtable(df, columns=None, mode='remove'):
    """
    Ekstrakcja podtablicy na podstawie kolumn.
    :param df: DataFrame z danymi
    :param columns: Lista kolumn do pozostawienia/usunięcia
    :param mode: Tryb działania ('remove' lub 'keep')
    :return: DataFrame z podtablicą
    """
    if mode == 'keep':
        return df[columns]
    elif mode == 'remove':
        return df.drop(columns=columns, errors='ignore')
    return df

def replace_values(df, column, to_replace, value):
    """
    Zamienia wartości w wybranej kolumnie.
    :param df: DataFrame z danymi
    :param column: Nazwa kolumny
    :param to_replace: Wartość do zamiany
    :param value: Nowa wartość
    :return: DataFrame z zamienionymi wartościami
    """
    df[column] = df[column].replace(to_replace, value)
    return df

def fill_missing_values(df, column, method='mean'):
    """
    Wypełnia brakujące wartości w kolumnie.
    :param df: DataFrame z danymi
    :param column: Nazwa kolumny
    :param method: Metoda ('mean', 'median', 'mode')
    :return: DataFrame z wypełnionymi wartościami
    """
    if column not in df.columns:
        raise ValueError(f"Kolumna '{column}' nie istnieje w DataFrame.")

    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric columns, including integers (e.g., years)
        if method == 'mean':
            fill_value = df[column].mean()
        elif method == 'median':
            fill_value = df[column].median()
        elif method == 'mode':
            fill_value = df[column].mode()[0]
        else:
            raise ValueError("Nieprawidłowa metoda dla kolumn numerycznych: wybierz 'mean', 'median' lub 'mode'.")

        # Ensure integer columns remain as integers
        if pd.api.types.is_integer_dtype(df[column]):
            fill_value = int(fill_value)
            df[column] = df[column].fillna(fill_value).astype('Int64')
        else:
            df[column] = df[column].fillna(fill_value)

    elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
        # Categorical columns
        if method == 'mode':
            fill_value = df[column].mode()[0]
            df[column] = df[column].fillna(fill_value)
        else:
            raise ValueError("Nieprawidłowa metoda dla kolumn kategorycznych: wybierz 'mode'.")
    else:
        raise ValueError(f"Nieobsługiwany typ danych dla kolumny '{column}'.")

    return df
