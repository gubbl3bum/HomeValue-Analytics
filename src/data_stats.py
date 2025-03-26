def compute_basic_statistics(df, columns):
    """
    Calculates basic statistics for selected numeric columns.

    Parameters:
    -----------
    df : pandas.DataFrame
    A DataFrame with data
    columns : list
    A list of columns to analyze

    Returns:
    --------
    pandas.DataFrame or None
    A DataFrame with statistics or None if there are no columns to analyze
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
        'skew',  
        'kurt'   
    ]).round(2)

    # Translation of statistics names into Polish
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

def compute_correlation_matrix(df, selected_columns):
    """
    Calculates the correlation matrix between selected columns.
    :param df: DataFrame with data
    :param selected_columns: List of columns to analyze correlation
    :return: DataFrame with correlation matrix
    """
    if not selected_columns or len(selected_columns) < 2:
        return None

    return df[selected_columns].corr()

def analyze_categorical_columns(df, categorical_columns):
    """
    Analyzes categorical columns - calculates the frequency of values.

    :param df: DataFrame with data
    :param categorical_columns: List of categorical columns to analyze
    :return: Dictionary with DataFrames containing the analysis for each column
    """
    if not categorical_columns:
        return None

    results = {}

    for col in categorical_columns:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / len(df)) * 100
        results[col] = value_counts

    return results

def analyze_price_per_sqm(df, price_column, area_column):
    """
    Calculates price per square meter and basic statistics.

    :param df: DataFrame with data
    :param price_column: Name of the price column
    :param area_column: Name of the area column
    :return: DataFrame with statistics of price per square meter
    """
    if price_column not in df.columns or area_column not in df.columns:
        return None

    # Avoiding division by zero
    valid_data = df[(df[area_column] > 0) & df[price_column].notna()]

    if len(valid_data) == 0:
        return None

    valid_data['price_per_sqm'] = valid_data[price_column] / valid_data[area_column]

    stats = valid_data['price_per_sqm'].describe().to_frame().T
    stats['missing_values'] = len(df) - len(valid_data)
    stats['missing_percentage'] = (stats['missing_values'] / len(df)) * 100

    return stats
