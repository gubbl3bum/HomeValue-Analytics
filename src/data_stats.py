import pandas as pd
import numpy as np
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
        lambda x: x.skew(),  # skośność
        lambda x: x.kurtosis()  # kurtoza
    ]).round(2)

    # Zmiana nazw dla lambda funkcji
    stats = stats.rename({
        '<lambda_0>': 'skewness',
        '<lambda_1>': 'kurtosis'
    })

    return stats

def compute_correlation_matrix(df, selected_columns):
    """
    Oblicza macierz korelacji między wybranymi kolumnami.
    
    :param df: DataFrame z danymi
    :param selected_columns: Lista kolumn do analizy korelacji
    :return: DataFrame z macierzą korelacji
    """
    if not selected_columns or len(selected_columns) < 2:
        return None

    return df[selected_columns].corr()

def analyze_categorical_columns(df, categorical_columns):
    """
    Analizuje kolumny kategoryczne - oblicza częstotliwość występowania wartości.
    
    :param df: DataFrame z danymi
    :param categorical_columns: Lista kolumn kategorycznych do analizy
    :return: Słownik z DataFrames zawierającymi analizę dla każdej kolumny
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
