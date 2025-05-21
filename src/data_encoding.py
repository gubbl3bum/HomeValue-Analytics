import pandas as pd

def one_hot_encode(df, column):
    """
    One-Hot Encoding dla wybranej kolumny.
    :param df: DataFrame z danymi
    :param column: Nazwa kolumny
    :return: DataFrame z zakodowaną kolumną
    """
    return pd.get_dummies(df, columns=[column])

def target_encode(df, column, target):
    """
    Target Encoding dla wybranej kolumny.

    :param df: DataFrame z danymi
    :param column: Nazwa kolumny do zakodowania (symbolicznej/kategorycznej)
    :param target: Nazwa kolumny docelowej (zwykle liczbowej, np. cena, wartość, wynik)
                   Kolumna docelowa to ta, względem której liczona jest średnia dla każdej kategorii.
                   Przykład: jeśli chcesz zakodować kolumnę 'miasto' względem średniej ceny domu,
                   ustaw 'miasto' jako column, a 'cena' jako target.
    :return: DataFrame z zakodowaną kolumną
    """
    means = df.groupby(column)[target].mean()
    df[column] = df[column].map(means)
    return df
