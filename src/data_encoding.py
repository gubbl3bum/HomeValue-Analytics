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
    :param column: Nazwa kolumny
    :param target: Nazwa kolumny docelowej
    :return: DataFrame z zakodowaną kolumną
    """
    means = df.groupby(column)[target].mean()
    df[column] = df[column].map(means)
    return df
