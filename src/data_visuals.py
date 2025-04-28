import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np

def calculate_default_bins(df, column):
    """
    Oblicza sugerowaną liczbę przedziałów na podstawie danych.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    column : str
        Nazwa kolumny do analizy
    
    Returns:
    --------
    int
        Sugerowana liczba przedziałów
    """
    data = df[column].dropna()
    
    if pd.api.types.is_integer_dtype(df[column]):
        # Dla danych całkowitych - unikalne wartości
        return min(int(data.max() - data.min() + 1), 50)
    else:
        # Dla danych ciągłych - reguła Sturges'a
        n = len(data)
        return min(int(1 + 3.322 * np.log10(n)), 50)

def create_histogram(df, column, bins=None, title="Histogram"):
    """
    Tworzy histogram dla wybranej kolumny.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    column : str
        Nazwa kolumny do analizy
    bins : int, optional
        Liczba przedziałów histogramu
    title : str, optional
        Tytuł wykresu
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Obiekt wykresu plotly
    """
    try:
        if bins is None:
            bins = calculate_default_bins(df, column)
            
        fig = px.histogram(df, x=column, 
                           nbins=bins, 
                           title=title,
                           labels={column: column})
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia histogramu: {e}")
        return None

def create_bar_chart(df, x_column, y_column, agg_func='mean', title=None):
    """
    Tworzy rozszerzony wykres słupkowy dla wybranych kolumn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    x_column : str
        Nazwa kolumny dla osi X (kategoryczna)
    y_column : str
        Nazwa kolumny dla osi Y (numeryczna)
    agg_func : str, optional
        Funkcja agregująca ('mean', 'median', 'sum', 'count')
    title : str, optional
        Tytuł wykresu
    """
    try:
        # Słownik funkcji agregujących
        agg_functions = {
            'mean': ('średnia', np.mean),
            'median': ('mediana', np.median),
            'sum': ('suma', np.sum),
            'count': ('liczba', 'count')
        }
        
        # Grupowanie danych
        func_name, func = agg_functions[agg_func]
        grouped_data = df.groupby(x_column)[y_column].agg(func).reset_index()
        
        # Sortowanie wartości
        grouped_data = grouped_data.sort_values(y_column, ascending=False)
        
        # Tworzenie tytułu jeśli nie podano
        if title is None:
            title = f"{func_name.capitalize()} {y_column} według {x_column}"
        
        fig = px.bar(grouped_data, 
                    x=x_column, 
                    y=y_column,
                    title=title,
                    labels={
                        x_column: x_column,
                        y_column: f"{func_name.capitalize()} {y_column}"
                    })
        
        # Formatowanie osi jeśli wartości są całkowite
        if pd.api.types.is_integer_dtype(df[x_column]):
            fig.update_xaxes(tickformat="d")
        
        # Dodanie etykiet wartości nad słupkami
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia wykresu słupkowego: {e}")
        return None

def create_line_chart(df, x_column, y_column, title="Wykres liniowy"):
    """
    Tworzy wykres liniowy dla wybranych kolumn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    x_column : str
        Nazwa kolumny dla osi X
    y_column : str
        Nazwa kolumny dla osi Y
    title : str, optional
        Tytuł wykresu
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Obiekt wykresu plotly
    """
    try:
        fig = px.line(df, x=x_column, y=y_column, 
                      title=title,
                      labels={x_column: x_column, y_column: y_column})
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia wykresu liniowego: {e}")
        return None

def create_heatmap(df, columns=None, title="Mapa ciepła korelacji"):
    """
    Tworzy mapę ciepła korelacji między wybranymi kolumnami.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    columns : list, optional
        Lista nazw kolumn do uwzględnienia (domyślnie wszystkie numeryczne)
    title : str, optional
        Tytuł wykresu
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Obiekt wykresu matplotlib
    """
    try:
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        correlation = df[columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        plt.title(title)
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia mapy ciepła: {e}")
        return None

def create_pie_chart(df, column, title="Wykres kołowy"):
    """
    Tworzy wykres kołowy dla wybranej kolumny kategorycznej.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    column : str
        Nazwa kolumny kategorycznej
    title : str, optional
        Tytuł wykresu
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Obiekt wykresu plotly
    """
    try:
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        fig = px.pie(value_counts, values='count', names=column, 
                     title=title)
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia wykresu kołowego: {e}")
        return None

def create_category_frequency_plot(df, column, top_n=10, title=None):
    """
    Tworzy wykres słupkowy częstości dla kolumny kategorycznej.
    """
    try:
        # Obliczanie częstości
        value_counts = df[column].value_counts()
        
        # Wybieranie top N kategorii
        if len(value_counts) > top_n:
            other_count = value_counts[top_n:].sum()
            value_counts = value_counts[:top_n]
            value_counts['Pozostałe'] = other_count
        
        # Tworzenie DataFrame dla wykresu
        plot_df = pd.DataFrame({
            'Kategoria': value_counts.index,
            'Liczba': value_counts.values,
            'Procent': (value_counts.values / len(df)) * 100
        })
        
        # Formatowanie tytułu
        if title is None:
            title = f"Top {top_n} najczęstszych wartości w kolumnie {column}"
            if len(value_counts) <= top_n:
                title = f"Wszystkie wartości w kolumnie {column}"
        
        # Tworzenie wykresu
        fig = px.bar(plot_df, 
                    x='Kategoria', 
                    y='Liczba',
                    text='Procent',
                    title=title)
        
        # Formatowanie etykiet procentowych
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        # Jeśli wartości są liczbami całkowitymi, ustaw odpowiedni format osi
        if pd.api.types.is_integer_dtype(df[column]):
            fig.update_xaxes(tickformat="d")
        
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia wykresu częstości: {e}")
        return None

def create_category_comparison_plot(df, cat_column, value_column, top_n=10, title=None):
    """
    Tworzy wykres słupkowy średnich wartości numerycznych dla kategorii.
    """
    try:
        # Obliczanie średnich dla każdej kategorii
        agg_data = df.groupby(cat_column)[value_column].agg(['mean', 'count']).reset_index()
        agg_data = agg_data.sort_values('count', ascending=False)
        
        # Wybieranie top N kategorii
        if len(agg_data) > top_n:
            agg_data = agg_data.head(top_n)
        
        # Tworzenie wykresu
        fig = px.bar(agg_data, 
                    x=cat_column, 
                    y='mean',
                    title=title or f"Średnia {value_column} dla top {top_n} kategorii w {cat_column}")
        
        # Jeśli wartości są liczbami całkowitymi, ustaw odpowiedni format osi
        if pd.api.types.is_integer_dtype(df[cat_column]):
            fig.update_xaxes(tickformat="d")
        
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia wykresu porównawczego: {e}")
        return None

def create_violin_plot(df, x_column, y_column, title="Violin Plot"):
    """
    Tworzy wykres violin plot.
    """
    fig = px.violin(df, x=x_column, y=y_column, box=True, points="all", title=title)
    return fig

def create_pair_plot(df, columns, title="Pair Plot"):
    """
    Tworzy wykres pair plot.
    """
    fig = sns.pairplot(df[columns])
    fig.fig.suptitle(title, y=1.02)
    return fig

def display_chart_ui(df):
    """
    Wyświetla interfejs użytkownika do tworzenia wykresów.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    """
    st.subheader("Wizualizacja danych 📊")
    
    if df is None or df.empty:
        st.warning("Brak danych do wizualizacji. Najpierw wczytaj plik CSV.")
        return
    
    # Pobieranie typów kolumn
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    if not numeric_cols:
        st.warning("Brak kolumn numerycznych do wizualizacji.")
        return
    
    # Wybór typu wykresu
    chart_type = st.selectbox(
        "Wybierz typ wykresu",
        ["Histogram", "Wykres słupkowy", 
         "Mapa ciepła korelacji", "Wykres kołowy", "Wykres częstości kategorii", "Wykres porównawczy kategorii"]
    )
    
    # Tworzenie wykresu w zależności od typu
    if chart_type == "Histogram":
        st.write("""
        ### Histogram - instrukcja:
        - Wybierz zmienną numeryczną do analizy rozkładu
        - Dostosuj liczbę przedziałów (bins) aby lepiej zobaczyć rozkład danych
        - Wykres pokaże:
            - Rozkład wartości zmiennej
            - Częstość występowania wartości w przedziałach
            - Możliwe skupienia i wartości odstające
            - Kształt rozkładu (normalny, skośny, etc.)
        """)
        col = st.selectbox("Wybierz kolumnę", numeric_cols)
        bins = st.slider("Liczba przedziałów", min_value=5, max_value=100, value=30)
        
        if st.button("Generuj histogram"):
            fig = create_histogram(df, col, bins)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Wykres słupkowy":
        st.write("""
        ### Wykres słupkowy - instrukcja:
        - Wybierz zmienną kategoryczną (oś X)
        - Wybierz zmienną numeryczną (oś Y)
        - Wybierz funkcję agregującą:
            - Średnia: średnia wartość dla każdej kategorii
            - Mediana: wartość środkowa dla każdej kategorii
            - Suma: suma wartości dla każdej kategorii
            - Liczba wystąpień: liczba elementów w każdej kategorii
        - Oś OY przedstawia wartości numeryczne po zastosowaniu wybranej funkcji agregującej.
        """)
        if not categorical_cols:
            st.warning("Do wykresu słupkowego potrzebna jest co najmniej jedna kolumna kategoryczna.")
        else:
            x_col = st.selectbox("Wybierz kolumnę kategoryczną (oś X)", categorical_cols)
            y_col = st.selectbox("Wybierz kolumnę numeryczną (oś Y)", numeric_cols)
            agg_func = st.selectbox(
                "Wybierz funkcję agregującą",
                ['mean', 'median', 'sum', 'count'],
                format_func=lambda x: {
                    'mean': 'Średnia',
                    'median': 'Mediana',
                    'sum': 'Suma',
                    'count': 'Liczba wystąpień'
                }[x]
            )
            
            if st.button("Generuj wykres słupkowy"):
                fig = create_bar_chart(df, x_col, y_col, agg_func)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Mapa ciepła korelacji":
        st.write("""
        ### Mapa ciepła korelacji - instrukcja:
        - Wybierz zmienne numeryczne do analizy korelacji
        - Wykres pokaże:
            - Siłę korelacji między wszystkimi wybranymi zmiennymi
            - Wartości od -1 (silna korelacja ujemna) do 1 (silna korelacja dodatnia)
            - 0 oznacza brak korelacji
            - Kolory ciepłe (czerwone) - korelacja dodatnia
            - Kolory zimne (niebieskie) - korelacja ujemna
        - Użyteczne do:
            - Identyfikacji silnie skorelowanych zmiennych
            - Wykrywania wzorców w danych
            - Wyboru zmiennych do analiz
        """)
        selected_cols = st.multiselect("Wybierz kolumny numeryczne", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        
        if st.button("Generuj mapę ciepła"):
            if len(selected_cols) < 2:
                st.warning("Wybierz co najmniej dwie kolumny numeryczne.")
            else:
                fig = create_heatmap(df, selected_cols)
                if fig:
                    st.pyplot(fig)
    
    elif chart_type == "Wykres kołowy":
        st.write("""
        ### Wykres kołowy - instrukcja:
        - Wybierz zmienną kategoryczną
        - Wykres pokaże:
            - Procentowy udział każdej kategorii w całości
            - Względne proporcje między kategoriami
        - Najlepszy do:
            - Pokazania struktury danych
            - Porównania udziałów poszczególnych kategorii
            - Danych, gdzie suma wszystkich części daje 100%
        """)
        if not categorical_cols:
            st.warning("Do wykresu kołowego potrzebna jest co najmniej jedna kolumna kategoryczna.")
        else:
            col = st.selectbox("Wybierz kolumnę kategoryczną", categorical_cols)
            
            if st.button("Generuj wykres kołowy"):
                fig = create_pie_chart(df, col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Wykres częstości kategorii":
        st.write("""
        ### Wykres częstości kategorii - instrukcja:
        - Wybierz zmienną kategoryczną do analizy
        - Określ liczbę najczęstszych kategorii do wyświetlenia
        - Wykres pokaże:
            - Liczbę wystąpień każdej kategorii
            - Procentowy udział każdej kategorii
            - Ranking kategorii od najczęstszej do najrzadszej
        - Idealny do:
            - Analizy popularności kategorii
            - Identyfikacji dominujących wartości
            - Wykrywania rzadkich przypadków
        """)
        cat_col = st.selectbox("Wybierz kolumnę kategoryczną", categorical_cols, key='freq')
        top_n = st.slider("Liczba najczęstszych kategorii", 
                         min_value=5, 
                         max_value=50, 
                         value=10,
                         key='freq_slider')
        
        if st.button("Generuj wykres częstości", key='freq_btn'):
            fig = create_category_frequency_plot(df, cat_col, top_n)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Wykres porównawczy kategorii":
        st.write("""
        ### Wykres porównawczy kategorii - instrukcja:
        - Wybierz zmienną kategoryczną do grupowania
        - Wybierz zmienną numeryczną do porównania
        - Określ liczbę kategorii do pokazania
        - Wykres pokaże:
            - Średnią wartość zmiennej numerycznej dla każdej kategorii
            - Porównanie kategorii według wybranej miary
            - Top N najczęstszych kategorii
        - Przydatny do:
            - Porównywania średnich wartości między kategoriami
            - Identyfikacji kategorii o najwyższych/najniższych wartościach
            - Analizy zależności między zmienną kategoryczną a numeryczną
        """)
        if not categorical_cols:
            st.warning("Brak kolumn kategorycznych do wizualizacji.")
        else:
            cat_col_2 = st.selectbox("Wybierz kolumnę kategoryczną", categorical_cols, key='comp')
            num_col = st.selectbox("Wybierz kolumnę numeryczną", numeric_cols, key='comp_num')
            top_n_2 = st.slider("Liczba kategorii", 
                               min_value=5, 
                               max_value=50, 
                               value=10,
                               key='comp_slider')
            
            if st.button("Generuj wykres porównawczy", key='comp_btn'):
                fig = create_category_comparison_plot(df, cat_col_2, num_col, top_n_2)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)