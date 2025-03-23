import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def create_scatter_plot(df, x_column, y_column, color_column=None, title="Wykres rozrzutu"):
    """
    Tworzy wykres rozrzutu (scatter plot) z podanych danych.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    x_column : str
        Nazwa kolumny dla osi X
    y_column : str
        Nazwa kolumny dla osi Y
    color_column : str, optional
        Nazwa kolumny do kolorowania punktów
    title : str, optional
        Tytuł wykresu
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Obiekt wykresu plotly
    """
    try:
        fig = px.scatter(df, x=x_column, y=y_column, 
                         color=color_column, 
                         title=title,
                         labels={x_column: x_column, y_column: y_column},
                         hover_data=df.columns)
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia wykresu rozrzutu: {e}")
        return None

def create_histogram(df, column, bins=30, title="Histogram"):
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
        fig = px.histogram(df, x=column, 
                           nbins=bins, 
                           title=title,
                           labels={column: column})
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia histogramu: {e}")
        return None

def create_bar_chart(df, x_column, y_column, title="Wykres słupkowy"):
    """
    Tworzy wykres słupkowy dla wybranych kolumn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    x_column : str
        Nazwa kolumny dla osi X (kategoryczna)
    y_column : str
        Nazwa kolumny dla osi Y (numeryczna)
    title : str, optional
        Tytuł wykresu
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Obiekt wykresu plotly
    """
    try:
        # Grupowanie danych
        grouped_data = df.groupby(x_column)[y_column].mean().reset_index()
        
        fig = px.bar(grouped_data, x=x_column, y=y_column, 
                     title=title,
                     labels={x_column: x_column, y_column: f"Średnia {y_column}"})
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
        ["Wykres rozrzutu", "Histogram", "Wykres pudełkowy", "Wykres słupkowy", 
         "Wykres liniowy", "Mapa ciepła korelacji", "Wykres kołowy"]
    )
    
    # Tworzenie wykresu w zależności od typu
    if chart_type == "Wykres rozrzutu":
        x_col = st.selectbox("Wybierz kolumnę dla osi X", numeric_cols)
        y_col = st.selectbox("Wybierz kolumnę dla osi Y", numeric_cols)
        color_col = st.selectbox("Wybierz kolumnę do kolorowania (opcjonalnie)", 
                                ["Brak"] + categorical_cols + numeric_cols)
        color_col = None if color_col == "Brak" else color_col
        
        if st.button("Generuj wykres rozrzutu"):
            fig = create_scatter_plot(df, x_col, y_col, color_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        col = st.selectbox("Wybierz kolumnę", numeric_cols)
        bins = st.slider("Liczba przedziałów", min_value=5, max_value=100, value=30)
        
        if st.button("Generuj histogram"):
            fig = create_histogram(df, col, bins)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Wykres słupkowy":
        if not categorical_cols:
            st.warning("Do wykresu słupkowego potrzebna jest co najmniej jedna kolumna kategoryczna.")
        else:
            x_col = st.selectbox("Wybierz kolumnę kategoryczną (oś X)", categorical_cols)
            y_col = st.selectbox("Wybierz kolumnę numeryczną (oś Y)", numeric_cols)
            
            if st.button("Generuj wykres słupkowy"):
                fig = create_bar_chart(df, x_col, y_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Wykres liniowy":
        x_col = st.selectbox("Wybierz kolumnę dla osi X", all_cols)
        y_col = st.selectbox("Wybierz kolumnę dla osi Y", numeric_cols)
        
        if st.button("Generuj wykres liniowy"):
            fig = create_line_chart(df, x_col, y_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Mapa ciepła korelacji":
        selected_cols = st.multiselect("Wybierz kolumny numeryczne", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        
        if st.button("Generuj mapę ciepła"):
            if len(selected_cols) < 2:
                st.warning("Wybierz co najmniej dwie kolumny numeryczne.")
            else:
                fig = create_heatmap(df, selected_cols)
                if fig:
                    st.pyplot(fig)
    
    elif chart_type == "Wykres kołowy":
        if not categorical_cols:
            st.warning("Do wykresu kołowego potrzebna jest co najmniej jedna kolumna kategoryczna.")
        else:
            col = st.selectbox("Wybierz kolumnę kategoryczną", categorical_cols)
            
            if st.button("Generuj wykres kołowy"):
                fig = create_pie_chart(df, col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)