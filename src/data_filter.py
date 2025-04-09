import streamlit as st
from data_loader import preview_data

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

def parse_range_string(range_str, max_idx):
    """
    Parsuje string z zakresami do listy indeksów.
    
    Obsługiwane formaty:
    - "1:10" - zakres od 1 do 10
    - "1,5,9" - konkretne wiersze
    - "1:10,20:30" - multiple ranges
    - "-10:" - ostatnie 10 wierszy
    - ":10" - pierwsze 10 wierszy
    
    Parameters:
    -----------
    range_str : str
        String z zakresami
    max_idx : int
        Maksymalny dozwolony indeks
    
    Returns:
    --------
    list
        Lista indeksów
    """
    if not range_str.strip():
        return []

    indices = set()
    
    # Dzielimy na części po przecinku
    parts = range_str.split(',')

    for part in parts:
        part = part.strip()
        if ':' in part:  # Zakres typu "start:end"
            start, end = part.split(':')

            # Obsługa ujemnych indeksów i pustych wartości
            if not start:  # ":n" - pierwsze n wierszy
                start = 0
            else:
                start = int(start)
                if start < 0:
                    start = max(0, max_idx + start)
    
            if not end:  # "n:" - do końca
                end = max_idx
            else:
                end = int(end)
                if end < 0:
                    end = max_idx + end

            indices.update(range(start, min(end + 1, max_idx + 1)))

        else:  # Pojedynczy indeks
            idx = int(part)
            if idx < 0:
                idx = max_idx + idx
            if 0 <= idx <= max_idx:
                indices.add(idx)
   
    return sorted(list(indices))

def display_subtable_ui(df):
    """
    Wyświetla interfejs użytkownika do ekstrakcji podtablic.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame źródłowy
        
    Returns:
    --------
    tuple
        (wynikowy DataFrame, bool czy dokonano ekstrakcji)
    """
    st.subheader("Ekstrakcja podtablicy 📋")

    if df is None or df.empty:
        st.warning("Brak danych do ekstrakcji. Najpierw wczytaj plik CSV.")
        return df, False

    # Initialize session state
    if 'extracted_df' not in st.session_state:
        st.session_state.extracted_df = df.copy()
        st.session_state.extraction_applied = False

    # Column selection
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Wybierz kolumny:",
        options=all_columns,
        default=all_columns[:3] if len(all_columns) > 3 else all_columns
    )

    # Row range selection
    st.subheader("Wybór wierszy")
    st.markdown("""
    Wprowadź zakresy wierszy w jednym z formatów:
    - `1:10` - wiersze od 1 do 10
    - `1,5,9` - konkretne wiersze 1, 5 i 9
    - `1:10,20:30` - wiersze od 1 do 10 oraz od 20 do 30
    - `-10:` - ostatnie 10 wierszy
    - `:10` - pierwsze 10 wiersze
    """)

    range_str = st.text_input(
        "Zakresy wierszy:",
        value=":10",
        help="Wprowadź zakresy wierszy w podanym formacie"
    )

    # Extract button
    if st.button("📋 Wyodrębnij podtablicę", use_container_width=True):
        if not selected_columns:
            st.warning("Wybierz co najmniej jedną kolumnę.")
            return st.session_state.extracted_df, False

        try:
            selected_rows = parse_range_string(range_str, len(df) - 1)
            if not selected_rows:
                st.warning("Nie wybrano żadnego zakresu wierszy.")
                return st.session_state.extracted_df, False

            extracted_data = extract_subtable(df, rows=selected_rows, columns=selected_columns)

            if extracted_data is not None:
                st.session_state.extracted_df = extracted_data
                st.session_state.extraction_applied = True
                st.success(f"Wyodrębniono podtablicę o wymiarach: {extracted_data.shape}")
                st.write(f"Wybrane wiersze: {len(selected_rows)}")
  
        except ValueError as e:
            st.error(f"Błąd w formacie zakresów: {str(e)}")
            return st.session_state.extracted_df, False

    # Reset button
    if st.button("🔄 Resetuj", type="secondary", use_container_width=True):
        st.session_state.extracted_df = df.copy()
        st.session_state.extraction_applied = False
        st.success("Zresetowano do oryginalnych danych.")

    # Display preview using the same formatting as in data_loader
    if st.session_state.extraction_applied:
        st.subheader("Podgląd wyodrębnionej podtablicy")
        preview_data(st.session_state.extracted_df)

    return st.session_state.extracted_df, st.session_state.extraction_applied
