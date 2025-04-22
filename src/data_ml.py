import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff
import plotly.express as px

class ClassificationModel:
    def __init__(self):
        self.feature_encoder = None
        self.target_encoder = None
        self.model = None
        self.feature_names = None
        self.class_names = None
    
    def prepare_features(self, X, categorical_features, fit=True):
        """
        Przygotowuje cechy do klasyfikacji, kodując kolumny kategoryczne.
        """
        if not categorical_features:
            return X.values, X.columns.tolist()
        
        try:
            # Kodowanie kolumn kategorycznych
            if fit:
                # Dla danych treningowych - dopasowanie i transformacja
                self.feature_encoder = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', [col for col in X.columns if col not in categorical_features]),
                        ('cat', OneHotEncoder(
                            drop='first', 
                            sparse_output=False,
                            handle_unknown='ignore'  # Dodano obsługę nieznanych kategorii
                        ), categorical_features)
                    ]
                )
                X_encoded = self.feature_encoder.fit_transform(X)
                
                # Generowanie nazw cech
                numeric_features = [col for col in X.columns if col not in categorical_features]
                categorical_names = []
                
                if categorical_features:
                    encoder = self.feature_encoder.named_transformers_['cat']
                    for feature in categorical_features:
                        # Pobierz unikalne wartości dla każdej cechy kategorycznej
                        unique_values = sorted(X[feature].dropna().unique())
                        # Pomiń pierwszą wartość (drop='first')
                        if len(unique_values) > 1:
                            feature_values = [f"{feature}_{val}" for val in unique_values[1:]]
                            categorical_names.extend(feature_values)
                
                self.feature_names = numeric_features + categorical_names
                return X_encoded, self.feature_names
            else:
                # Dla danych testowych - tylko transformacja
                if self.feature_encoder is None:
                    raise ValueError("Feature encoder not fitted. Call with fit=True first.")
                X_encoded = self.feature_encoder.transform(X)
                return X_encoded, self.feature_names
                
        except Exception as e:
            st.error(f"Błąd podczas przygotowania cech: {str(e)}")
            st.write("X shape:", X.shape)
            st.write("Categorical features:", categorical_features)
            st.write("X columns:", X.columns.tolist())
            raise e
    
    def prepare_target(self, y):
        """
        Przygotowuje zmienną docelową, kodując etykiety.
        """
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        self.class_names = self.target_encoder.classes_
        return y_encoded
    
    def train(self, X, y, model_params):
        """
        Trenuje model Random Forest z podanymi parametrami.
        """
        self.model = RandomForestClassifier(**model_params)
        self.model.fit(X, y)
    
    def evaluate(self, X, y, X_test=None, y_test=None, cv_folds=5):
        """
        Przeprowadza ewaluację modelu różnymi metodami.
        """
        results = {}
        
        try:
            # Walidacja krzyżowa
            cv_scores = cross_val_score(self.model, X, y, cv=cv_folds)
            results['cv_scores'] = cv_scores
            
            # Leave-one-out validation
            loo = LeaveOneOut()
            loo_scores = cross_val_score(self.model, X, y, cv=loo)
            results['loo_scores'] = loo_scores
            
            # Testowanie na zbiorze testowym
            if X_test is not None and y_test is not None:
                y_pred = self.model.predict(X_test)
                results['test_predictions'] = y_pred
                results['test_score'] = self.model.score(X_test, y_test)
                
                # Obliczanie macierzy pomyłek
                conf_matrix = confusion_matrix(y_test, y_pred)
                results['confusion_matrix'] = conf_matrix
                
                # Tworzenie raportu klasyfikacji
                report = classification_report(
                    y_test, y_pred,
                    target_names=self.class_names,
                    output_dict=True
                )
                results['classification_report'] = report
            
            # Ważność cech
            if hasattr(self.model, 'feature_importances_'):
                results['feature_importance'] = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            return results
            
        except Exception as e:
            st.error(f"Błąd podczas ewaluacji modelu: {str(e)}")
            return None

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Tworzy wizualizację macierzy pomyłek bez kolorów.
    """
    try:
        # Konwersja macierzy pomyłek na numpy array
        conf_matrix = np.array(conf_matrix)
        
        # Tworzenie wykresu
        fig = ff.create_annotated_heatmap(
            z=conf_matrix,
            x=list(class_names),  # Konwersja na listę
            y=list(class_names),  # Konwersja na listę
            showscale=False  # Usunięcie skali kolorów
        )
        
        # Aktualizacja układu
        fig.update_layout(
            title='Macierz pomyłek',
            xaxis_title='Przewidziana klasa',
            yaxis_title='Rzeczywista klasa',
            xaxis={'side': 'bottom'}  # Przesunięcie etykiet osi X na dół
        )
        
        # Dostosowanie pozycji adnotacji
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 10
        
        return fig
    except Exception as e:
        st.error(f"Błąd podczas tworzenia macierzy pomyłek: {e}")
        return None

def display_ml_ui(df):
    """
    Wyświetla interfejs użytkownika do klasyfikacji.
    """
    st.subheader("Klasyfikacja 🤖")
    
    if df is None or df.empty:
        st.warning("Brak danych do analizy. Najpierw wczytaj plik CSV.")
        return
    
    # Pobieranie typów kolumn
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Wybór atrybutów
    st.write("### 1. Wybór atrybutów")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_numeric = st.multiselect(
            "Wybierz atrybuty numeryczne",
            numeric_cols,
            default=numeric_cols[:2] if len(numeric_cols) > 2 else numeric_cols
        )
    
    with col2:
        selected_categorical = st.multiselect(
            "Wybierz atrybuty kategoryczne",
            categorical_cols,
            default=[]
        )
    
    selected_features = selected_numeric + selected_categorical
    
    # Wybór zmiennej docelowej (tylko kategoryczne)
    available_targets = [col for col in categorical_cols if col not in selected_features]
    if not available_targets:
        st.error("Brak dostępnych kolumn kategorycznych jako zmienne docelowe.")
        return
    
    target_col = st.selectbox("Zmienna do przewidywania", available_targets)
    
    # Konfiguracja modelu
    st.write("### 2. Konfiguracja modelu Random Forest")
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.number_input(
            "Liczba drzew",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        min_samples_split = st.number_input(
            "Minimalna liczba próbek do podziału",
            min_value=2,
            max_value=20,
            value=2
        )
    
    with col2:
        max_depth = st.number_input(
            "Maksymalna głębokość drzew",
            min_value=1,
            max_value=50,
            value=10
        )
        
        min_samples_leaf = st.number_input(
            "Minimalna liczba próbek w liściu",
            min_value=1,
            max_value=20,
            value=1
        )
    
    # Konfiguracja walidacji
    st.write("### 3. Konfiguracja walidacji")
    test_size = st.slider(
        "Rozmiar zbioru testowego",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05
    )
    
    cv_folds = st.number_input(
        "Liczba foldów walidacji krzyżowej",
        min_value=2,
        max_value=10,
        value=5
    )
    
    # Trenowanie i ewaluacja modelu
    if st.button("Trenuj model", use_container_width=True):
        with st.spinner("Trwa trenowanie modelu..."):
            try:
                # Sprawdzenie czy dane zawierają te same kolumny
                missing_features = [col for col in selected_features if col not in df.columns]
                if missing_features:
                    st.error(f"Brakujące kolumny w danych: {missing_features}")
                    return
        
                # Czyszczenie danych
                df_clean = df[selected_features + [target_col]].copy()

                # Informacja o oryginalnej liczbie wierszy
                original_rows = len(df_clean)
                
                # Usuwanie wierszy z brakującymi wartościami
                df_clean = df_clean.dropna()
                removed_rows = original_rows - len(df_clean)
                
                if removed_rows > 0:
                    st.warning(f"Usunięto {removed_rows} wierszy z brakującymi wartościami. Pozostało {len(df_clean)} wierszy.")
                    
                if len(df_clean) < 10:
                    st.error("Za mało danych do przeprowadzenia analizy (minimum 10 wierszy).")
                    return
                    
                # Sprawdzenie liczby unikalnych wartości w zmiennej docelowej
                unique_targets = df_clean[target_col].nunique()
                if unique_targets < 2:
                    st.error(f"Zmienna docelowa '{target_col}' ma tylko {unique_targets} unikalną wartość. Potrzebne są co najmniej 2 klasy.")
                    return

                # Sprawdź liczebność klas przed treningiem
                class_counts = df_clean[target_col].value_counts()
                min_class_size = class_counts.min()
                
                # Dostosuj liczbę foldów do najmniejszej klasy
                max_possible_folds = min(min_class_size, 10)
                if cv_folds > max_possible_folds:
                    cv_folds = max_possible_folds
                    st.warning(f"Zmniejszono liczbę foldów do {cv_folds} ze względu na małą liczebność niektórych klas.")
                
                if min_class_size < 2:
                    st.error(f"Niektóre klasy mają zbyt mało próbek (minimum 2 wymagane). Najmniejsza klasa ma {min_class_size} próbek.")
                    st.write("Liczebność klas:")
                    st.write(class_counts)
                    return

                # Wyświetl informację o liczebności klas
                st.info("Liczebność klas:")
                st.write(class_counts)

                # Inicjalizacja modelu i przygotowanie danych
                clf = ClassificationModel()
                
                X = df_clean[selected_features]
                y = df_clean[target_col]
                
                # Podział na zbiór treningowy i testowy
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Przygotowanie cech
                X_train_encoded, feature_names = clf.prepare_features(
                    X_train, selected_categorical, fit=True
                )
                X_test_encoded, _ = clf.prepare_features(
                    X_test, selected_categorical, fit=False
                )
                
                # Przygotowanie zmiennej docelowej
                y_train_encoded = clf.prepare_target(y_train)
                y_test_encoded = clf.prepare_target(y_test)
                
                # Konfiguracja i trenowanie modelu
                model_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'random_state': 42
                }
                
                clf.feature_names = feature_names
                clf.train(X_train_encoded, y_train_encoded, model_params)
                
                # Ewaluacja modelu
                results = clf.evaluate(
                    X_train_encoded, y_train_encoded,
                    X_test_encoded, y_test_encoded,
                    cv_folds
                )
                
                # Wyświetlanie wyników
                st.success("Model wytrenowany pomyślnie!")
                
                # 1. Podstawowe metryki
                st.write("#### Metryki modelu")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Średnia dokładność CV",
                        f"{results['cv_scores'].mean():.3f}",
                        f"±{results['cv_scores'].std():.3f}"
                    )
                with col2:
                    st.metric(
                        "Średnia dokładność LOO",
                        f"{results['loo_scores'].mean():.3f}"
                    )
                with col3:
                    st.metric(
                        "Dokładność na zbiorze testowym",
                        f"{results['test_score']:.3f}"
                    )
                
                # 2. Macierz pomyłek
                st.write("#### Macierz pomyłek")
                fig_confusion = plot_confusion_matrix(
                    results['confusion_matrix'],
                    clf.class_names
                )
                st.plotly_chart(fig_confusion, use_container_width=True)
                
                # 3. Ważność cech
                st.write("#### Ważność cech")
                fig_importance = px.bar(
                    results['feature_importance'],
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Ważność cech w modelu'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # 4. Szczegółowy raport
                st.write("#### Szczegółowy raport klasyfikacji")
                report_df = pd.DataFrame(results['classification_report']).T
                st.dataframe(report_df)
                
            except Exception as e:
                st.error(f"Wystąpił błąd podczas trenowania modelu: {str(e)}")
                st.write("Szczegóły błędu dla debugowania:")
                st.write(f"Wybrane cechy: {selected_features}")
                st.write(f"Zmienna docelowa: {target_col}")
                st.write(f"Liczba wierszy w danych: {len(df)}")
