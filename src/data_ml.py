import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def prepare_data_for_ml(df, features, target=None, test_size=0.2, random_state=42):
    """
    Przygotowuje dane do modelowania.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    features : list
        Lista kolumn do wykorzystania jako cechy
    target : str, optional
        Nazwa kolumny docelowej (dla uczenia nadzorowanego)
    test_size : float, optional
        Proporcja podziału na zbiór testowy (0.0-1.0)
    random_state : int, optional
        Ziarno losowości dla powtarzalności wyników
        
    Returns:
    --------
    dict
        Słownik zawierający przygotowane dane
    """
    try:
        # Usuwamy wiersze z brakującymi wartościami
        df_cleaned = df[features + ([target] if target else [])].dropna()
        
        if df_cleaned.shape[0] == 0:
            st.error("Po usunięciu brakujących wartości nie pozostały żadne dane.")
            return None
            
        if target:
            # Uczenie nadzorowane
            X = df_cleaned[features]
            y = df_cleaned[target]
            
            # Podział na zbiór treningowy i testowy
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'features': features,
                'target': target
            }
        else:
            # Uczenie nienadzorowane
            X = df_cleaned[features]
            return {
                'X': X,
                'features': features
            }
    except Exception as e:
        st.error(f"Błąd podczas przygotowania danych: {e}")
        return None

def perform_clustering(data, n_clusters=3, random_state=42):
    """
    Wykonuje grupowanie metodą K-średnich.
    
    Parameters:
    -----------
    data : dict
        Słownik z danymi przygotowanymi przez prepare_data_for_ml
    n_clusters : int, optional
        Liczba klastrów
    random_state : int, optional
        Ziarno losowości dla powtarzalności wyników
        
    Returns:
    --------
    dict
        Słownik zawierający wyniki grupowania
    """
    try:
        if data is None:
            return None
            
        X = data['X']
        features = data['features']
        
        # Skalowanie danych
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Grupowanie
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Dodajemy klastry do oryginalnych danych
        X_with_clusters = X.copy()
        X_with_clusters['cluster'] = clusters
        
        # Redukcja wymiarowości do wizualizacji (jeśli więcej niż 2 cechy)
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            X_pca_df['cluster'] = clusters
            
            # Tworzymy wykres PCA
            fig_pca = px.scatter(
                X_pca_df, 
                x='PC1', 
                y='PC2', 
                color='cluster',
                title='Wizualizacja klastrów (PCA)',
                labels={'cluster': 'Klaster'}
            )
        else:
            # Tworzymy wykres bezpośrednio na oryginalnych cechach
            fig_pca = px.scatter(
                X_with_clusters, 
                x=features[0], 
                y=features[1], 
                color='cluster',
                title='Wizualizacja klastrów',
                labels={'cluster': 'Klaster'}
            )
        
        # Analiza klastrów - średnie wartości cech w każdym klastrze
        cluster_analysis = X_with_clusters.groupby('cluster').mean()
        
        return {
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_analysis': cluster_analysis,
            'X_with_clusters': X_with_clusters,
            'pca_plot': fig_pca,
            'features': features,
            'n_clusters': n_clusters
        }
    except Exception as e:
        st.error(f"Błąd podczas grupowania: {e}")
        return None

def train_regression_model(data, model_type='linear'):
    """
    Trenuje model regresji.
    
    Parameters:
    -----------
    data : dict
        Słownik z danymi przygotowanymi przez prepare_data_for_ml
    model_type : str, optional
        Typ modelu ('linear' lub 'random_forest')
        
    Returns:
    --------
    dict
        Słownik zawierający wyniki trenowania modelu
    """
    try:
        if data is None:
            return None
            
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Wybór modelu
        if model_type == 'linear':
            model = LinearRegression()
            model_name = "Regresja liniowa"
        else:  # random_forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_name = "Las losowy"
        
        # Trenowanie modelu
        model.fit(X_train, y_train)
        
        # Predykcja
        y_pred = model.predict(X_test)
        
        # Ocena modelu
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Wizualizacja predykcji vs rzeczywiste wartości
        results_df = pd.DataFrame({
            'Rzeczywiste': y_test,
            'Predykcja': y_pred
        }).reset_index(drop=True)
        
        fig = px.scatter(
            results_df, 
            x='Rzeczywiste', 
            y='Predykcja',
            title=f'Predykcja vs Rzeczywiste wartości - {model_name}',
            labels={
                'Rzeczywiste': 'Rzeczywiste wartości',
                'Predykcja': 'Przewidywane wartości'
            }
        )
        
        # Dodanie linii y=x (idealna predykcja)
        fig.add_shape(
            type='line',
            x0=results_df['Rzeczywiste'].min(),
            y0=results_df['Rzeczywiste'].min(),
            x1=results_df['Rzeczywiste'].max(),
            y1=results_df['Rzeczywiste'].max(),
            line=dict(color='red', dash='dash')
        )
        
        # Ważność cech (tylko dla modelu RandomForest)
        if model_type == 'random_forest':
            feature_importance = pd.DataFrame({
                'Feature': data['features'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance, 
                x='Feature', 
                y='Importance',
                title='Ważność cech',
                labels={
                    'Feature': 'Cecha',
                    'Importance': 'Ważność'
                }
            )
        else:
            fig_importance = None
            feature_importance = pd.DataFrame({
                'Feature': data['features'],
                'Coefficient': model.coef_
            })
        
        return {
            'model': model,
            'model_name': model_name,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            },
            'y_pred': y_pred,
            'plot': fig,
            'feature_importance': feature_importance,
            'importance_plot': fig_importance
        }
    except Exception as e:
        st.error(f"Błąd podczas trenowania modelu regresji: {e}")
        return None

def train_classification_model(data, model_type='logistic'):
    """
    Trenuje model klasyfikacji.
    
    Parameters:
    -----------
    data : dict
        Słownik z danymi przygotowanymi przez prepare_data_for_ml
    model_type : str, optional
        Typ modelu ('logistic' lub 'random_forest')
        
    Returns:
    --------
    dict
        Słownik zawierający wyniki trenowania modelu
    """
    try:
        if data is None:
            return None
            
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Sprawdzenie, czy zmienna docelowa jest kategoryczna (niezbędne dla klasyfikacji)
        if not pd.api.types.is_categorical_dtype(y_train) and not pd.api.types.is_object_dtype(y_train):
            if y_train.nunique() > 10:
                st.error("Wybrana kolumna docelowa ma zbyt wiele unikalnych wartości dla klasyfikacji.")
                return None
        
        # Wybór modelu
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "Regresja logistyczna"
        else:  # random_forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_name = "Las losowy"
        
        # Trenowanie modelu
        model.fit(X_train, y_train)
        
        # Predykcja
        y_pred = model.predict(X_test)
        
        # Ocena modelu
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Ważność cech (tylko dla modelu RandomForest)
        if model_type == 'random_forest':
            feature_importance = pd.DataFrame({
                'Feature': data['features'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance, 
                x='Feature', 
                y='Importance',
                title='Ważność cech',
                labels={
                    'Feature': 'Cecha',
                    'Importance': 'Ważność'
                }
            )
        else:
            fig_importance = None
            try:
                feature_importance = pd.DataFrame({
                    'Feature': data['features'],
                    'Coefficient': model.coef_[0]
                })
            except:
                feature_importance = None
        
        return {
            'model': model,
            'model_name': model_name,
            'metrics': {
                'accuracy': accuracy,
                'report': report
            },
            'y_pred': y_pred,
            'feature_importance': feature_importance,
            'importance_plot': fig_importance
        }
    except Exception as e:
        st.error(f"Błąd podczas trenowania modelu klasyfikacji: {e}")
        return None

def display_ml_ui(df):
    """
    Wyświetla interfejs użytkownika do analizy machine learning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame z danymi
    """
    st.subheader("Analiza Machine Learning 🤖")
    
    if df is None or df.empty:
        st.warning("Brak danych do analizy ML. Najpierw wczytaj plik CSV.")
        return
    
    # Pobieranie typów kolumn
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    if not numeric_cols:
        st.warning("Brak kolumn numerycznych do analizy ML.")
        return
    
    # Wybór typu analizy ML
    ml_type = st.radio(
        "Wybierz typ analizy", 
        ["Grupowanie (K-means)", "Regresja", "Klasyfikacja"]
    )
    
    if ml_type == "Grupowanie (K-means)":
        st.subheader("Grupowanie K-means")
        
        # Wybór cech
        selected_features = st.multiselect(
            "Wybierz cechy do grupowania", 
            numeric_cols, 
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        # Liczba klastrów
        n_clusters = st.slider("Liczba klastrów", min_value=2, max_value=10, value=3)
        
        if len(selected_features) < 1:
            st.warning("Wybierz co najmniej jedną cechę.")
        elif st.button("Wykonaj grupowanie"):
            with st.spinner("Grupowanie danych..."):
                # Przygotowanie danych
                data = prepare_data_for_ml(df, features=selected_features)
                
                # Wykonanie grupowania
                clustering_results = perform_clustering(data, n_clusters=n_clusters)
                
                if clustering_results:
                    # Wyświetlenie wyników
                    st.success(f"Grupowanie zakończone - utworzono {n_clusters} klastrów")
                    
                    # Wykres klastrów
                    st.plotly_chart(clustering_results['pca_plot'], use_container_width=True)
                    
                    # Analiza klastrów
                    st.subheader("Charakterystyka klastrów")
                    st.dataframe(clustering_results['cluster_analysis'])
                    
                    # Dodanie klastrów do oryginalnych danych
                    st.subheader("Dane z przypisanymi klastrami")
                    st.dataframe(clustering_results['X_with_clusters'])
    
    elif ml_type == "Regresja":
        st.subheader("Regresja")
        
        # Wybór cech
        selected_features = st.multiselect(
            "Wybierz cechy (zmienne niezależne)", 
            numeric_cols, 
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        # Wybór zmiennej docelowej
        target_col = st.selectbox(
            "Wybierz zmienną docelową (do przewidywania)", 
            [col for col in numeric_cols if col not in selected_features]
        ) if len(numeric_cols) > len(selected_features) else None
        
        # Wybór modelu
        model_type = st.radio("Wybierz typ modelu", ["Regresja liniowa", "Las losowy"])
        
        if not target_col:
            st.warning("Brak dostępnych zmiennych docelowych. Wybierz mniej cech.")
        elif len(selected_features) < 1:
            st.warning("Wybierz co najmniej jedną cechę.")
        elif st.button("Trenuj model"):
            with st.spinner("Trenowanie modelu regresji..."):
                # Przygotowanie danych
                data = prepare_data_for_ml(
                    df, 
                    features=selected_features, 
                    target=target_col
                )
                
                # Trenowanie modelu
                model_results = train_regression_model(
                    data, 
                    model_type='linear' if model_type == "Regresja liniowa" else 'random_forest'
                )
                
                if model_results:
                    # Wyświetlenie wyników
                    st.success(f"Model {model_results['model_name']} wytrenowany pomyślnie")
                    
                    # Metryki modelu
                    st.subheader("Metryki modelu")
                    metrics = model_results['metrics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", round(metrics['rmse'], 4))
                    with col2:
                        st.metric("MSE", round(metrics['mse'], 4))
                    with col3:
                        st.metric("R²", round(metrics['r2'], 4))
                    
                    # Wykres predykcji vs rzeczywiste wartości
                    st.subheader("Porównanie predykcji z rzeczywistymi wartościami")
                    st.plotly_chart(model_results['plot'], use_container_width=True)
                    
                    # Ważność cech
                    st.subheader("Ważność cech")
                    st.dataframe(model_results['feature_importance'])
                    
                    if model_results['importance_plot']:
                        st.plotly_chart(model_results['importance_plot'], use_container_width=True)
    
    elif ml_type == "Klasyfikacja":
        st.subheader("Klasyfikacja")
        
        # Wybór cech
        selected_features = st.multiselect(
            "Wybierz cechy (zmienne niezależne)", 
            numeric_cols, 
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        # Wybór zmiennej docelowej (preferujemy kategoryczne, ale można też numeryczne z małą liczbą unikalnych wartości)
        potential_targets = categorical_cols + [
            col for col in numeric_cols 
            if col not in selected_features and df[col].nunique() <= 10
        ]
        
        target_col = st.selectbox(
            "Wybierz zmienną docelową (do klasyfikacji)", 
            potential_targets
        ) if potential_targets else None
        
        # Wybór modelu
        model_type = st.radio("Wybierz typ modelu", ["Regresja logistyczna", "Las losowy"])
        
        if not target_col:
            st.warning("Brak odpowiednich zmiennych docelowych. Wybierz kategoryczną zmienną lub numeryczną z małą liczbą unikalnych wartości.")
        elif len(selected_features) < 1:
            st.warning("Wybierz co najmniej jedną cechę.")
        elif st.button("Trenuj model"):
            with st.spinner("Trenowanie modelu klasyfikacji..."):
                # Przygotowanie danych
                data = prepare_data_for_ml(
                    df, 
                    features=selected_features, 
                    target=target_col
                )
                
                # Trenowanie modelu
                model_results = train_classification_model(
                    data, 
                    model_type='logistic' if model_type == "Regresja logistyczna" else 'random_forest'
                )
                
                if model_results:
                    # Wyświetlenie wyników
                    st.success(f"Model {model_results['model_name']} wytrenowany pomyślnie")
                    
                    # Metryki modelu
                    st.subheader("Metryki modelu")
                    metrics = model_results['metrics']
                    st.metric("Dokładność (Accuracy)", f"{round(metrics['accuracy'] * 100, 2)}%")
                    
                    # Raport klasyfikacji
                    st.subheader("Raport klasyfikacji")
                    report_df = pd.DataFrame(metrics['report']).T
                    st.dataframe(report_df)
                    
                    # Ważność cech
                    if model_results['feature_importance'] is not None:
                        st.subheader("Ważność cech")
                        st.dataframe(model_results['feature_importance'])
                        
                        if model_results['importance_plot']:
                            st.plotly_chart(model_results['importance_plot'], use_container_width=True)