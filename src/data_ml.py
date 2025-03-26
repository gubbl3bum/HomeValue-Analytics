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
    Prepares data for modeling.

    Parameters:
    -----------
    df : pandas.DataFrame
    DataFrame with data
    features : list
    List of columns to use as features
    target : str, optional
    Name of target column (for supervised learning)
    test_size : float, optional
    Test set split ratio (0.0-1.0)
    random_state : int, optional
    Randomness seed for repeatability of results

    Returns:
    --------
    dict
    Dictionary containing prepared data
    """
    try:
        # Deleting rows with missing values
        df_cleaned = df[features + ([target] if target else [])].dropna()
        
        if df_cleaned.shape[0] == 0:
            st.error("Po usunięciu brakujących wartości nie pozostały żadne dane.")
            return None
            
        if target:
            # Supervised learning
            X = df_cleaned[features]
            y = df_cleaned[target]
            
            # Division into training and test sets
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
            # Unsupervised learning
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
    Performs K-means clustering.

    Parameters:
    -----------
    data : dict
    Dictionary with data prepared by prepare_data_for_ml
    n_clusters : int, optional
    Number of clusters
    random_state : int, optional
    Randomness seed for repeatability of results

    Returns:
    --------
    dict
    Dictionary containing clustering results
    """
    try:
        if data is None:
            return None
            
        X = data['X']
        features = data['features']
        
        # Data scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Grouping
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Adding clusters to the original data
        X_with_clusters = X.copy()
        X_with_clusters['cluster'] = clusters
        
        # Dimensionality reduction for visualization (if more than 2 features)
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            X_pca_df['cluster'] = clusters
            
            # Create a PCA chart
            fig_pca = px.scatter(
                X_pca_df, 
                x='PC1', 
                y='PC2', 
                color='cluster',
                title='Wizualizacja klastrów (PCA)',
                labels={'cluster': 'Klaster'}
            )
        else:
            # Creating a graph directly on the original features
            fig_pca = px.scatter(
                X_with_clusters, 
                x=features[0], 
                y=features[1], 
                color='cluster',
                title='Wizualizacja klastrów',
                labels={'cluster': 'Klaster'}
            )
        
        # Cluster analysis - average feature values ​​in each cluster
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
    Trains a regression model.

    Parameters:
    -----------
    data : dict
    A dictionary containing data prepared by prepare_data_for_ml
    model_type : str, optional
    The model type ('linear' or 'random_forest')

    Returns:
    --------
    dict
    A dictionary containing the results of training the model
    """
    try:
        if data is None:
            return None
            
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Model selection
        if model_type == 'linear':
            model = LinearRegression()
            model_name = "Regresja liniowa"
        else:  # random_forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_name = "Las losowy"
        
        # Model training
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test)
        
        # Model evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Visualization of predictions vs. actual values
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
        
        # Adding the line y=x (perfect prediction)
        fig.add_shape(
            type='line',
            x0=results_df['Rzeczywiste'].min(),
            y0=results_df['Rzeczywiste'].min(),
            x1=results_df['Rzeczywiste'].max(),
            y1=results_df['Rzeczywiste'].max(),
            line=dict(color='red', dash='dash')
        )
        
        # Feature Importance (RandomForest model only)
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
    Trains a classification model.

    Parameters:
    -----------
    data : dict
    A dictionary containing data prepared by prepare_data_for_ml
    model_type : str, optional
    The model type ('logistic' or 'random_forest')

    Returns:
    --------
    dict
    A dictionary containing the results of training the model
    """
    try:
        if data is None:
            return None
            
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Checking if the target variable is categorical (necessary for classification)
        if not pd.api.types.is_categorical_dtype(y_train) and not pd.api.types.is_object_dtype(y_train):
            if y_train.nunique() > 10:
                st.error("Wybrana kolumna docelowa ma zbyt wiele unikalnych wartości dla klasyfikacji.")
                return None
        
        # Model selection
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "Regresja logistyczna"
        else:  # random_forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_name = "Las losowy"
        
        # Model training
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test)
        
        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature Importance (RandomForest model only)
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
    Displays a user interface for machine learning analysis.

    Parameters:
    -----------
    df : pandas.DataFrame
    DataFrame with data
    """
    st.subheader("Analiza Machine Learning 🤖")
    
    if df is None or df.empty:
        st.warning("Brak danych do analizy ML. Najpierw wczytaj plik CSV.")
        return
    
    # Getting column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    if not numeric_cols:
        st.warning("Brak kolumn numerycznych do analizy ML.")
        return
    
    # Selecting the type of ML analysis
    ml_type = st.radio(
        "Wybierz typ analizy", 
        ["Grupowanie (K-means)", "Regresja", "Klasyfikacja"]
    )
    
    if ml_type == "Grupowanie (K-means)":
        st.subheader("Grupowanie K-means")
        
        # Feature selection
        selected_features = st.multiselect(
            "Wybierz cechy do grupowania", 
            numeric_cols, 
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        # Number of clusters
        n_clusters = st.slider("Liczba klastrów", min_value=2, max_value=10, value=3)
        
        if len(selected_features) < 1:
            st.warning("Wybierz co najmniej jedną cechę.")
        elif st.button("Wykonaj grupowanie"):
            with st.spinner("Grupowanie danych..."):
                # Data preparation
                data = prepare_data_for_ml(df, features=selected_features)
                
                # Perform grouping
                clustering_results = perform_clustering(data, n_clusters=n_clusters)
                
                if clustering_results:
                    # Display the results
                    st.success(f"Grupowanie zakończone - utworzono {n_clusters} klastrów")
                    
                    # Cluster chart
                    st.plotly_chart(clustering_results['pca_plot'], use_container_width=True)
                    
                    # Cluster analysis
                    st.subheader("Charakterystyka klastrów")
                    st.dataframe(clustering_results['cluster_analysis'])
                    
                    # Adding clusters to the original data
                    st.subheader("Dane z przypisanymi klastrami")
                    st.dataframe(clustering_results['X_with_clusters'])
    
    elif ml_type == "Regresja":
        st.subheader("Regresja")
        
        # Feature selection
        selected_features = st.multiselect(
            "Wybierz cechy (zmienne niezależne)", 
            numeric_cols, 
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        # Selecting the target variable
        target_col = st.selectbox(
            "Wybierz zmienną docelową (do przewidywania)", 
            [col for col in numeric_cols if col not in selected_features]
        ) if len(numeric_cols) > len(selected_features) else None
        
        # Model selection
        model_type = st.radio("Wybierz typ modelu", ["Regresja liniowa", "Las losowy"])
        
        if not target_col:
            st.warning("Brak dostępnych zmiennych docelowych. Wybierz mniej cech.")
        elif len(selected_features) < 1:
            st.warning("Wybierz co najmniej jedną cechę.")
        elif st.button("Trenuj model"):
            with st.spinner("Trenowanie modelu regresji..."):
                # Data preparation
                data = prepare_data_for_ml(
                    df, 
                    features=selected_features, 
                    target=target_col
                )
                
                # Model training
                model_results = train_regression_model(
                    data, 
                    model_type='linear' if model_type == "Regresja liniowa" else 'random_forest'
                )
                
                if model_results:
                    # Display the results
                    st.success(f"Model {model_results['model_name']} wytrenowany pomyślnie")
                    
                    # Model metrics
                    st.subheader("Metryki modelu")
                    metrics = model_results['metrics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", round(metrics['rmse'], 4))
                    with col2:
                        st.metric("MSE", round(metrics['mse'], 4))
                    with col3:
                        st.metric("R²", round(metrics['r2'], 4))
                    
                    # Prediction Graph vs. Actual Values
                    st.subheader("Porównanie predykcji z rzeczywistymi wartościami")
                    st.plotly_chart(model_results['plot'], use_container_width=True)
                    
                    # Importance of features
                    st.subheader("Ważność cech")
                    st.dataframe(model_results['feature_importance'])
                    
                    if model_results['importance_plot']:
                        st.plotly_chart(model_results['importance_plot'], use_container_width=True)
    
    elif ml_type == "Klasyfikacja":
        st.subheader("Klasyfikacja")
        
        # Feature selection
        selected_features = st.multiselect(
            "Wybierz cechy (zmienne niezależne)", 
            numeric_cols, 
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        # Selecting a target variable (we prefer categorical, but you can also use numeric with a small number of unique values)
        potential_targets = categorical_cols + [
            col for col in numeric_cols 
            if col not in selected_features and df[col].nunique() <= 10
        ]
        
        target_col = st.selectbox(
            "Wybierz zmienną docelową (do klasyfikacji)", 
            potential_targets
        ) if potential_targets else None
        
        # Model selection
        model_type = st.radio("Wybierz typ modelu", ["Regresja logistyczna", "Las losowy"])
        
        if not target_col:
            st.warning("Brak odpowiednich zmiennych docelowych. Wybierz kategoryczną zmienną lub numeryczną z małą liczbą unikalnych wartości.")
        elif len(selected_features) < 1:
            st.warning("Wybierz co najmniej jedną cechę.")
        elif st.button("Trenuj model"):
            with st.spinner("Trenowanie modelu klasyfikacji..."):
                # Data preparation
                data = prepare_data_for_ml(
                    df, 
                    features=selected_features, 
                    target=target_col
                )
                
                # Model training
                model_results = train_classification_model(
                    data, 
                    model_type='logistic' if model_type == "Regresja logistyczna" else 'random_forest'
                )
                
                if model_results:
                    # Display the results
                    st.success(f"Model {model_results['model_name']} wytrenowany pomyślnie")
                    
                    # Model metrics
                    st.subheader("Metryki modelu")
                    metrics = model_results['metrics']
                    st.metric("Dokładność (Accuracy)", f"{round(metrics['accuracy'] * 100, 2)}%")
                    
                    # Classification report
                    st.subheader("Raport klasyfikacji")
                    report_df = pd.DataFrame(metrics['report']).T
                    st.dataframe(report_df)
                    
                    # Importance of features
                    if model_results['feature_importance'] is not None:
                        st.subheader("Ważność cech")
                        st.dataframe(model_results['feature_importance'])
                        
                        if model_results['importance_plot']:
                            st.plotly_chart(model_results['importance_plot'], use_container_width=True)