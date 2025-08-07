from modules.prompts import prompt_generar_insights
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import streamlit as st
import pandas as pd

def generar_insights_ia(insights, model):
    prompt = prompt_generar_insights(insights)
    response = model.generate_content(prompt)
    texto = response.text.strip()
    return texto

# --- Función 4: Clustering con K-means ---
def aplicar_kmeans(df, columnas, k_range=(2, 10)):
    """Aplica K-means y encuentra el número óptimo de clusters"""
    # Preparar datos
    X = df[columnas].fillna(df[columnas].mean())
    
    # Encontrar número óptimo de clusters usando método del codo y silhouette
    inertias = []
    silhouette_scores = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(0)
    
    # Seleccionar el mejor k basado en silhouette score
    best_k = k_values[np.argmax(silhouette_scores)]
    
    # Aplicar K-means con el mejor k
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters_kmeans = final_kmeans.fit_predict(X)
    
    return clusters_kmeans, best_k, silhouette_scores, inertias, final_kmeans

# --- Función 5: Clustering con DBSCAN ---
def aplicar_dbscan(df, columnas, eps_range=(0.3, 2.0), min_samples_range=(3, 10)):
    """Aplica DBSCAN y encuentra los mejores parámetros"""
    X = df[columnas].fillna(df[columnas].mean())
    
    best_score = -1
    best_params = None
    best_labels = None
    
    # Búsqueda de mejores parámetros
    eps_values = np.linspace(eps_range[0], eps_range[1], 5)
    min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1, 2)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Evaluar solo si hay más de un cluster y no todo es ruido
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and n_clusters < len(X) * 0.8:
                try:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_labels = labels
                except:
                    continue
    
    if best_labels is None:
        # Si no se encuentra una buena configuración, usar parámetros por defecto
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        best_labels = dbscan.fit_predict(X)
        best_params = {'eps': 0.5, 'min_samples': 5}
        best_score = 0
    
    return best_labels, best_params, best_score

# --- Función 6: Generar insights automáticos ---
def generar_insights(df, clusters_kmeans, clusters_dbscan, columnas_numericas, nombre_archivo):
    """Genera insights automáticos para análisis con IA"""
    insights = []
    
    # Información básica del dataset
    insights.append(f"DATASET: {nombre_archivo}")
    insights.append(f"REGISTROS_TOTALES: {len(df)}")
    insights.append(f"VARIABLES_NUMERICAS: {len(columnas_numericas)}")
    insights.append(f"VARIABLES_ANALIZADAS: {', '.join(columnas_numericas)}")
    
    # Análisis de K-means
    n_clusters_kmeans = len(set(clusters_kmeans))
    insights.append(f"KMEANS_CLUSTERS: {n_clusters_kmeans}")
    
    for i in range(n_clusters_kmeans):
        cluster_data = df[clusters_kmeans == i]
        size_percent = (len(cluster_data) / len(df)) * 100
        insights.append(f"KMEANS_CLUSTER_{i}_TAMAÑO: {len(cluster_data)} registros ({size_percent:.1f}%)")
        
        # Estadísticas por cluster
        for col in columnas_numericas:
            mean_val = cluster_data[col].mean()
            std_val = cluster_data[col].std()
            insights.append(f"KMEANS_CLUSTER_{i}_{col}_PROMEDIO: {mean_val:.3f}")
            insights.append(f"KMEANS_CLUSTER_{i}_{col}_DESVIACION: {std_val:.3f}")
    
    # Análisis de DBSCAN
    n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
    ruido_dbscan = sum(clusters_dbscan == -1)
    ruido_percent = (ruido_dbscan / len(df)) * 100
    
    insights.append(f"DBSCAN_CLUSTERS: {n_clusters_dbscan}")
    insights.append(f"DBSCAN_RUIDO: {ruido_dbscan} registros ({ruido_percent:.1f}%)")
    
    for i in set(clusters_dbscan):
        if i != -1:  # Excluir ruido
            cluster_data = df[clusters_dbscan == i]
            size_percent = (len(cluster_data) / len(df)) * 100
            insights.append(f"DBSCAN_CLUSTER_{i}_TAMAÑO: {len(cluster_data)} registros ({size_percent:.1f}%)")
    
    # Análisis de correlaciones
    correlaciones = df[columnas_numericas].corr()
    correlaciones_altas = []
    
    for i in range(len(correlaciones.columns)):
        for j in range(i+1, len(correlaciones.columns)):
            corr_val = correlaciones.iloc[i, j]
            if abs(corr_val) > 0.7:  # Correlación fuerte
                var1 = correlaciones.columns[i]
                var2 = correlaciones.columns[j]
                correlaciones_altas.append(f"CORRELACION_{var1}_{var2}: {corr_val:.3f}")
    
    insights.extend(correlaciones_altas)
    
    # Análisis de outliers por variable
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_percent = (outliers / len(df)) * 100
        insights.append(f"OUTLIERS_{col}: {outliers} registros ({outlier_percent:.1f}%)")
    
    # Estadísticas generales
    for col in columnas_numericas:
        insights.append(f"ESTADISTICA_{col}_PROMEDIO: {df[col].mean():.3f}")
        insights.append(f"ESTADISTICA_{col}_MEDIANA: {df[col].median():.3f}")
        insights.append(f"ESTADISTICA_{col}_DESVIACION: {df[col].std():.3f}")
        insights.append(f"ESTADISTICA_{col}_MINIMO: {df[col].min():.3f}")
        insights.append(f"ESTADISTICA_{col}_MAXIMO: {df[col].max():.3f}")
    
    return insights

# --- Visualización de clustering ---
def visualizar_clustering(df, clusters_kmeans, clusters_dbscan, columnas, nombre):
    """Crea visualizaciones interactivas de los clusters"""
    
    # Reducir dimensionalidad para visualización
    if len(columnas) > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df[columnas].fillna(df[columnas].mean()))
        df_vis = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_vis['KMeans'] = clusters_kmeans
        df_vis['DBSCAN'] = clusters_dbscan
    else:
        df_vis = df[columnas].copy()
        df_vis['KMeans'] = clusters_kmeans
        df_vis['DBSCAN'] = clusters_dbscan
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K-means Clustering")
        if len(columnas) > 2:
            fig = px.scatter(df_vis, x='PCA1', y='PCA2', color='KMeans', 
                           title="K-means Clusters (PCA)", 
                           color_continuous_scale='viridis')
        else:
            fig = px.scatter(df_vis, x=columnas[0], y=columnas[1] if len(columnas) > 1 else columnas[0], 
                           color='KMeans', title="K-means Clusters")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("DBSCAN Clustering")
        if len(columnas) > 2:
            fig = px.scatter(df_vis, x='PCA1', y='PCA2', color='DBSCAN', 
                           title="DBSCAN Clusters (PCA)", 
                           color_continuous_scale='plasma')
        else:
            fig = px.scatter(df_vis, x=columnas[0], y=columnas[1] if len(columnas) > 1 else columnas[0], 
                           color='DBSCAN', title="DBSCAN Clusters")
        st.plotly_chart(fig, use_container_width=True)
