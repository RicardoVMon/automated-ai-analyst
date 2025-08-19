from modules.prompts import prompt_generar_insights
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import streamlit as st
import pandas as pd

def generar_insights_ia(insights, model):
    """
    Genera insights de inteligencia artificial a partir de datos de clustering.
    
    Parámetros:
    -----------
    insights : list
        Lista de insights estructurados obtenidos del análisis de clustering
    model : object
        Modelo de IA (como Gemini) para generar contenido de texto
    
    Retorna:
    --------
    str
        Texto con insights generados por IA en lenguaje natural
    
    Proceso:
    --------
    1. Convierte los insights en un prompt estructurado
    2. Solicita al modelo de IA que genere análisis en lenguaje natural
    3. Retorna el texto limpio generado por la IA
    """
    prompt = prompt_generar_insights(insights)  # Convierte insights a prompt
    response = model.generate_content(prompt)   # Genera respuesta con IA
    texto = response.text.strip()               # Limpia espacios en blanco
    return texto

# --- Función 4: Clustering con K-means ---
def aplicar_kmeans(df, columnas, k_range=(2, 10)):
    """
    Aplica algoritmo K-means y encuentra el número óptimo de clusters.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos a analizar
    columnas : list
        Lista de nombres de columnas numéricas para clustering
    k_range : tuple, opcional
        Rango de valores k a evaluar (min, max). Por defecto (2, 10)
    
    Retorna:
    --------
    tuple
        - clusters_kmeans: array con etiquetas de cluster para cada registro
        - best_k: número óptimo de clusters encontrado
        - silhouette_scores: lista de scores de silhouette para cada k
        - inertias: lista de inercias (suma de distancias al centroide) para cada k
        - final_kmeans: objeto KMeans entrenado con el mejor k
    
    Conceptos Técnicos:
    ------------------
    - K-means: Algoritmo de clustering que agrupa datos en k clusters minimizando 
      la distancia de cada punto al centroide de su cluster
    - Método del codo: Técnica para encontrar k óptimo observando el punto donde 
      la inercia deja de decrecer significativamente
    - Silhouette Score: Métrica que mide qué tan bien separados están los clusters.
      Valores cercanos a 1 indican clusters bien separados, cercanos a 0 indican
      clusters superpuestos, negativos indican asignación incorrecta
    - Inercia: Suma de distancias cuadradas de cada punto a su centroide. 
      Menor inercia = clusters más compactos
    """
    # Preparar datos
    X = df[columnas]
    
    # Inicializar listas para almacenar métricas de evaluación
    inertias = []           # Almacena inercia para cada k
    silhouette_scores = []  # Almacena silhouette score para cada k
    k_values = range(k_range[0], k_range[1] + 1)  # Genera rango de k a evaluar
    
    # Evaluar cada valor de k en el rango especificado
    for k in k_values:
        # Crear y entrenar modelo K-means
        # n_init=10: ejecuta el algoritmo 10 veces con diferentes inicializaciones
        # random_state=42: semilla para reproducibilidad
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)  # Entrenar y predecir clusters
        
        inertias.append(kmeans.inertia_)  # Guardar inercia del modelo
        
        # Calcular silhouette score (requiere al menos 2 clusters)
        if k > 1:
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(0)  # Score 0 para k=1
    
    # Seleccionar el mejor k basado en el mayor silhouette score
    best_k = k_values[np.argmax(silhouette_scores)]
    
    # Aplicar K-means final con el número óptimo de clusters
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters_kmeans = final_kmeans.fit_predict(X)  # Obtener etiquetas finales
    
    return clusters_kmeans, best_k, silhouette_scores, inertias, final_kmeans

# --- Función 5: Clustering con DBSCAN ---
def aplicar_dbscan(df, columnas, eps_range=(0.3, 2.0), min_samples_range=(3, 10)):
    """
    Aplica algoritmo DBSCAN y encuentra los mejores parámetros mediante búsqueda grid.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos a analizar
    columnas : list
        Lista de nombres de columnas numéricas para clustering
    eps_range : tuple, opcional
        Rango de valores epsilon a evaluar. Por defecto (0.3, 2.0)
    min_samples_range : tuple, opcional
        Rango de valores min_samples a evaluar. Por defecto (3, 10)
    
    Retorna:
    --------
    tuple
        - best_labels: array con etiquetas de cluster óptimas (-1 = ruido)
        - best_params: diccionario con mejores parámetros {'eps': valor, 'min_samples': valor}
        - best_score: mejor silhouette score encontrado
    
    Conceptos Técnicos:
    ------------------
    - DBSCAN: Density-Based Spatial Clustering. Agrupa puntos que están densamente 
      empaquetados y marca como outliers los puntos en regiones de baja densidad
    - Epsilon (eps): Radio máximo de vecindario. Puntos dentro de esta distancia 
      se consideran vecinos
    - Min_samples: Número mínimo de puntos en el vecindario de un punto para que 
      este sea considerado un punto central (core point)
    - Ruido: Puntos marcados como -1, que no pertenecen a ningún cluster por 
      estar en regiones de baja densidad
    - Grid Search: Búsqueda exhaustiva de la mejor combinación de parámetros
    """
    # Preparar datos
    X = df[columnas]
    
    # Inicializar variables para almacenar los mejores resultados
    best_score = -1      # Mejor silhouette score encontrado
    best_params = None   # Mejores parámetros encontrados
    best_labels = None   # Mejores etiquetas de cluster encontradas
    
    # Crear grids de parámetros a evaluar
    eps_values = np.linspace(eps_range[0], eps_range[1], 5)  # 5 valores entre min y max
    min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1, 2)  # cada 2 valores
    
    # Búsqueda exhaustiva de mejores parámetros (Grid Search)
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Crear y entrenar modelo DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)  # Obtener etiquetas de cluster
            
            # Calcular número de clusters (excluyendo ruido marcado como -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Evaluar solo configuraciones válidas:
            # - Más de 1 cluster
            # - No más del 80% de puntos en clusters (evita sobre-clustering)
            if n_clusters > 1 and n_clusters < len(X) * 0.8:
                try:
                    # Calcular silhouette score para esta configuración
                    score = silhouette_score(X, labels)
                    
                    # Actualizar mejores resultados si este score es mejor
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_labels = labels
                except:
                    # Continuar si hay error en el cálculo (ej: todos los puntos son ruido)
                    continue
    
    # Si no se encuentra una configuración válida, usar parámetros por defecto
    if best_labels is None:
        dbscan = DBSCAN(eps=0.5, min_samples=5)  # Parámetros conservadores
        best_labels = dbscan.fit_predict(X)
        best_params = {'eps': 0.5, 'min_samples': 5}
        best_score = 0
    
    return best_labels, best_params, best_score

# --- Función 6: Generar insights automáticos ---
def generar_insights(df, clusters_kmeans, clusters_dbscan, columnas_numericas, nombre_archivo):
    """
    Genera insights automáticos detallados para análisis posterior con IA.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame original con todos los datos
    clusters_kmeans : array
        Etiquetas de cluster obtenidas con K-means
    clusters_dbscan : array
        Etiquetas de cluster obtenidas con DBSCAN
    columnas_numericas : list
        Lista de nombres de columnas numéricas analizadas
    nombre_archivo : str
        Nombre del archivo analizado
    
    Retorna:
    --------
    list
        Lista de strings con insights estructurados para procesamiento por IA
    
    Conceptos Técnicos:
    ------------------
    - Outliers (IQR): Valores atípicos detectados usando el rango intercuartílico.
      Se consideran outliers los valores fuera del rango [Q1-1.5*IQR, Q3+1.5*IQR]
    - Correlación de Pearson: Mide la relación lineal entre variables (-1 a 1).
      |r| > 0.7 se considera correlación fuerte
    - Estadísticas descriptivas: Media, mediana, desviación estándar, min, max
      proporcionan un resumen completo de la distribución de cada variable
    """
    insights = []  # Lista para almacenar todos los insights
    
    # === INFORMACIÓN BÁSICA DEL DATASET ===
    insights.append(f"DATASET: {nombre_archivo}")
    insights.append(f"REGISTROS_TOTALES: {len(df)}")
    insights.append(f"VARIABLES_NUMERICAS: {len(columnas_numericas)}")
    insights.append(f"VARIABLES_ANALIZADAS: {', '.join(columnas_numericas)}")
    
    # === ANÁLISIS DETALLADO DE K-MEANS ===
    n_clusters_kmeans = len(set(clusters_kmeans))  # Contar clusters únicos
    insights.append(f"KMEANS_CLUSTERS: {n_clusters_kmeans}")
    
    # Analizar cada cluster de K-means individualmente
    for i in range(n_clusters_kmeans):
        # Filtrar datos del cluster actual
        cluster_data = df[clusters_kmeans == i]
        size_percent = (len(cluster_data) / len(df)) * 100  # Porcentaje del total
        insights.append(f"KMEANS_CLUSTER_{i}_TAMAÑO: {len(cluster_data)} registros ({size_percent:.1f}%)")
        
        # Calcular estadísticas por variable para este cluster
        for col in columnas_numericas:
            mean_val = cluster_data[col].mean()    # Media del cluster
            std_val = cluster_data[col].std()      # Desviación estándar del cluster
            insights.append(f"KMEANS_CLUSTER_{i}_{col}_PROMEDIO: {mean_val:.3f}")
            insights.append(f"KMEANS_CLUSTER_{i}_{col}_DESVIACION: {std_val:.3f}")
    
    # === ANÁLISIS DETALLADO DE DBSCAN ===
    # Contar clusters (excluyendo ruido marcado como -1)
    n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
    ruido_dbscan = sum(clusters_dbscan == -1)      # Contar puntos de ruido
    ruido_percent = (ruido_dbscan / len(df)) * 100  # Porcentaje de ruido
    
    insights.append(f"DBSCAN_CLUSTERS: {n_clusters_dbscan}")
    insights.append(f"DBSCAN_RUIDO: {ruido_dbscan} registros ({ruido_percent:.1f}%)")
    
    # Analizar cada cluster de DBSCAN (excluyendo ruido)
    for i in set(clusters_dbscan):
        if i != -1:  # Excluir puntos de ruido
            cluster_data = df[clusters_dbscan == i]
            size_percent = (len(cluster_data) / len(df)) * 100
            insights.append(f"DBSCAN_CLUSTER_{i}_TAMAÑO: {len(cluster_data)} registros ({size_percent:.1f}%)")
    
    # === ANÁLISIS DE CORRELACIONES ===
    correlaciones = df[columnas_numericas].corr()  # Matriz de correlación de Pearson
    print(correlaciones)  # Debug: imprimir matriz de correlaciones
    correlaciones_altas = []  # Lista para correlaciones significativas
    
    # Buscar correlaciones fuertes (|r| > 0.7) en la matriz triangular superior
    for i in range(len(correlaciones.columns)):
        for j in range(i+1, len(correlaciones.columns)):
            corr_val = correlaciones.iloc[i, j]
            if abs(corr_val) > 0.7:  # Correlación fuerte (positiva o negativa)
                var1 = correlaciones.columns[i]
                var2 = correlaciones.columns[j]
                correlaciones_altas.append(f"CORRELACION_{var1}_{var2}: {corr_val:.3f}")
    
    insights.extend(correlaciones_altas)  # Agregar correlaciones encontradas
    
    # === ANÁLISIS DE OUTLIERS (MÉTODO IQR) ===
    for col in columnas_numericas:
        # Calcular cuartiles para detección de outliers
        Q1 = df[col].quantile(0.25)    # Primer cuartil (25%)
        Q3 = df[col].quantile(0.75)    # Tercer cuartil (75%)
        IQR = Q3 - Q1                  # Rango intercuartílico
        
        # Detectar outliers usando regla 1.5*IQR
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_percent = (outliers / len(df)) * 100
        insights.append(f"OUTLIERS_{col}: {outliers} registros ({outlier_percent:.1f}%)")
    
    # === ESTADÍSTICAS DESCRIPTIVAS GENERALES ===
    for col in columnas_numericas:
        # Calcular estadísticas descriptivas completas para cada variable
        insights.append(f"ESTADISTICA_{col}_PROMEDIO: {df[col].mean():.3f}")      # Media aritmética
        insights.append(f"ESTADISTICA_{col}_MEDIANA: {df[col].median():.3f}")     # Valor central
        insights.append(f"ESTADISTICA_{col}_DESVIACION: {df[col].std():.3f}")     # Dispersión
        insights.append(f"ESTADISTICA_{col}_MINIMO: {df[col].min():.3f}")         # Valor mínimo
        insights.append(f"ESTADISTICA_{col}_MAXIMO: {df[col].max():.3f}")         # Valor máximo
    
    return insights

# --- Visualización de clustering ---
def visualizar_clustering(df, clusters_kmeans, clusters_dbscan, columnas, nombre):
    """
    Crea visualizaciones interactivas de los resultados de clustering.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos originales
    clusters_kmeans : array
        Etiquetas de cluster de K-means
    clusters_dbscan : array
        Etiquetas de cluster de DBSCAN
    columnas : list
        Lista de columnas utilizadas para clustering
    nombre : str
        Nombre del dataset para títulos
    
    Efectos:
    --------
    Muestra gráficos interactivos en Streamlit usando Plotly
    
    Conceptos Técnicos:
    ------------------
    - PCA (Principal Component Analysis): Técnica de reducción de dimensionalidad
      que proyecta datos multidimensionales en 2D preservando la máxima varianza
    - Reducción de dimensionalidad: Necesaria para visualizar datos con más de 
      2-3 dimensiones en gráficos 2D comprensibles
    - Visualización interactiva: Permite zoom, hover, y exploración de clusters
    """
    
    # === PREPARACIÓN DE DATOS PARA VISUALIZACIÓN ===
    if len(columnas) > 2:
        # Reducir dimensionalidad usando PCA para visualización en 2D
        pca = PCA(n_components=2)  # Reducir a 2 componentes principales
        # Aplicar PCA a datos sin valores faltantes
        X_pca = pca.fit_transform(df[columnas].fillna(df[columnas].mean()))
        
        # Crear DataFrame con componentes principales
        df_vis = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_vis['KMeans'] = clusters_kmeans   # Agregar etiquetas de K-means
        df_vis['DBSCAN'] = clusters_dbscan   # Agregar etiquetas de DBSCAN
    else:
        # Si hay 2 o menos dimensiones, usar datos originales
        df_vis = df[columnas].copy()
        df_vis['KMeans'] = clusters_kmeans
        df_vis['DBSCAN'] = clusters_dbscan
    
    # === CREAR VISUALIZACIONES LADO A LADO ===
    col1, col2 = st.columns(2)  # Dividir pantalla en 2 columnas
    
    # Visualización de K-means
    with col1:
        st.subheader("K-means Clustering")
        if len(columnas) > 2:
            # Gráfico con componentes principales
            fig = px.scatter(df_vis, x='PCA1', y='PCA2', color='KMeans', 
                           title="K-means Clusters (PCA)", 
                           color_continuous_scale='viridis')
        else:
            # Gráfico con variables originales
            fig = px.scatter(df_vis, x=columnas[0], 
                           y=columnas[1] if len(columnas) > 1 else columnas[0], 
                           color='KMeans', title="K-means Clusters")
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualización de DBSCAN
    with col2:
        st.subheader("DBSCAN Clustering")
        if len(columnas) > 2:
            # Gráfico con componentes principales
            fig = px.scatter(df_vis, x='PCA1', y='PCA2', color='DBSCAN', 
                           title="DBSCAN Clusters (PCA)", 
                           color_continuous_scale='plasma')
        else:
            # Gráfico con variables originales
            fig = px.scatter(df_vis, x=columnas[0], 
                           y=columnas[1] if len(columnas) > 1 else columnas[0], 
                           color='DBSCAN', title="DBSCAN Clusters")
        st.plotly_chart(fig, use_container_width=True)


def agregar_o_actualizar_insight_final(nombre, vars_clustering, insights_ia):
    """
    Gestiona el almacenamiento de insights finales en el estado de sesión de Streamlit.
    
    Parámetros:
    -----------
    nombre : str
        Nombre del archivo analizado
    vars_clustering : list
        Lista de variables utilizadas para clustering
    insights_ia : str
        Insights generados por IA
    
    Efectos:
    --------
    Actualiza o agrega entrada en st.session_state['insights_finales']
    
    Proceso:
    --------
    1. Inicializa lista de insights si no existe
    2. Busca si ya existe un análisis para el mismo archivo y variables
    3. Actualiza si existe o agrega nuevo registro si no existe
    """
    # Inicializar lista de insights finales si no existe
    if 'insights_finales' not in st.session_state:
        st.session_state['insights_finales'] = []
    
    # Convertir variables a tupla ordenada para comparación consistente
    vars_tuple = tuple(sorted(vars_clustering))
    
    # Buscar si ya existe un análisis con el mismo archivo y variables
    for insight in st.session_state['insights_finales']:
        if insight['archivo'] == nombre and insight['variables'] == vars_tuple:
            # Actualizar insight existente
            insight['insight'] = insights_ia
            return
    
    # Si no existe, agregar nuevo registro
    st.session_state['insights_finales'].append({
        "archivo": nombre,
        "variables": vars_tuple,
        "insight": insights_ia
    })