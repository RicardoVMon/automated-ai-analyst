import streamlit as st
import pandas as pd
import numpy as np
from modules.ml import generar_insights, generar_insights_ia, aplicar_kmeans, aplicar_dbscan, visualizar_clustering

# --- Punto de entrada ---
if st.session_state.get('pasar_a_analisis') == True:
    st.title("Análisis de Aprendizaje No Supervisado")
    st.markdown("Detecta valores atípicos y agrupa datos automáticamente con K-means y DBSCAN")

    for nombre, datos in st.session_state['archivos'].items():
        
        df_final = datos['dataframe_final']
        columnas_finales = df_final.select_dtypes(include=[np.number]).columns.tolist()
        clasificacion = datos.get('clasificacion', {})

        st.markdown(f"### Archivo: `{nombre}`")

        # Identificar columnas categóricas y numéricas
        cat_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Categórica" and col.strip() in df_final.columns]
        num_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df_final.columns]

        if not num_attribs:
            st.warning("No se encontraron columnas numéricas para el análisis.")
            continue

        # CLUSTERING
        st.markdown("#### Análisis de Clustering")
        
        # Selección de variables para clustering
        vars_clustering = st.multiselect(
            "Selecciona variables para clustering:", 
            columnas_finales, 
            default=columnas_finales[:min(2, len(columnas_finales))],
            key=f"vars_cluster_{nombre}"
        )
        
        if len(vars_clustering) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### K-means")
                with st.spinner("Aplicando K-means..."):
                    clusters_kmeans, best_k, silhouette_scores, inertias, kmeans_model = aplicar_kmeans(df_final, vars_clustering)
                    st.success(f"Número óptimo de clusters: {best_k}")
                    st.info(f"Silhouette Score: {silhouette_scores[best_k-2]:.3f}")
            
            with col2:
                st.markdown("##### DBSCAN")
                with st.spinner("Aplicando DBSCAN..."):
                    clusters_dbscan, best_params_dbscan, score_dbscan = aplicar_dbscan(df_final, vars_clustering)
                    n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
                    ruido = sum(clusters_dbscan == -1)
                    st.success(f"Clusters encontrados: {n_clusters_dbscan}")
                    st.info(f"Registros de ruido: {ruido} ({ruido/len(df_final)*100:.1f}%)")

            # Agregar clusters al dataframe
            df_final['Cluster_KMeans'] = clusters_kmeans
            df_final['Cluster_DBSCAN'] = clusters_dbscan

            # Visualización
            st.markdown("#### Visualización de Clusters")
            visualizar_clustering(df_final, clusters_kmeans, clusters_dbscan, vars_clustering, nombre)

            # Generar insights automáticos
            st.markdown("#### Conclusiones Finales")
            
            if st.button(f"Generar Insights", key=f"insights_{nombre}"):
                with st.spinner("Generando insights..."):
                    
                    if st.session_state['archivos'][nombre].get('insights', None) is not None:
                        insights = st.session_state['archivos'][nombre]['insights']
                    else:
                        insights = generar_insights(df_final, clusters_kmeans, clusters_dbscan, vars_clustering, nombre)
                        st.session_state['archivos'][nombre]['insights'] = insights

                    # Mostrar insights en formato de tabla
                    st.success(f"Generados {len(insights)} insights")
                    
                    # Crear DataFrame para mostrar insights
                    insights_df = pd.DataFrame({
                        'Tipo_Insight': [insight.split(':')[0] for insight in insights],
                        'Valor': [':'.join(insight.split(':')[1:]).strip() if ':' in insight else insight for insight in insights]
                    })
                    
                    st.dataframe(insights_df, use_container_width=True)
                    
                    # Exportar insights como texto plano para IA
                    insights_texto = "\n".join(insights)
                    st.session_state['insights_finales'] = insights_texto
                    insights = generar_insights_ia(insights_texto, st.session_state['model'])
                    st.markdown("#### Insights Generados por IA")
                    st.markdown(insights)
            
            # Guardar dataframe final
            st.session_state['archivos'][nombre]['dataframe_clustering'] = df_final
            
        else:
            st.warning("Selecciona al menos 2 variables para clustering")
    st.session_state['pasar_a_q&a'] = True

else:
    st.warning("Primero completa la sección de Limpieza de Datos.")
    st.session_state['pasar_a_q&a'] = False