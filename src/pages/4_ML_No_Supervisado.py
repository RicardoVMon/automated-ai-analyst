import streamlit as st
import pandas as pd
from modules.ml import generar_insights, generar_insights_ia, aplicar_kmeans, aplicar_dbscan, visualizar_clustering, agregar_o_actualizar_insight_final
from modules.settings import cargar_api_gemini

model = cargar_api_gemini()

# --- Punto de entrada ---
if st.session_state.get('pasar_a_analisis') == True:
    st.title("Análisis de Aprendizaje No Supervisado")
    st.markdown("Detecta valores atípicos y agrupa datos automáticamente con K-means y DBSCAN")

    archivos = list(st.session_state['archivos'].items())
    nombres = [nombre for nombre, _ in archivos]
    tabs = st.tabs(nombres)
    
    for i, (nombre, datos) in enumerate(archivos):
        with tabs[i]:
            df_final = datos['dataframe_final']
            clasificacion = datos.get('clasificacion', {})

            st.markdown(f"### Archivo: `{nombre}`")
            columnas_finales = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df_final.columns]

            if not columnas_finales:
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

                # Agregar clusters a un dataframe copiado para que no se muestren las columnas de clusters en la selección de variables para clustering
                df_clusters = df_final.copy()
                df_clusters['Cluster_KMeans'] = clusters_kmeans
                df_clusters['Cluster_DBSCAN'] = clusters_dbscan

                # Visualización
                st.markdown("#### Visualización de Clusters")
                visualizar_clustering(df_clusters, clusters_kmeans, clusters_dbscan, vars_clustering, nombre)

                
                insights = None
                if st.session_state['archivos'][nombre].get('insights') is not None:
                    insights = st.session_state['archivos'][nombre]['insights']
                    
                    
                if st.button(f"Generar Insights", key=f"gen_insights_{nombre}"):
                    with st.spinner(f"Generando insights para {nombre}..."):
                        insights = generar_insights(df_clusters, clusters_kmeans, clusters_dbscan, vars_clustering, nombre)
                        st.success(f"Generados {len(insights)} insights")
                        
                        # Crear DataFrame para mostrar insights
                        insights_df = pd.DataFrame({
                            'Tipo_Insight': [insight.split(':')[0] for insight in insights],
                            'Valor': [':'.join(insight.split(':')[1:]).strip() if ':' in insight else insight for insight in insights]
                        })
                        
                        st.dataframe(insights_df, use_container_width=True)
                        
                        # Exportar insights como texto plano para IA
                        insights_texto = "\n".join(insights)
                        agregar_o_actualizar_insight_final(nombre, vars_clustering, insights_texto)
                        insights = generar_insights_ia(insights_texto, model)
                        st.session_state['archivos'][nombre]['insights'] = insights
                        

                    st.session_state.reiniciar_chat = True  # Indica que se debe reiniciar el chat para mostrar los insights nuevos
                
                # Generar insights automáticos
                if insights is not None:
                    st.markdown("#### Insights Generados")
                    st.markdown("Insights generados automáticamente:")
                    st.markdown(insights)

                    st.download_button(
                        label="Descargar Insights",
                        data=insights.encode('utf-8'),
                        file_name=f"insights_{nombre}.txt",  # o .txt si prefieres
                        mime="text/markdown"
                    )

                # Guardar dataframe final
                st.session_state['archivos'][nombre]['dataframe_clustering'] = df_clusters

            else:
                st.warning("Selecciona al menos 2 variables para clustering")

    # Control de acceso a Q&A
    if st.session_state.get('insights_finales') and len(st.session_state['insights_finales']) > 0:
        st.session_state['pasar_a_q&a'] = True
    else:
        st.session_state['pasar_a_q&a'] = False

else:
    st.warning("Primero completa la sección de Limpieza de Datos.")
    st.session_state['pasar_a_q&a'] = False