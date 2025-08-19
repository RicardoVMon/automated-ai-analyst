# =========================
# Importación de librerías y módulos
# =========================

import streamlit as st  # Importa Streamlit para la interfaz web
import pandas as pd  # Importa pandas para manipulación de datos
from modules.ml import generar_insights, generar_insights_ia, aplicar_kmeans, aplicar_dbscan, visualizar_clustering, agregar_o_actualizar_insight_final  # Importa funciones personalizadas de ML
from modules.settings import cargar_api_gemini  # Importa función para cargar el modelo de IA Gemini

# =========================
# Carga del modelo de IA
# =========================

model = cargar_api_gemini()  # Inicializa el modelo de IA Gemini

# =========================
# Análisis de Aprendizaje No Supervisado
# =========================

if st.session_state.get('pasar_a_analisis') == True:  # Verifica si se puede avanzar a la etapa de análisis
    st.title("Análisis de Aprendizaje No Supervisado")  # Muestra el título principal de la página
    st.markdown("Detecta valores atípicos y agrupa datos automáticamente con K-means y DBSCAN")  # Explicación breve

    archivos = list(st.session_state['archivos'].items())  # Obtiene la lista de archivos y sus datos del estado de sesión
    nombres = [nombre for nombre, _ in archivos]  # Extrae los nombres de los archivos
    tabs = st.tabs(nombres)  # Crea una pestaña para cada archivo
    
    for i, (nombre, datos) in enumerate(archivos):  # Itera sobre cada archivo y sus datos
        with tabs[i]:  # Selecciona la pestaña correspondiente
            df_final = datos['dataframe_final']  # Obtiene el DataFrame final procesado
            clasificacion = datos.get('clasificacion', {})  # Obtiene la clasificación de variables

            st.markdown(f"### Archivo: `{nombre}`")  # Muestra el nombre del archivo como subtítulo
            columnas_finales = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df_final.columns]  # Selecciona columnas numéricas

            if not columnas_finales:  # Si no hay columnas numéricas
                st.warning("No se encontraron columnas numéricas para el análisis.")  # Advierte al usuario
                continue  # Pasa al siguiente archivo

            # =========================
            # Análisis de Clustering
            # =========================
            st.markdown("#### Análisis de Clustering")  # Muestra subtítulo
            
            # Selección de variables para clustering
            vars_clustering = st.multiselect(
                "Selecciona variables para clustering:",  # Texto de ayuda
                columnas_finales,  # Opciones disponibles
                default=st.session_state['archivos'][nombre].get('vars_clustering', columnas_finales[:min(2, len(columnas_finales))]),  # Selección por defecto
                key=f"vars_cluster_{nombre}"
            )

            # Guarda las variables de clustering seleccionadas en el session state del archivo actual
            st.session_state['archivos'][nombre]['vars_clustering'] = vars_clustering
            
            if len(vars_clustering) >= 2:  # Si se seleccionan al menos 2 variables
                col1, col2 = st.columns(2)  # Divide la pantalla en dos columnas
                
                with col1:
                    st.markdown("##### K-means")  # Subtítulo para K-means
                    with st.spinner("Aplicando K-means..."):  # Muestra spinner de carga
                        clusters_kmeans, best_k, silhouette_scores, inertias, kmeans_model = aplicar_kmeans(df_final, vars_clustering)  # Aplica K-means
                        st.success(f"Número óptimo de clusters: {best_k}")  # Muestra el número óptimo de clusters
                        st.info(f"Silhouette Score: {silhouette_scores[best_k-2]:.3f}")  # Muestra el Silhouette Score
                
                with col2:
                    st.markdown("##### DBSCAN")  # Subtítulo para DBSCAN
                    with st.spinner("Aplicando DBSCAN..."):  # Muestra spinner de carga
                        clusters_dbscan, best_params_dbscan, score_dbscan = aplicar_dbscan(df_final, vars_clustering)  # Aplica DBSCAN
                        n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)  # Calcula número de clusters
                        ruido = sum(clusters_dbscan == -1)  # Cuenta registros de ruido
                        st.success(f"Clusters encontrados: {n_clusters_dbscan}")  # Muestra número de clusters
                        st.info(f"Registros de ruido: {ruido} ({ruido/len(df_final)*100:.1f}%)")  # Muestra porcentaje de ruido

                # Agregar clusters al DataFrame copiado para visualización
                df_clusters = df_final.copy()  # Copia el DataFrame final
                df_clusters['Cluster_KMeans'] = clusters_kmeans  # Agrega columna de clusters KMeans
                df_clusters['Cluster_DBSCAN'] = clusters_dbscan  # Agrega columna de clusters DBSCAN

                # =========================
                # Visualización de Clusters
                # =========================
                st.markdown("#### Visualización de Clusters")  # Muestra subtítulo
                visualizar_clustering(df_clusters, clusters_kmeans, clusters_dbscan, vars_clustering, nombre)  # Visualiza los clusters

                # =========================
                # Generación de Insights
                # =========================
                insights = None
                if st.session_state['archivos'][nombre].get('insights') is not None:  # Si ya existen insights previos
                    insights = st.session_state['archivos'][nombre]['insights']
                    
                if st.button(f"Generar Insights", key=f"gen_insights_{nombre}"):  # Botón para generar insights
                    with st.spinner(f"Generando insights para {nombre}..."):  # Muestra spinner de carga
                        insights = generar_insights(df_clusters, clusters_kmeans, clusters_dbscan, vars_clustering, nombre)  # Genera insights automáticos
                        st.success(f"Generados {len(insights)} insights")  # Muestra mensaje de éxito
                        
                        # Crear DataFrame para mostrar insights
                        insights_df = pd.DataFrame({
                            'Tipo_Insight': [insight.split(':')[0] for insight in insights],
                            'Valor': [':'.join(insight.split(':')[1:]).strip() if ':' in insight else insight for insight in insights]
                        })
                        
                        st.dataframe(insights_df, use_container_width=True)  # Muestra insights en tabla
                        
                        # Exportar insights como texto plano para IA
                        insights_texto = "\n".join(insights)  # Une los insights en texto plano
                        agregar_o_actualizar_insight_final(nombre, vars_clustering, insights_texto)  # Actualiza insights finales
                        insights = generar_insights_ia(insights_texto, model)  # Genera insights usando IA
                        st.session_state['archivos'][nombre]['insights'] = insights  # Guarda insights en sesión

                    st.session_state.reiniciar_chat = True  # Indica que se debe reiniciar el chat para mostrar los insights nuevos
                
                # =========================
                # Visualización y descarga de Insights
                # =========================
                if insights is not None:  # Si existen insights generados
                    st.markdown("#### Insights Generados")  # Muestra subtítulo
                    st.markdown("Insights generados automáticamente:")  # Texto explicativo
                    st.markdown(insights)  # Muestra los insights

                    st.download_button(
                        label="Descargar Insights",  # Etiqueta del botón
                        data=insights.encode('utf-8'),  # Datos a descargar
                        file_name=f"insights_{nombre}.txt",  # Nombre del archivo de descarga
                        mime="text/markdown"  # Tipo MIME
                    )
            else:
                st.warning("Selecciona al menos 2 variables para clustering")  # Advierte si no hay suficientes variables

    # =========================
    # Control de acceso a Q&A
    # =========================
    st.session_state['pasar_a_q&a'] = bool(st.session_state.get('insights_finales')) and len(st.session_state['insights_finales']) > 0

else:
    st.warning("Primero completa la sección de Limpieza de Datos.")  # Advierte si no se ha completado la etapa anterior
    st.session_state['pasar_a_q&a'] = False  # No permite avanzar a la etapa de Q&A