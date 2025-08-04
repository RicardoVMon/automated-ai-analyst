import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.preprocessing import mostrar_outliers

# --- Análisis de los datos ---
if st.session_state.get('pasar_a_analisis') == True:
    st.title("Preparación para IA")
    st.markdown("En esta sección, se realiza la codificación de las columnas categóricas y la estandarización de las columnas numéricas para análisis avanzado.")

    for nombre, datos in st.session_state['archivos'].items():
        df = datos['dataframe_limpio'] if 'dataframe_limpio' in datos else datos['dataframe']
        clasificacion = datos.get('clasificacion', {})

        if not clasificacion:
            st.warning(f"No se puede analizar el archivo '{nombre}' porque no tiene clasificación.")
            continue

        st.markdown(f"### {nombre}")

        # Identificar columnas categóricas y numéricas
        cat_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Categórica" and col.strip() in df.columns]
        num_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df.columns]

        if not cat_attribs and not num_attribs:
            st.info(f"No se encontraron columnas categóricas o numéricas en el archivo '{nombre}'.")
            continue

        # Crear el OneHotEncoder para columnas categóricas
        encoder = OneHotEncoder(handle_unknown='ignore')

        # Aplicar el encoder a las columnas categóricas
        try:
            encoded_cats = encoder.fit_transform(df[cat_attribs])

            # Crear un DataFrame con las columnas codificadas
            encoded_cats_df = pd.DataFrame(
                encoded_cats.toarray(),
                columns=encoder.get_feature_names_out(cat_attribs),
                index=df.index
            )

            # Concatenar las columnas codificadas con el DataFrame original
            df_encoded = pd.concat([df.drop(columns=cat_attribs), encoded_cats_df], axis=1)

            # Combinar columnas numéricas originales con las columnas codificadas
            all_num_attribs = num_attribs + list(encoded_cats_df.columns)

            # Aplicar el StandardScaler a todas las columnas numéricas y codificadas
            scaler = StandardScaler()
            scaled_nums = scaler.fit_transform(df_encoded[all_num_attribs])

            # Crear un DataFrame con las columnas estandarizadas
            scaled_nums_df = pd.DataFrame(
                scaled_nums,
                columns=all_num_attribs,
                index=df_encoded.index
            )

            # Reemplazar las columnas originales con las estandarizadas
            df_final = df_encoded.copy()
            df_final[all_num_attribs] = scaled_nums_df

            # Mostrar vista previa del DataFrame final
            st.markdown("#### Vista previa del dataFrame tras codificación y estandarización")
            st.dataframe(df_final.head(), use_container_width=True)

            # Mostrar gráfico de outliers
            st.markdown("#### Gráfico de Outliers")
            mostrar_outliers(df_final, num_attribs) 

            # 2. KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(df_final[all_num_attribs])
            # 3. Añadir al dataframe
            df_final["cluster"] = clusters

            st.markdown("#### Tabla de Datos Finales con Clusters")
            st.dataframe(df_final, use_container_width=True)

            # Media por cluster
            insight_table = df_final.groupby("cluster").mean(numeric_only=True)
            st.subheader("Promedios por grupo")
            st.dataframe(insight_table)

            # Visualizar agrupaciones con PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df_final[all_num_attribs])
            df_final["PC1"] = X_pca[:,0]
            df_final["PC2"] = X_pca[:,1]

            sns.scatterplot(data=df_final, x="PC1", y="PC2", hue="cluster", palette="Set2")
            plt.title("Clusters visualizados con PCA")
            st.pyplot(plt.gcf())

            # Guardar el DataFrame final en session_state
            st.session_state['archivos'][nombre]['dataframe_final'] = df_final

            st.success(f"Codificación y estandarización completadas para el archivo: {nombre}")
        except Exception as e:
            st.error(f"Error al procesar el archivo '{nombre}': {e}")

else:
    st.warning("Primero completa la sección de Limpieza de Datos.")