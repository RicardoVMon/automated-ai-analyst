def prompt_clasificacion_variables(df):
    """
    Genera un prompt para clasificar automáticamente las variables de un DataFrame.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos a analizar
    
    Retorna:
    --------
    str
        Prompt estructurado para que la IA clasifique las variables
    
    Proceso:
    --------
    1. Extrae los nombres de todas las columnas del DataFrame
    2. Obtiene una muestra de 2 filas para dar contexto a la IA
    3. Estructura un prompt solicitando clasificación en categorías específicas
    4. Define formato de respuesta esperado
    """
    # Obtener lista de nombres de columnas del DataFrame
    columnas = list(df.columns)
    
    # Extraer muestra de 2 filas como diccionario para contexto
    muestra = df.head(2).to_dict(orient='records')

    # Crear prompt estructurado para clasificación automática
    prompt = f"""
        Tengo un conjunto de datos con las siguientes columnas:

        {columnas}

        Y estas son dos filas de ejemplo de datos:

        {muestra}

        Por favor, clasifica cada columna en una de las siguientes categorías:

        - Numérica
        - Categórica
        - Temporal
        - Booleana
        - Desconocida

        Devuelve la clasificación en formato:

         TipoColumna1:
        Columna2: Tipo
        ...

        No agregues nada más.
        """
    return prompt

def prompt_describir_archivo(df):
    """
    Genera un prompt para obtener una descripción general del dataset.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos a describir
    
    Retorna:
    --------
    str
        Prompt para generar descripción narrativa del dataset
    
    Proceso:
    --------
    1. Extrae nombres de columnas para dar contexto
    2. Incluye muestra de datos para análisis
    3. Solicita descripción en prosa y bullet points
    4. Especifica que no incluya tipos de variables
    """
    # Obtener lista de nombres de columnas
    columnas = list(df.columns)
    
    # Extraer muestra de 2 filas para contexto
    muestra = df.head(2).to_dict(orient='records')

    # Crear prompt para descripción narrativa del dataset
    prompt = f"""
        Tengo un conjunto de datos con las siguientes columnas:
        {columnas}
        Y estas son dos filas de ejemplo de datos:
        {muestra}
        Por favor, proporciona una descripción general del conjunto de datos, en prosa y 
        describe en bulletpoints cada variable, sin especificar su tipo.
        """
    return prompt

def prompt_generar_insights(insights):
    """
    Genera un prompt para convertir datos estadísticos en insights interpretativos.
    
    Parámetros:
    -----------
    insights : list o str
        Lista de insights estructurados o texto con análisis estadístico
    
    Retorna:
    --------
    str
        Prompt para generar insights interpretativos en lenguaje natural
    
    Proceso:
    --------
    1. Incluye los datos estadísticos como contexto
    2. Solicita interpretación explicativa, no solo descriptiva
    3. Define formato específico (bullet points, extensión)
    4. Limita cantidad de insights para mantener calidad
    """
    # Crear prompt para generar insights interpretativos
    prompt = f"""
        Analiza el siguiente resumen estadístico y de clustering de un dataset relacionado con calidad del aire.

        Tu objetivo es generar insights generales, explicativos e interpretativos sobre el comportamiento de los datos. No te limites a describir correlaciones o porcentajes: debes generar ideas sobre lo que podría significar o explicar los datos, en frases como:

        - "Esto sugiere que..."
        - "Una posible explicación es..."
        - "Se observa una tendencia a..."
        - "Esto podría deberse a..."

        El formato debe ser:
        - En bullet points de Markdown (`-`)
        - Cada frase debe tener una extensión similar a 40 palabras:  
        - No des más de 8 insights

        Aquí está el resumen del dataset:

        {insights}

        """
    return prompt