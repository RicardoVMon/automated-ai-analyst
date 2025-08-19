def prompt_clasificacion_variables(df):
    columnas = list(df.columns)
    muestra = df.head(2).to_dict(orient='records')

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
    columnas = list(df.columns)
    muestra = df.head(2).to_dict(orient='records')

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