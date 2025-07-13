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

        Columna1: Tipo
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
