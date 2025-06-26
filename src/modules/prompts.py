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

def prompt_relacion_semantica(dataframes):
    info_archivos = []
    for i, df in enumerate(dataframes, start=1):
        columnas = list(df.columns)
        muestra = df.head(2).to_dict(orient='records')
        info_archivos.append(f"Archivo {i}:\n- Columnas: {columnas}\n- Ejemplos (2 filas): {muestra}")

    info_texto = "\n\n".join(info_archivos)

    prompt = f"""
        Tengo {len(dataframes)} conjuntos de datos representados por {len(dataframes)} archivos. Necesito saber si **alguno de estos archivos está relacionado semánticamente con alguno de los otros**.

        Considera:
        - Relación semántica significa que los datos tienen sentido juntos, pertenecen a un mismo dominio o contexto.
        - No te bases solo en nombres similares o valores iguales, sino en el significado real.
        - Por ejemplo, archivos de perros y archivos de aviones NO están relacionados.
        - Devuélveme solo "Sí" si todos los archivos forman parte de un mismo contexto semántico, o "No" si no tienen relación.

        Aquí está la información resumida de cada archivo:

        {info_texto}

        Responde solo con "Sí" o "No".
        """
    return prompt

def relacion_semantica(dataframes, model):
    prompt = prompt_relacion_semantica(dataframes)
    response = model.generate_content(prompt)
    texto = response.text.strip().lower()
    if "sí" in texto or "si" in texto:
        return True
    elif "no" in texto:
        return False
    else:
        # Si la respuesta no es clara, mejor asumir False o manejar el caso
        return None