from modules.prompts import prompt_clasificacion_variables, prompt_describir_archivo

def clasificar_variables(df, model):
    prompt = prompt_clasificacion_variables(df)
    response = model.generate_content(prompt)
    texto = response.text.strip()

    clasificacion = {}
    for linea in texto.splitlines():
        if ':' in linea:
            col, tipo = linea.split(':', 1)
            clasificacion[col.strip()] = tipo.strip()
    return clasificacion

def describir_archivo(df, model):
    prompt = prompt_describir_archivo(df)
    response = model.generate_content(prompt)
    texto = response.text.strip()
    return texto