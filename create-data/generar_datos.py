import os
import json
import time
import glob
from google import genai
from google.genai import types
from google.genai.errors import APIError

# 1. Configuración
cliente = genai.Client(api_key="AIzaSyBb4Q16o3GXhZUA10LQo3HCCST2TfQyCkc")
modelo_elegido = "gemini-2.5-flash"
input_dir = "./create-data/filtered_datasets"
output_file = "./data/train_dataset.jsonl"

prompt_maestro = """
Eres un experto creador de datasets de entrenamiento. Tu tarea es leer un problema de programación y su solución, y transformarlo en un diálogo socrático de 4 turnos en español.

Reglas estrictas:
1. El usuario hace la pregunta inicial basada en el problema.
2. Tú (el asistente) respondes como un tutor socrático: NO des el código final. Da una pista conceptual, explica la lógica o haz una pregunta que guíe al usuario.
3. El usuario responde intentando razonar o mostrando un código a medias con dudas.
4. Tú respondes validando su esfuerzo y guiándolo al siguiente paso lógico.
5. El tono debe ser alentador, técnico y en español neutro.

Formato de salida OBLIGATORIO (JSON estricto, sin markdown):
[
  {"role": "user", "content": "Pregunta inicial..."},
  {"role": "assistant", "content": "Respuesta socrática..."},
  {"role": "user", "content": "Intento del estudiante..."},
  {"role": "assistant", "content": "Siguiente pista..."}
]

Aquí tienes el problema y la solución original para transformar:
"""

input_files = glob.glob(f"{input_dir}/*.jsonl")

# Contar líneas para poder reanudar
lineas_procesadas = 0
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        lineas_procesadas = sum(1 for _ in f)
    print(f"Reanudando desde el ejemplo {lineas_procesadas}...")

contador_actual = 0

for file_path in input_files:
    print(f"Abriendo {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            contador_actual += 1
            
            if contador_actual <= lineas_procesadas:
                continue
                
            datos_crudos = line.strip()
            prompt_final = prompt_maestro + "\n\nDatos originales:\n" + datos_crudos
            
            exito = False
            reintentos = 0
            
            while not exito and reintentos < 3: # Intentamos hasta 3 veces por ejemplo
                try:
                    response = cliente.models.generate_content(
                        model=modelo_elegido,
                        contents=prompt_final,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                        )
                    )
                    
                    # Intentamos parsear el JSON. Si falla, va al 'except json.JSONDecodeError'
                    texto_limpio = response.text.strip()
                    # A veces los modelos meten ```json al principio, esto lo limpia por si acaso
                    if texto_limpio.startswith("```json"):
                        texto_limpio = texto_limpio[7:-3]
                        
                    dialogo_json = json.loads(texto_limpio)
                    
                    chatml_format = {
                        "messages": [
                            {"role": "system", "content": "Eres un tutor socrático experto en programación. Guías al estudiante paso a paso sin darle el código completo de inmediato."}
                        ] + dialogo_json
                    }
                    
                    with open(output_file, 'a', encoding='utf-8') as outfile:
                        json.dump(chatml_format, outfile, ensure_ascii=False)
                        outfile.write('\n')
                    
                    print(f"[{contador_actual}/6000] Ejemplo generado con éxito.")
                    exito = True
                    
                    # Aumentamos el tiempo de espera base a 5 segundos para ayudar a la cuota
                    time.sleep(5)
                    
                except json.JSONDecodeError as e:
                    print(f"Error de JSON en ejemplo {contador_actual} (Reintento {reintentos+1}): El modelo generó un formato inválido.")
                    reintentos += 1
                    time.sleep(2) # Pausa corta antes de reintentar
                    
                except APIError as e:
                    # Capturamos el error de Google y buscamos si es el 429
                    if e.code == 429:
                        print(f"Límite de cuota alcanzado en el ejemplo {contador_actual}. Pausando por 10 minutos...")
                        time.sleep(600) 
                    else:
                        print(f"Error de API inesperado en ejemplo {contador_actual}: {e.message}")
                        reintentos += 1
                        time.sleep(10)
                        
                except Exception as e:
                    print(f"Error general en el ejemplo {contador_actual}: {e}")
                    reintentos += 1
                    time.sleep(5)
            
            if not exito:
                print(f"⏭Se saltó el ejemplo {contador_actual} después de 3 intentos fallidos.")

print("¡Proceso finalizado!")
