import os
from datasets import load_dataset

# Configuración de rutas
input_dir = "./create-data/datasets base"
output_dir = "./create-data/filtered_datasets"

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Mapeo de archivos locales
datasets_local_paths = {
    "leetcode": os.path.join(input_dir, "leetcode-train.jsonl"), 
    "codefeedback": os.path.join(input_dir, "CodeFeedback-Filtered-Instruction.jsonl"),
    "alpaca_code": os.path.join(input_dir, "code_instructions_120k.jsonl")
}

NUM_SAMPLES = 2000
SEED = 42 

# Función de filtro de "alta calidad" (heurística básica)
# Filtramos ejemplos que tengan menos de 50 caracteres en total para evitar ruido o respuestas vacías
def high_quality_filter(example):
    # Convertimos todo el registro a string para medir su longitud total rápidamente
    content_length = len(str(example))
    return content_length > 50

# Procesamiento iterativo
for name, file_path in datasets_local_paths.items():
    print(f"Procesando: {name}...")
    
    # Validar que el archivo o carpeta exista
    if not os.path.exists(file_path):
        print(f"⚠️ Advertencia: No se encontró la ruta {file_path}. Verifica el nombre y la extensión.\n")
        continue

    try:
        # Determinar el formato basado en la extensión
        file_ext = file_path.split('.')[-1]
        if file_ext == 'jsonl':
            file_ext = 'json' # La librería 'datasets' usa 'json' para leer archivos .jsonl
            
        # Cargar el dataset local
        dataset = load_dataset(file_ext, data_files=file_path, split="train")
        
        # Aplicar filtro de calidad
        dataset_filtered = dataset.filter(high_quality_filter)
        
        # Mezclar de forma aleatoria
        dataset_shuffled = dataset_filtered.shuffle(seed=SEED)
        
        # Seleccionar la muestra (protección en caso de que el dataset filtrado tenga menos de 5000)
        sample_size = min(NUM_SAMPLES, len(dataset_shuffled))
        dataset_sampled = dataset_shuffled.select(range(sample_size))
        
        # Guardar el resultado en formato JSONL (ideal para la siguiente etapa con la API)
        output_file = os.path.join(output_dir, f"{name}_2k_sample.jsonl")
        dataset_sampled.to_json(output_file, force_ascii=False)
        
        print(f"Guardado exitosamente: {output_file} ({sample_size} ejemplos)\n")
        
    except Exception as e:
        print(f"Error al procesar {name}: {e}\n")

print("¡Proceso de muestreo completado! Revisa la carpeta:", output_dir)