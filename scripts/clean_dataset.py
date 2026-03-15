import json
import os

# ================= CONFIGURACIÓN =================
INPUT_PATH = './data/dataset_algoritmia/train_dataset.jsonl'
OUTPUT_PATH = './data/dataset_algoritmia/train_dataset_clean.jsonl'

# ================= VERIFICAR ARCHIVO =================
if not os.path.exists(INPUT_PATH):
    print("-" * 50)
    print("[ERROR] Archivo no encontrado:", INPUT_PATH)
    print("-" * 50)
    exit()

# ================= LEER DATASET ORIGINAL =================
print("-" * 50)
print("[INFO] Leyendo dataset original...")
print("-" * 50)

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"[INFO] Lineas totales: {len(lines)}")

# ================= PROCESAR Y LIMPIAR =================
seen = set()
clean_examples = []
stats = {
    'originales': len(lines),
    'json_invalido': 0,
    'estructura_invalida': 0,
    'truncados': 0,
    'duplicados': 0,
    'variacion_removida': 0,
    'limpios': 0
}

for line_num, line in enumerate(lines, 1):
    try:
        # Limpiar línea
        line = line.strip()
        
        # Saltar líneas vacías
        if not line:
            continue
        
        # Remover caracter "?" inicial si existe (error de formato)
        if line.startswith('?'):
            line = line[1:]
        
        # Parsear JSON
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            stats['json_invalido'] += 1
            continue
        
        # Verificar estructura mínima
        if 'messages' not in data or len(data['messages']) < 3:
            stats['estructura_invalida'] += 1
            continue
        
        # Procesar contenido del asistente
        assistant_content = None
        for msg in data['messages']:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                
                # Remover tags "# Variación X"
                original_content = msg['content']
                msg['content'] = '\n'.join(
                    l for l in msg['content'].split('\n') 
                    if not l.strip().startswith('# Variación')
                )
                if original_content != msg['content']:
                    stats['variacion_removida'] += 1
        
        # Si no hay contenido de asistente, saltar
        if assistant_content is None:
            stats['estructura_invalida'] += 1
            continue
        
        # Verificar que no esté truncado
        if assistant_content.endswith('...') or assistant_content.endswith('...'):
            stats['truncados'] += 1
            continue
        
        # Verificar longitud mínima (100 caracteres)
        if len(assistant_content) < 100:
            stats['truncados'] += 1
            continue
        
        # Crear clave única para detectar duplicados
        user_content = data['messages'][1]['content'] if len(data['messages']) > 1 else ''
        content_key = (
            user_content[:100],
            assistant_content[:200]
        )
        
        # Solo agregar si no es duplicado
        if content_key in seen:
            stats['duplicados'] += 1
            continue
        
        seen.add(content_key)
        clean_examples.append(data)
        stats['limpios'] += 1
        
    except Exception as e:
        stats['estructura_invalida'] += 1
        continue

# ================= IMPRIMIR RESULTADOS =================
print("\n" + "=" * 50)
print("RESUMEN DE LIMPIEZA")
print("=" * 50)
print(f"[OK] Originales:        {stats['originales']}")
print(f"[OK] Limpios:           {stats['limpios']}")
print(f"[--] JSON inválido:     {stats['json_invalido']}")
print(f"[--] Estructura inválida: {stats['estructura_invalida']}")
print(f"[--] Truncados:         {stats['truncados']}")
print(f"[--] Duplicados:        {stats['duplicados']}")
print(f"[--] Tags variación:    {stats['variacion_removida']}")
print("-" * 50)

if stats['limpios'] > 0:
    porcentaje_limpio = (stats['limpios'] / stats['originales']) * 100
    print(f"[OK] Porcentaje útil:   {porcentaje_limpio:.2f}%")
else:
    print("[ERROR] No hay ejemplos válidos para guardar")
    print("=" * 50)
    exit()

if stats['limpios'] < 100:
    print("[WARNING] Muy pocos ejemplos limpios (<100)")
    print("[INFO] Considera generar más datos antes de entrenar")
print("=" * 50)

# ================= GUARDAR DATASET LIMPIO =================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for example in clean_examples:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

print("\n[OK] Dataset limpio guardado en:", OUTPUT_PATH)
print("[OK] Total ejemplos guardados:", len(clean_examples))
print("=" * 50)

# ================= VALIDACIÓN FINAL =================
print("\n" + "=" * 50)
print("VALIDACIÓN FINAL")
print("=" * 50)

# Verificar archivo guardado
with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
    saved_lines = f.readlines()

print(f"[OK] Lineas guardadas: {len(saved_lines)}")

# Verificar que todos sean JSON válidos
valid_count = 0
for line in saved_lines:
    try:
        json.loads(line.strip())
        valid_count += 1
    except:
        pass

if valid_count == len(saved_lines):
    print("[OK] Todos los ejemplos son JSON válido")
else:
    print(f"[ERROR] {len(saved_lines) - valid_count} ejemplos con JSON inválido")

print("=" * 50)
print("[COMPLETADO] Limpieza finalizada exitosamente")
print("=" * 50)