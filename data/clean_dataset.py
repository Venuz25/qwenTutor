import json

# Leer dataset original
with open('./data/dataset_algoritmia/train_dataset.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Procesar y limpiar
seen = set()
clean_examples = []

for line in lines:
    try:
        data = json.loads(line)
        
        # Remover tags "# Variación X" del contenido
        for msg in data['messages']:
            if msg['role'] == 'assistant':
                # Remover líneas que empiezan con # Variación
                msg['content'] = '\n'.join(
                    l for l in msg['content'].split('\n') 
                    if not l.strip().startswith('# Variación')
                )
        
        # Crear clave única para detectar duplicados
        content_key = (
            data['messages'][1]['content'][:100],
            data['messages'][2]['content'][:200]
        )
        
        # Solo agregar si no es duplicado
        if content_key not in seen:
            seen.add(content_key)
            clean_examples.append(data)
    except Exception as e:
        print(f"Error procesando linea: {e}")
        continue

# Imprimir resultados (con caracteres ASCII para Windows)
print("-" * 50)
print(f"[OK] Originales: {len(lines)}")
print(f"[OK] Limpios: {len(clean_examples)}")
print(f"[OK] Duplicados removidos: {len(lines) - len(clean_examples)}")
print("-" * 50)

# Guardar dataset limpio
with open('./data/dataset_algoritmia/train_dataset_clean.jsonl', 'w', encoding='utf-8') as f:
    for example in clean_examples:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

print("[OK] Dataset limpio guardado en: ./data/dataset_algoritmia/train_dataset_clean.jsonl")
print("-" * 50)