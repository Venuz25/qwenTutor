import torch
import os
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ================= CONFIGURACIÓN =================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "./data/train_dataset_clean.jsonl"
OUTPUT_DIR = "./models/qwen-algo-tutor-1.5b-v3"  # ← Sin espacio

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Verificar Dataset =================
print("="*60)
print("📊 VERIFICANDO DATASET...")
print("="*60)

valid_examples = 0
invalid_examples = 0
duplicate_keys = set()
duplicates = 0

try:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Verificar estructura mínima
                if 'messages' not in data or len(data['messages']) < 3:
                    invalid_examples += 1
                    continue
                
                # Verificar que no esté truncado
                assistant_content = data['messages'][-1]['content']
                if assistant_content.endswith('...') or len(assistant_content) < 100:
                    invalid_examples += 1
                    continue
                
                # Detectar duplicados
                content_key = (
                    data['messages'][1]['content'][:50],
                    data['messages'][2]['content'][:100]
                )
                
                if content_key in duplicate_keys:
                    duplicates += 1
                    continue
                
                duplicate_keys.add(content_key)
                valid_examples += 1
                
            except Exception as e:
                invalid_examples += 1
                continue
    
    print(f"   Ejemplos válidos únicos: {valid_examples}")
    print(f"   Ejemplos inválidos: {invalid_examples}")
    print(f"   Duplicados detectados: {duplicates}")
    if valid_examples + duplicates > 0:
        print(f"   Porcentaje duplicación: {(duplicates/(valid_examples+duplicates))*100:.2f}%")
    
    if valid_examples < 100:
        print("   ERROR: Muy pocos ejemplos válidos. Ejecutá el script de limpieza primero.")
        exit()
    
    if duplicates > valid_examples * 0.3:
        print("   WARNING: Más del 30% de duplicación. El script de limpieza debería haberlo resuelto.")
    
except FileNotFoundError:
    print(f"   ERROR: Dataset no encontrado en {DATASET_PATH}")
    exit()

# ================= Cuantización QLoRA =================
print("\n" + "="*60)
print("CONFIGURANDO CUANTIZACIÓN QLoRA...")
print("="*60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

# ================= Tokenizer =================
print("\nCargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= Modelo =================
print("Cargando modelo Qwen2.5-1.5B (esto puede tardar 2-5 minutos)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ================= LoRA =================
print("\nConfigurando LoRA...")

# Ajustar r según cantidad de ejemplos únicos
if valid_examples < 500:
    lora_r = 8
    lora_alpha = 16
    print(f"   Dataset pequeño: r={lora_r} (para evitar overfitting)")
elif valid_examples < 1000:
    lora_r = 16
    lora_alpha = 32
    print(f"   Dataset mediano: r={lora_r}")
else:
    lora_r = 16
    lora_alpha = 32
    print(f"   Dataset grande: r={lora_r}")

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= Dataset =================
print("\nCargando dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"   Ejemplos entrenamiento: {len(dataset['train'])}")
print(f"   Ejemplos evaluación: {len(dataset['test'])}")

# ================= Entrenamiento =================
print("\nConfigurando entrenamiento...")

if valid_examples < 500:
    num_epochs = 5
    print(f"   Dataset pequeño: {num_epochs} epochs")
elif valid_examples < 1000:
    num_epochs = 3
    print(f"   Dataset mediano: {num_epochs} epochs")
else:
    num_epochs = 2
    print(f"   Dataset grande: {num_epochs} epochs")

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=200,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

# ================= SFTTrainer =================
print("\nCreando trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    packing=False,
)

# ================= Iniciar Entrenamiento =================
print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO...")
print("="*60)
print(f"   Tiempo estimado: {(len(dataset['train']) * num_epochs * 0.005):.1f} horas")
print(f"   VRAM esperada: ~2.5-3.2 GB") 
print(f"   Guardado en: {os.path.abspath(OUTPUT_DIR)}")
print(f"   Ejemplos únicos: {valid_examples}")
print(f"   Epochs: {num_epochs}")
print(f"   Max seq length: 256 tokens")
print(f"   Evaluación: Desactivada (para evitar OOM)")
print("="*60 + "\n")

trainer.train()

# ================= Guardar Modelo =================
print("\n" + "="*60)
print("GUARDANDO MODELO...")
print("="*60)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Guardar metadata del entrenamiento
metadata = {
    "model_name": MODEL_NAME,
    "dataset_path": DATASET_PATH,
    "train_examples": len(dataset['train']),
    "eval_examples": len(dataset['test']),
    "unique_examples": valid_examples,
    "duplicates_removed": duplicates,
    "epochs": num_epochs,
    "lora_r": lora_r,
    "lora_alpha": lora_alpha,
    "max_seq_length": 256,
    "train_loss": trainer.state.log_history[-1].get('loss', 'N/A') if trainer.state.log_history else 'N/A',
    "eval_loss": "N/A (eval desactivada)",
}

with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"   ¡ENTRENAMIENTO COMPLETADO!")
print(f"   Modelo guardado en: {os.path.abspath(OUTPUT_DIR)}")
print(f"   Metadata guardada en: {os.path.join(OUTPUT_DIR, 'training_metadata.json')}")
print("="*60)

# ================= Resumen Final =================
print("\n" + "="*60)
print("RESUMEN DEL ENTRENAMIENTO")
print("="*60)
print(f"   Dataset original: {valid_examples + duplicates} ejemplos")
print(f"   Dataset único: {valid_examples} ejemplos")
print(f"   Duplicación: {(duplicates/(valid_examples+duplicates))*100:.2f}%")
print(f"   Epochs: {num_epochs}")
print(f"   LoRA Rank: {lora_r}")
print(f"   Train Loss Final: {metadata['train_loss']}")
print(f"   Eval Loss Final: {metadata['eval_loss']}")

if valid_examples > 800 and duplicates < valid_examples * 0.1:
    print("\n   CALIDAD DEL DATASET: EXCELENTE")
elif valid_examples > 500 and duplicates < valid_examples * 0.2:
    print("\n   CALIDAD DEL DATASET: BUENA")
else:
    print("\n   CALIDAD DEL DATASET: MEJORABLE (considerá limpiar más)")

print("="*60)