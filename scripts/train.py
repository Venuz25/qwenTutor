import torch
import os
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ================= CONFIGURACIÓN =================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "./data/dataset_algoritmia/train_dataset_clean.jsonl"
OUTPUT_DIR = "./models/qwen-algo-tutor-1.5b"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Verificar Dataset =================
print("="*60)
print("📊 VERIFICANDO DATASET...")
print("="*60)

# Contar ejemplos válidos antes de cargar
valid_examples = 0
invalid_examples = 0

try:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'messages' in data and len(data['messages']) >= 3:
                    valid_examples += 1
                else:
                    invalid_examples += 1
            except:
                invalid_examples += 1
    
    print(f"   ✓ Ejemplos válidos: {valid_examples}")
    print(f"   ⚠ Ejemplos inválidos: {invalid_examples}")
    
    if valid_examples < 100:
        print("   ❌ ERROR: Muy pocos ejemplos válidos. Verificá el dataset.")
        exit()
except FileNotFoundError:
    print(f"   ❌ ERROR: Dataset no encontrado en {DATASET_PATH}")
    exit()

# ================= Cuantización QLoRA =================
print("\n" + "="*60)
print("⚙️ CONFIGURANDO CUANTIZACIÓN QLoRA...")
print("="*60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

# ================= Tokenizer =================
print("\n📝 Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= Modelo =================
print("🤖 Cargando modelo Qwen2.5-1.5B (esto puede tardar 2-5 minutos)...")
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
print("\n🔧 Configurando LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= Dataset =================
print("\n📊 Cargando dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"   ✓ Ejemplos entrenamiento: {len(dataset['train'])}")
print(f"   ✓ Ejemplos evaluación: {len(dataset['test'])}")

# ================= Entrenamiento =================
print("\n📈 Configurando entrenamiento...")
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="none",
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    save_strategy="steps",
    dataloader_num_workers=0,
)

# ================= SFTTrainer =================
print("\n🎯 Creando trainer...")
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
print("🚀 INICIANDO ENTRENAMIENTO...")
print("="*60)
print(f"   ⏱️  Tiempo estimado: 1.5-2.5 horas para {len(dataset['train'])} ejemplos")
print(f"   💾 VRAM esperada: ~3-3.5 GB")
print(f"   📁 Guardado en: {os.path.abspath(OUTPUT_DIR)}")
print(f"   🌐 Dataset multi-lenguaje: Python, Java, C++, Go, etc.")
print("="*60 + "\n")

trainer.train()

# ================= Guardar Modelo =================
print("\n" + "="*60)
print("💾 GUARDANDO MODELO...")
print("="*60)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Guardar metadata del entrenamiento
metadata = {
    "model_name": MODEL_NAME,
    "dataset_path": DATASET_PATH,
    "train_examples": len(dataset['train']),
    "eval_examples": len(dataset['test']),
    "epochs": 3,
    "lora_r": 16,
    "max_seq_length": 512,
}

with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"   ✅ ¡ENTRENAMIENTO COMPLETADO!")
print(f"   📁 Modelo guardado en: {os.path.abspath(OUTPUT_DIR)}")
print(f"   📄 Metadata guardada en: {os.path.join(OUTPUT_DIR, 'training_metadata.json')}")
print("="*60)