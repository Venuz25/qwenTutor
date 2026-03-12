import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ================= CONFIGURACIÓN =================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "./data/dataset_algoritmia/train_dataset.jsonl"
OUTPUT_DIR = "./models/qwen-algo-tutor-1.5b"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Cuantización QLoRA =================
print("="*60)
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
print("Cargando modelo Qwen2.5-1.5B ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ================= LoRA =================
print("\nConfigurando LoRA...")
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
print("\nCargando dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"   ✓ Ejemplos entrenamiento: {len(dataset['train'])}")
print(f"   ✓ Ejemplos evaluación: {len(dataset['test'])}")

# ================= Entrenamiento =================
print("\nConfigurando entrenamiento...")
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,
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
print(f"Tiempo estimado: 1-2 horas para 1000 ejemplos")
print(f"VRAM esperada: ~3-3.5 GB")
print(f"Guardado en: {os.path.abspath(OUTPUT_DIR)}")
print("="*60 + "\n")

trainer.train()

# ================= Guardar Modelo =================
print("\n" + "="*60)
print("GUARDANDO MODELO...")
print("="*60)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo guardado en: {os.path.abspath(OUTPUT_DIR)}")
print("="*60)