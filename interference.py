from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# ================= CONFIGURACIÓN =================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./models/qwen-algo-tutor-1.5b"

# ================= Cargar Modelo =================
print("="*60)
print("🤖 CARGANDO MODELO...")
print("="*60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print(f"✅ Modelo cargado exitosamente")
print(f"💾 VRAM usada: ~2.5 GB")
print("="*60)

# ================= Función de Chat =================
def chat(prompt, max_tokens=256):
    messages = [
        {"role": "system", "content": "Eres un tutor experto en Algoritmia y Estructuras de Datos."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    attention_mask = torch.ones_like(inputs)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta.split("assistant\n")[-1].strip()

# ================= Bucle de Chat =================
print("\n" + "="*60)
print("🎓 TUTOR DE ALGORITMIA (Qwen2.5-1.5B)")
print("="*60)
print("Escribe 'salir' para terminar\n")

while True:
    try:
        user = input("👤 Tú: ")
        if user.lower() in ["salir", "exit", "quit"]:
            print("\n¡Hasta luego! 🎓")
            break
        
        if user.strip():
            respuesta = chat(user)
            print(f"🤖 Tutor: {respuesta}\n")
    except KeyboardInterrupt:
        print("\n\n¡Hasta luego! 🎓")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")