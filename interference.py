import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Rutas
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./models/qwen-algoritmia-tutor"

print("Cargando modelo base...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("Cargando adaptadores LoRA...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

def chat(usuario_input):
    messages = [
        {"role": "system", "content": "Eres un tutor experto en Algoritmia y Estructuras de Datos."},
        {"role": "user", "content": usuario_input}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Limpiar la respuesta para mostrar solo lo nuevo
    return respuesta.split("assistant\n")[-1].strip()

# Bucle de prueba
print("\n--- Tutor de Algoritmia Listo (Escribe 'salir' para terminar) ---")
while True:
    user = input("\nTú: ")
    if user.lower() in ["salir", "exit"]:
        break
    response = chat(user)
    print(f"Tutor: {response}")