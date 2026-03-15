# test_model.py (sin emojis en el código)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./models/qwen-algo-tutor-1.5b-v2"

# Configurar cuantización
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=bnb_config, 
    device_map="auto", 
    torch_dtype=torch.float32
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

preguntas_test = [
    # Preguntas que DEBERÍA saber (del dataset)
    "¿Qué es Big O?",
    "Implementa una pila en Python",
    
    # Preguntas que NO debería saber bien (nuevas)
    "¿Qué es un árbol rojo-negro?",
    "Explica el algoritmo de Floyd-Warshall",
    "¿Cuándo usar un árbol B+?",
]

print("Evaluando modelo con 141 ejemplos de entrenamiento...")
print("-" * 60)

for p in preguntas_test:
    print(f"\nPregunta: {p}")
    messages = [
        {"role": "system", "content": "Eres un tutor experto en Algoritmia..."},
        {"role": "user", "content": p}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(inputs)
    
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
    print(f"Respuesta: {respuesta[:400]}...")
    print("-" * 60)