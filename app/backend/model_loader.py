# app/backend/model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from app.config import MODEL_NAME, ADAPTER_PATH

class ModelLoader:
    """Carga y gestiona el modelo fine-tuneado."""
    
    def __init__(self, adapter_path=None):
        self.model_name = MODEL_NAME
        self.adapter_path = adapter_path or str(ADAPTER_PATH)
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def load_model(self):
        """Carga el modelo y tokenizer."""
        print(f"[ModelLoader] Cargando tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print("[ModelLoader] Configurando cuantización QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"[ModelLoader] Cargando modelo base: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        
        if self.adapter_path:
            print(f"[ModelLoader] Cargando adaptadores desde {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        self.model.eval()
        self.device = next(self.model.parameters()).device
        
        print("[ModelLoader] Modelo cargado exitosamente!")
        return self
    
    def generate_response(self, messages, max_tokens=512, temperature=0.7):
        """Genera respuesta del modelo."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Modelo no cargado. Llamá a load_model() primero.")
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(inputs)
        
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant\n")[-1].strip()