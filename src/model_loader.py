import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional


class ModelLoader:
    """Carga y gestiona el modelo fine-tuneado"""
    
    def __init__(self, model_name: str, adapter_path: Optional[str] = None):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load(self):
        """Carga el modelo y tokenizer"""
        if self.is_loaded:
            return
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        
        if self.adapter_path:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        self.model.eval()
        self.is_loaded = True
    
    def generate(self, prompt: str, system_prompt: str, max_tokens: int = 512) -> str:
        """Genera una respuesta del modelo"""
        if not self.is_loaded:
            self.load()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(inputs)
        
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant\n")[-1].strip()
    
    def unload(self):
        """Libera memoria del modelo"""
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.is_loaded = False