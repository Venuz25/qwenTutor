#!/usr/bin/env python3
"""
Tutor de Algoritmia - Interfaz Gráfica Mejorada
Ejecuta: python run_gui.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.chat_interface import ChatInterface


def main():
    print("=" * 60)
    print("TUTOR DE ALGORITMIA - Interfaz Grafica")
    print("=" * 60)
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "./models/qwen-algo-tutor-1.5b-v2"
    
    if not os.path.exists(adapter_path):
        print(f"Adapter no encontrado en {adapter_path}")
        print("   Usando modelo base sin fine-tuning")
        adapter_path = None
    
    print(f"\nModelo: {model_name}")
    print(f"Adapter: {adapter_path if adapter_path else 'Ninguno'}")
    print(f"\nIniciando interfaz...")
    print(f"   URL: http://localhost:7860")
    print(f"\nPresiona Ctrl+C para detener\n")
    
    chat = ChatInterface(
        model_name=model_name,
        adapter_path=adapter_path,
        history_dir="./chat_history"
    )
    
    try:
        chat.launch(share=False, server_port=7860)
    except KeyboardInterrupt:
        print("\n\nDeteniendo servidor...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()