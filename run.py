#!/usr/bin/env python
"""Script principal para iniciar la aplicación."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Obtener ruta ABSOLUTA del proyecto
    project_root = Path(__file__).resolve().parent.absolute()
    
    # Cambiar al directorio raíz
    os.chdir(project_root)
    
    print("🚀 Iniciando Tutor de Algoritmia...")
    print(f"📁 Directorio: {project_root}")
    print("🔗 URL: http://localhost:8501")
    print("-" * 50)
    
    # Configurar PYTHONPATH EXPLÍCITAMENTE para el subprocess
    env = os.environ.copy()
    
    # PYTHONPATH debe ser el project_root (donde está la carpeta 'app')
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        env['PYTHONPATH'] = f"{project_root};{current_pythonpath}"
    else:
        env['PYTHONPATH'] = str(project_root)
    
    print(f"[INFO] PYTHONPATH configurado: {env['PYTHONPATH']}")
    
    # Comando para Streamlit
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run", 
        str(project_root / "app" / "frontend" / "app.py"),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--theme.base", "dark",
        "--logger.level", "info"
    ]
    
    try:
        # Ejecutar con el entorno configurado
        result = subprocess.run(cmd, env=env, cwd=project_root)
        if result.returncode != 0:
            print(f"❌ Streamlit terminó con código {result.returncode}")
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada por el usuario")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()