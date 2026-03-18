#!/usr/bin/env python
import subprocess
import sys
import os
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    
    print("🚀 Iniciando Tutor de Algoritmia...")
    print(f"📁 Directorio: {project_root}")
    print("🔗 URL: http://localhost:8501")
    print("-" * 50)
    
    # Configurar PYTHONPATH explícitamente
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root) + ';' + str(project_root / "modules")
    
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run", "app/frontend/app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--theme.base", "dark"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Cerrado por usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()