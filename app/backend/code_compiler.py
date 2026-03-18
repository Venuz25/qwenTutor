import requests
import json
import subprocess
import tempfile
import os
import sys
import time

class CodeCompiler:
    """Compilador multi-lenguaje con fallback offline para Python."""
    
    PISTON_API_URL = "https://emkc.org/api/v2/piston"
    
    SUPPORTED_LANGUAGES = {
        'python': {'version': '3.10.0', 'name': 'python'},
        'java': {'version': '15.0.2', 'name': 'java'},
        'cpp': {'version': '10.2.0', 'name': 'cpp'},
        'javascript': {'version': '18.15.0', 'name': 'javascript'},
        'go': {'version': '1.16.2', 'name': 'go'},
        'rust': {'version': '1.68.2', 'name': 'rust'},
        'csharp': {'version': '6.12.0', 'name': 'csharp'},
        'c': {'version': '10.2.0', 'name': 'c'}
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'QwenTutor/1.0'
        })
    
    def get_supported_languages(self):
        return list(self.SUPPORTED_LANGUAGES.keys())
    
    def compile_code(self, code, language, stdin_input=""):
        if language not in self.SUPPORTED_LANGUAGES:
            return {
                'success': False,
                'output': '',
                'error': f'Lenguaje no soportado: {language}',
                'execution_time': 0
            }
        
        # Intentar API online primero
        result = self._compile_online(code, language, stdin_input)
        
        # Fallback offline para Python si falla conexión
        if language == 'python' and not result['success']:
            if 'conexión' in result['error'].lower() or 'Unauthorized' in result['error']:
                return self._compile_python_offline(code, stdin_input)
        
        return result
    
    def _compile_online(self, code, language, stdin_input):
        lang_config = self.SUPPORTED_LANGUAGES[language]
        payload = {
            'language': lang_config['name'],
            'version': lang_config['version'],
            'files': [{'content': code}],
            'stdin': stdin_input,
            'compile_timeout': 10000,
            'run_timeout': 10000
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.PISTON_API_URL}/execute",
                    json=payload,
                    timeout=15
                )
                
                if response.status_code == 401:
                    return {'success': False, 'output': '', 'error': 'API no autorizada', 'execution_time': 0}
                elif response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                elif response.status_code >= 500:
                    return {'success': False, 'output': '', 'error': f'Error del servidor: {response.status_code}', 'execution_time': 0}
                
                response.raise_for_status()
                result = response.json()
                
                if 'run' in result:
                    run = result['run']
                    return {
                        'success': run.get('code', 1) == 0,
                        'output': run.get('output', ''),
                        'error': run.get('stderr', ''),
                        'execution_time': result.get('time', 0)
                    }
                return {'success': False, 'output': '', 'error': result.get('message', 'Error'), 'execution_time': 0}
                
            except requests.exceptions.Timeout:
                return {'success': False, 'output': '', 'error': 'Tiempo agotado (10s)', 'execution_time': 0}
            except requests.exceptions.ConnectionError:
                if attempt == max_retries - 1:
                    return {'success': False, 'output': '', 'error': 'No se pudo conectar', 'execution_time': 0}
                time.sleep(2 ** attempt)
            except Exception as e:
                return {'success': False, 'output': '', 'error': f'Error: {str(e)}', 'execution_time': 0}
        
        return {'success': False, 'output': '', 'error': 'Máximo reintentos', 'execution_time': 0}
    
    def _compile_python_offline(self, code, stdin_input="", timeout=10):
        try:
            safe_code = self._sandbox_python_code(code)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(safe_code)
                temp_path = f.name
            
            result = subprocess.run(
                [sys.executable, temp_path],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': 0
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'output': '', 'error': f'Tiempo agotado ({timeout}s)', 'execution_time': timeout}
        except Exception as e:
            return {'success': False, 'output': '', 'error': f'Error local: {str(e)}', 'execution_time': 0}
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _sandbox_python_code(self, code):
        sandbox_header = '''
import builtins
_original_import = builtins.__import__
def _safe_import(name, *args, **kwargs):
    blocked = ['os','sys','subprocess','socket','requests','urllib','http','ftplib','smtplib','pickle','marshal','ctypes','importlib','__main__']
    if name in blocked or name.split('.')[0] in blocked:
        raise ImportError(f"Modulo '{name}' no permitido")
    return _original_import(name, *args, **kwargs)
builtins.__import__ = _safe_import
'''
        return sandbox_header + code
    
    def get_language_template(self, language):
        templates = {
            'python': 'def main():\n    print("Hola desde Python!")\n\nif __name__ == "__main__":\n    main()\n',
            'java': 'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hola desde Java!");\n    }\n}\n',
            'cpp': '#include <iostream>\nusing namespace std;\n\nint main() {\n    cout << "Hola desde C++!" << endl;\n    return 0;\n}\n',
            'javascript': 'console.log("Hola desde JavaScript!");\n',
            'go': 'package main\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hola desde Go!")\n}\n',
            'rust': 'fn main() {\n    println!("Hola desde Rust!");\n}\n',
            'csharp': 'using System;\n\nclass Program {\n    static void Main() {\n        Console.WriteLine("Hola desde C#!");\n    }\n}\n',
            'c': '#include <stdio.h>\n\nint main() {\n    printf("Hola desde C!\\n");\n    return 0;\n}\n'
        }
        return templates.get(language, templates['python'])