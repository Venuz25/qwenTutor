import subprocess
import tempfile
import os
import re
import json
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str
    execution_time: float
    language: str
    code: str


class CodeExecutor:
    """Ejecutor de código seguro para múltiples lenguajes"""
    
    def __init__(self, timeout: int = 5, max_output: int = 10000):
        self.timeout = timeout
        self.max_output = max_output
        self.allowed_languages = {
            "python": {"ext": ".py", "cmd": ["python", "-u"]},
            "javascript": {"ext": ".js", "cmd": ["node"]},
            "java": {"ext": ".java", "cmd": ["java"]},
            "cpp": {"ext": ".cpp", "cmd": ["g++", "-o"]},
            "c": {"ext": ".c", "cmd": ["gcc", "-o"]},
            "go": {"ext": ".go", "cmd": ["go", "run"]},
            "rust": {"ext": ".rs", "cmd": ["rustc", "-o"]},
        }
        self.blocked_imports = {
            "python": [
                "os", "sys", "subprocess", "socket", "requests",
                "urllib", "http", "ftplib", "smtplib", "pickle",
                "marshal", "shelve", "importlib", "__import__"
            ]
        }
    
    def extract_code_blocks(self, text: str, language: str = None) -> list:
        """Extrae bloques de código del texto"""
        if language:
            pattern = rf'```{language}\n(.*?)```'
            blocks = re.findall(pattern, text, re.DOTALL)
        else:
            blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        
        return [block.strip() for block in blocks if block.strip()]
    
    def is_safe_python_code(self, code: str) -> Tuple[bool, str]:
        """Verifica si el código Python es seguro"""
        blocked_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'import\s+socket',
            r'import\s+requests',
            r'import\s+urllib',
            r'os\.system',
            r'subprocess\.',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'__import__',
            r'open\s*\([^)]*[\'"][^\'"]*[\'"]',
            r'with\s+open',
        ]
        
        for pattern in blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Código bloqueado: patrón '{pattern}' detectado"
        
        if len(code) > 5000:
            return False, "Código demasiado largo"
        
        return True, "OK"
    
    def execute_python(self, code: str) -> ExecutionResult:
        """Ejecuta código Python de forma segura"""
        is_safe, message = self.is_safe_python_code(code)
        
        if not is_safe:
            return ExecutionResult(
                success=False,
                output="",
                error=message,
                execution_time=0,
                language="python",
                code=code
            )
        
        import time
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ["python", "-u", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            execution_time = time.time() - start_time
            
            output = result.stdout[:self.max_output]
            error = result.stderr[:self.max_output]
            
            os.unlink(temp_file)
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                execution_time=execution_time,
                language="python",
                code=code
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Timeout: Código excedió {self.timeout} segundos",
                execution_time=self.timeout,
                language="python",
                code=code
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0,
                language="python",
                code=code
            )
    
    def execute_javascript(self, code: str) -> ExecutionResult:
        """Ejecuta código JavaScript de forma segura"""
        import time
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.js', 
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ["node", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            execution_time = time.time() - start_time
            
            output = result.stdout[:self.max_output]
            error = result.stderr[:self.max_output]
            
            os.unlink(temp_file)
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                execution_time=execution_time,
                language="javascript",
                code=code
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Timeout: Código excedió {self.timeout} segundos",
                execution_time=self.timeout,
                language="javascript",
                code=code
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                output="",
                error="Node.js no está instalado en el sistema",
                execution_time=0,
                language="javascript",
                code=code
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0,
                language="javascript",
                code=code
            )
    
    def execute(self, code: str, language: str = "python") -> ExecutionResult:
        """Ejecuta código en el lenguaje especificado"""
        language = language.lower()
        
        if language not in self.allowed_languages:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Lenguaje '{language}' no soportado",
                execution_time=0,
                language=language,
                code=code
            )
        
        if language == "python":
            return self.execute_python(code)
        elif language == "javascript":
            return self.execute_javascript(code)
        else:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Ejecución para '{language}' no implementada aún",
                execution_time=0,
                language=language,
                code=code
            )
    
    def format_result(self, result: ExecutionResult) -> str:
        """Formatea el resultado para mostrar en el chat"""
        if result.success:
            output = f"""✅ **Ejecución Exitosa**

            **Tiempo:** {result.execution_time:.3f}s

            **Salida:**{result.output if result.output else '(sin salida)'}"""
        else:
            output = f"""❌ **Error de Ejecución**
            
            **Error:**{result.error}"""
        return output