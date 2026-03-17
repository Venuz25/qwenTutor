import re
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class IntentType(Enum):
    EXPLAIN = "explain"           # Explicar conceptos
    CODE = "code"                 # Implementar código
    PROBLEM = "problem"           # Proponer problema
    FEEDBACK = "feedback"         # Revisar código del usuario
    DEBUG = "debug"               # Debuggear código
    COMPARE = "compare"           # Comparar algoritmos
    COMPLEXITY = "complexity"     # Analizar complejidad
    GENERAL = "general"           # Pregunta general


@dataclass
class UserIntent:
    intent: IntentType
    confidence: float
    language: str
    topics: List[str]
    code_snippet: Optional[str]
    original_prompt: str


class StateMachine:
    """Máquina de estados para filtrar y categorizar prompts del usuario"""
    
    def __init__(self):
        self.current_state = "IDLE"
        self.conversation_history = []
        self.intent_patterns = self._load_intent_patterns()
        self.language_patterns = self._load_language_patterns()
        self.topic_patterns = self._load_topic_patterns()
        self.blocked_patterns = self._load_blocked_patterns()
    
    def _load_intent_patterns(self) -> Dict[IntentType, List[str]]:
        return {
            IntentType.EXPLAIN: [
                r"¿qué es", r"explica", r"definición", r"describe",
                r"qué significa", r"para qué sirve", r"concepto"
            ],
            IntentType.CODE: [
                r"implementa", r"código", r"función", r"clase",
                r"escribe.*python", r"escribe.*java", r"escribe.*cpp",
                r"crea.*función", r"haz.*código"
            ],
            IntentType.PROBLEM: [
                r"problema", r"ejercicio", r"práctica", r"reto",
                r"desafío", r"propon.*ejercicio"
            ],
            IntentType.FEEDBACK: [
                r"revisa", r"está bien", r"correcto", r"mi código",
                r"mi solución", r"evalúa", r"feedback"
            ],
            IntentType.DEBUG: [
                r"error", r"no funciona", r"bug", r"falla",
                r"por qué.*error", r"debug"
            ],
            IntentType.COMPARE: [
                r"diferencia", r"vs", r"versus", r"compara",
                r"cuál es mejor", r"ventajas"
            ],
            IntentType.COMPLEXITY: [
                r"complejidad", r"big o", r"o\(", r"tiempo.*espacio",
                r"eficiencia", r"rendimiento"
            ],
        }
    
    def _load_language_patterns(self) -> Dict[str, List[str]]:
        return {
            "python": [r"python", r"py", r"\.py"],
            "java": [r"java", r"\.java"],
            "cpp": [r"c\+\+", r"cpp", r"c\+\+", r"\.cpp", r"\.hpp"],
            "c": [r"\bc\b", r"\.c", r"\.h"],
            "javascript": [r"javascript", r"js", r"\.js", r"node"],
            "go": [r"golang", r"go ", r"\.go"],
            "rust": [r"rust", r"\.rs", r"cargo"],
            "csharp": [r"c#", r"csharp", r"\.cs"],
        }
    
    def _load_topic_patterns(self) -> Dict[str, List[str]]:
        return {
            "arrays": [r"array", r"lista", r"vector"],
            "linked_list": [r"lista enlazada", r"linked list", r"nodo"],
            "stack": [r"pila", r"stack", r"lifo"],
            "queue": [r"cola", r"queue", r"fifo"],
            "tree": [r"árbol", r"tree", r"bst", r"avl"],
            "graph": [r"grafo", r"graph", r"vértice", r"arista"],
            "sorting": [r"ordenamiento", r"sorting", r"sort"],
            "searching": [r"búsqueda", r"searching", r"search"],
            "dynamic_programming": [r"programación dinámica", r"dynamic programming", r"dp"],
            "greedy": [r"greedy", r"voraz"],
            "backtracking": [r"backtracking", r"vuelta atrás"],
            "hash": [r"hash", r"tabla hash", r"diccionario"],
            "heap": [r"heap", r"montículo", r"priority queue"],
        }
    
    def _load_blocked_patterns(self) -> List[str]:
        return [
            r"ignore.*instrucciones",
            r"omit.*restrictions",
            r"bypass.*filter",
            r"execute.*malicious",
            r"delete.*file",
            r"drop.*table",
            r"rm -rf",
            r"sudo.*rm",
            r"import.*os.*system",
            r"subprocess.*call",
            r"eval\(",
            r"exec\(",
        ]
    
    def is_safe_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Verifica si el prompt es seguro"""
        prompt_lower = prompt.lower()
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return False, f"Prompt bloqueado por seguridad: patrón '{pattern}' detectado"
        
        if len(prompt) > 2000:
            return False, "Prompt demasiado largo (máx 2000 caracteres)"
        
        return True, "OK"
    
    def detect_language(self, prompt: str) -> str:
        """Detecta el lenguaje de programación mencionado"""
        prompt_lower = prompt.lower()
        
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return lang
        
        return "python"  # Default
    
    def detect_topics(self, prompt: str) -> List[str]:
        """Detecta los temas mencionados"""
        prompt_lower = prompt.lower()
        topics = []
        
        for topic, patterns in self.topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    topics.append(topic)
                    break
        
        return topics if topics else ["general"]
    
    def extract_code(self, prompt: str) -> Optional[str]:
        """Extrae código del prompt del usuario"""
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', prompt, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        inline_code = re.findall(r'`([^`]+)`', prompt)
        if inline_code and len(inline_code[0]) > 20:
            return inline_code[0]
        
        return None
    
    def classify_intent(self, prompt: str) -> IntentType:
        """Clasifica la intención del usuario"""
        prompt_lower = prompt.lower()
        scores = {intent: 0 for intent in IntentType}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    scores[intent] += 1
        
        max_intent = max(scores, key=scores.get)
        
        if scores[max_intent] == 0:
            return IntentType.GENERAL
        
        return max_intent
    
    def calculate_confidence(self, intent: IntentType, prompt: str) -> float:
        """Calcula la confianza de la clasificación"""
        prompt_lower = prompt.lower()
        matches = 0
        
        for pattern in self.intent_patterns.get(intent, []):
            if re.search(pattern, prompt_lower):
                matches += 1
        
        if matches == 0:
            return 0.3
        elif matches == 1:
            return 0.6
        else:
            return min(0.95, 0.6 + (matches * 0.1))
    
    def process_prompt(self, prompt: str) -> UserIntent:
        """Procesa el prompt y devuelve la intención detectada"""
        is_safe, message = self.is_safe_prompt(prompt)
        
        if not is_safe:
            return UserIntent(
                intent=IntentType.GENERAL,
                confidence=1.0,
                language="python",
                topics=["security"],
                code_snippet=None,
                original_prompt=f"[BLOCKED] {message}"
            )
        
        intent = self.classify_intent(prompt)
        confidence = self.calculate_confidence(intent, prompt)
        language = self.detect_language(prompt)
        topics = self.detect_topics(prompt)
        code = self.extract_code(prompt)
        
        user_intent = UserIntent(
            intent=intent,
            confidence=confidence,
            language=language,
            topics=topics,
            code_snippet=code,
            original_prompt=prompt
        )
        
        self.conversation_history.append({
            "state": self.current_state,
            "intent": intent.value,
            "prompt": prompt
        })
        
        return user_intent
    
    def get_system_prompt(self, intent: UserIntent) -> str:
        """Genera el system prompt según la intención detectada"""
        base_prompt = "Eres un tutor experto en Algoritmia, Estructuras de Datos y Programación Competitiva."
        
        prompts = {
            IntentType.EXPLAIN: f"""{base_prompt}
Tu objetivo es EXPLICAR conceptos de forma clara y pedagógica.
- Usa ejemplos concretos
- Incluye analogías cuando sea útil
- Explica paso a paso
- Menciona complejidad temporal y espacial
- Lenguaje preferido: {intent.language}
- Temas: {', '.join(intent.topics)}""",
            
            IntentType.CODE: f"""{base_prompt}
Tu objetivo es PROPORCIONAR CÓDIGO completo y funcional.
- Código completo, no fragmentos
- Incluye comentarios explicativos
- Manejo de errores adecuado
- Casos de prueba
- Complejidad temporal y espacial
- Lenguaje: {intent.language}""",
            
            IntentType.PROBLEM: f"""{base_prompt}
Tu objetivo es PROPONER problemas de práctica.
- Enunciado claro
- Casos de prueba
- Dificultad escalonada
- Pistas opcionales
- Solución guiada
- Tema: {', '.join(intent.topics)}""",
            
            IntentType.FEEDBACK: f"""{base_prompt}
Tu objetivo es REVISAR código de estudiantes.
- Feedback constructivo
- Señala errores específicos
- Sugiere mejoras
- Explica por qué
- Proporciona código corregido
- Lenguaje: {intent.language}""",
            
            IntentType.DEBUG: f"""{base_prompt}
Tu objetivo es AYUDAR A DEBUGGEAR código.
- Identifica el error
- Explica la causa
- Proporciona solución
- Prevención futura
- Lenguaje: {intent.language}""",
            
            IntentType.COMPARE: f"""{base_prompt}
Tu objetivo es COMPARAR algoritmos o estructuras.
- Tabla comparativa
- Ventajas y desventajas
- Casos de uso de cada uno
- Complejidad comparada
- Recomendación""",
            
            IntentType.COMPLEXITY: f"""{base_prompt}
Tu objetivo es ANALIZAR complejidad algorítmica.
- Tiempo y espacio
- Mejor, promedio, peor caso
- Explicación del análisis
- Optimizaciones posibles""",
            
            IntentType.GENERAL: f"""{base_prompt}
Responde de forma útil y educativa.
- Sé claro y conciso
- Incluye ejemplos cuando sea relevante
- Menciona complejidad si aplica
- Lenguaje: {intent.language}""",
        }
        
        return prompts.get(intent.intent, prompts[IntentType.GENERAL])
    
    def get_state_summary(self) -> Dict:
        """Resume el estado actual de la conversación"""
        return {
            "current_state": self.current_state,
            "total_messages": len(self.conversation_history),
            "intents_detected": list(set(
                msg["intent"] for msg in self.conversation_history
            )),
            "languages_used": list(set(
                self.detect_language(msg["prompt"]) 
                for msg in self.conversation_history
            ))
        }