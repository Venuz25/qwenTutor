import re

class PromptFilter:
    """Filtra y transforma preguntas para enfoque pedagógico socrático."""
    
    DIRECT_ANSWER_PATTERNS = [
        r'dame la respuesta', r'resuelve esto', r'hacelo por mi',
        r'quiero la solucion', r'dame el codigo completo', r'escribime la funcion',
        r'mostrame como se hace', r'dame la implementacion', r'resuelve el problema',
        r'dame la respuesta correcta'
    ]
    
    TOPIC_KEYWORDS = {
        'big_o': ['big o', 'complejidad', 'orden', 'tiempo', 'espacio', 'o(n)', 'o(1)'],
        'pila': ['pila', 'stack', 'lifo', 'push', 'pop'],
        'cola': ['cola', 'queue', 'fifo', 'enqueue', 'dequeue'],
        'lista_enlazada': ['lista enlazada', 'linked list', 'nodo', 'siguiente'],
        'arbol': ['arbol', 'tree', 'bst', 'avl', 'nodo', 'raiz'],
        'grafo': ['grafo', 'graph', 'vertice', 'arista', 'bfs', 'dfs'],
        'ordenamiento': ['ordenamiento', 'sorting', 'bubble', 'quick', 'merge', 'heap'],
        'busqueda': ['busqueda', 'searching', 'binaria', 'lineal'],
        'programacion_dinamica': ['programacion dinamica', 'dynamic programming', 'dp', 'memoizacion'],
        'recursion': ['recursion', 'recursivo', 'caso base', 'llamada']
    }
    
    def __init__(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DIRECT_ANSWER_PATTERNS
        ]
    
    def is_direct_answer_request(self, prompt):
        return any(p.search(prompt) for p in self.compiled_patterns)
    
    def detect_topic(self, prompt):
        prompt_lower = prompt.lower()
        detected = []
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(k in prompt_lower for k in keywords):
                detected.append(topic)
        return detected if detected else ['general']
    
    def transform_to_socratic(self, prompt):
        if not self.is_direct_answer_request(prompt):
            return prompt
        
        topics = self.detect_topic(prompt)
        transformations = {
            'big_o': "En lugar de darte la respuesta, pensemos: ¿Qué operaciones se repiten? ¿Cómo crece el tiempo con más datos?",
            'pila': "Antes del código: ¿Qué principio sigue una pila? ¿Qué operación usarías para agregar?",
            'cola': "Guitemos: ¿En qué orden se atienden los elementos? ¿Qué estructura de Python podría funcionar?",
            'lista_enlazada': "Analicemos: ¿Qué necesita cada nodo para conectarse? ¿Cómo recorrerías la lista?",
            'arbol': "Pensemos: ¿Qué propiedad tiene un BST? ¿Dónde iría un valor menor que la raíz?",
            'grafo': "Reflexionemos: ¿Cómo representarías conexiones? ¿Qué algoritmo usarías para explorar?",
            'ordenamiento': "Antes del código: ¿Cómo compararías elementos? ¿Qué harías si están desordenados?",
            'busqueda': "Guitemos: ¿Cómo aprovecharías que está ordenado? ¿Qué podrías descartar?",
            'programacion_dinamica': "Pensemos: ¿Hay subproblemas que se repiten? ¿Podrías guardar resultados?",
            'recursion': "Analicemos: ¿Cuál sería el caso base? ¿Cómo se reduce el problema?",
            'general': "En lugar de la solución completa, te guiaré con preguntas. ¿Qué entendés del problema?"
        }
        return transformations.get(topics[0] if topics else 'general', transformations['general'])
    
    def add_hints(self, prompt, topic):
        hints = {
            'big_o': "Pista: Contá los bucles anidados. Cada bucle sobre n suma un factor de n.",
            'pila': "Pista: Python tiene una estructura built-in que funciona como pila. ¿Cuál?",
            'cola': "Pista: Para una cola eficiente, buscá en collections.deque.",
            'lista_enlazada': "Pista: Cada nodo necesita: su valor y referencia al siguiente.",
            'arbol': "Pista: La búsqueda en BST es como búsqueda binaria pero en árbol.",
            'grafo': "Pista: BFS usa cola, DFS usa pila o recursión.",
            'ordenamiento': "Pista: Compará adyacentes e intercambiá si están mal ordenados.",
            'busqueda': "Pista: Dividí el espacio a la mitad en cada paso.",
            'programacion_dinamica': "Pista: Creá una tabla para guardar resultados de subproblemas.",
            'recursion': "Pista: Todo recursivo necesita un caso base que detenga las llamadas.",
            'general': "Pista: Descomponé el problema en pasos pequeños."
        }
        return prompt + "\n\n" + hints.get(topic, hints['general'])
    
    def process_prompt(self, prompt):
        metadata = {
            'is_direct_request': self.is_direct_answer_request(prompt),
            'topics': self.detect_topic(prompt),
            'original_prompt': prompt
        }
        
        if metadata['is_direct_request']:
            processed = self.transform_to_socratic(prompt)
            processed = self.add_hints(processed, metadata['topics'][0])
            metadata['transformed'] = True
        else:
            processed = prompt
            metadata['transformed'] = False
        
        return processed, metadata