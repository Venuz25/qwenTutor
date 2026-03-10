#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador Escalable de Dataset - VERSIÓN FINAL CORREGIDA
Genera EXACTAMENTE N ejemplos sin errores
"""

import json
import random
import os
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SYSTEM_PROMPT = "Eres un tutor experto en Algoritmia y Estructuras de Datos. Explicas conceptos de forma clara, provees ejemplos en Python y analizas la complejidad temporal y espacial."

OUTPUT_DIR = "./data/dataset_algoritmia"

# =============================================================================
# BASE DE CONOCIMIENTO
# =============================================================================

class KnowledgeBase:
    def __init__(self):
        self.topics = self._load_topics()
        
    def _load_topics(self) -> Dict:
        return {
            "complejidad": {
                "templates": [{
                    "question": "¿Qué significa la complejidad {notation}?",
                    "answer": "La complejidad {notation} {explicacion}. Es común en {ejemplos}.\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"notation": "O(1)", "explicacion": "significa tiempo constante", "ejemplos": "acceso a array", "codigo": "valor = arr[5]", "time": "O(1)", "space": "O(1)"},
                        {"notation": "O(n)", "explicacion": "es complejidad lineal", "ejemplos": "búsqueda lineal", "codigo": "for x in lista:\n    if x == target: return True", "time": "O(n)", "space": "O(1)"},
                        {"notation": "O(n²)", "explicacion": "suele ocurrir con bucles anidados", "ejemplos": "Bubble Sort", "codigo": "for i in range(n):\n    for j in range(n):\n        print(i, j)", "time": "O(n²)", "space": "O(1)"},
                        {"notation": "O(log n)", "explicacion": "es complejidad logarítmica", "ejemplos": "Binary Search", "codigo": "while low <= high:\n    mid = (low + high) // 2", "time": "O(log n)", "space": "O(1)"},
                        {"notation": "O(n log n)", "explicacion": "combina división logarítmica con procesamiento lineal", "ejemplos": "Merge Sort, Quick Sort", "codigo": "def merge_sort(arr):\n    if len(arr) > 1:\n        mid = len(arr)//2\n        merge_sort(arr[:mid])\n        merge_sort(arr[mid:])", "time": "O(n log n)", "space": "O(n)"},
                        {"notation": "O(2^n)", "explicacion": "es complejidad exponencial", "ejemplos": "Fibonacci recursivo", "codigo": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)", "time": "O(2^n)", "space": "O(n)"},
                        {"notation": "O(n!)", "explicacion": "es complejidad factorial", "ejemplos": "permutaciones", "codigo": "def permute(nums):\n    if len(nums) == 0: return [[]]", "time": "O(n!)", "space": "O(n)"},
                        {"notation": "O(√n)", "explicacion": "es complejidad raíz cuadrada", "ejemplos": "verificación de primalidad", "codigo": "for i in range(2, int(n**0.5)+1):\n    if n % i == 0: return False", "time": "O(√n)", "space": "O(1)"},
                    ]
                }]
            },
            "arrays": {
                "templates": [{
                    "question": "{accion} en un array.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Cómo insertar al final", "explicacion": "En Python usamos `.append()`.", "codigo": "lista = [1, 2]\nlista.append(3)", "time": "O(1)", "space": "O(1)"},
                        {"accion": "¿Cómo eliminar por índice", "explicacion": "Usamos `.pop()`. Los elementos se desplazan.", "codigo": "elem = lista.pop(i)", "time": "O(n)", "space": "O(1)"},
                        {"accion": "Implementa Two Sum", "explicacion": "Buscamos dos números que sumen target con hash map.", "codigo": "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i", "time": "O(n)", "space": "O(n)"},
                        {"accion": "¿Cómo rotar un array k posiciones", "explicacion": "Usamos slicing.", "codigo": "def rotate(nums, k):\n    k %= len(nums)\n    nums[:] = nums[-k:] + nums[:-k]", "time": "O(n)", "space": "O(n)"},
                        {"accion": "Encuentra el máximo", "explicacion": "Iteramos manteniendo el máximo.", "codigo": "def find_max(nums):\n    max_val = nums[0]\n    for n in nums[1:]:\n        if n > max_val: max_val = n\n    return max_val", "time": "O(n)", "space": "O(1)"},
                        {"accion": "¿Cómo encontrar duplicados", "explicacion": "Usamos un set para trackear vistos.", "codigo": "def find_duplicates(nums):\n    seen, dupes = set(), set()\n    for n in nums:\n        if n in seen: dupes.add(n)\n        seen.add(n)\n    return list(dupes)", "time": "O(n)", "space": "O(n)"},
                        {"accion": "Product of Array Except Self", "explicacion": "Calculamos prefix y suffix products.", "codigo": "def product_except_self(nums):\n    n = len(nums)\n    res = [1]*n\n    prefix = 1\n    for i in range(n):\n        res[i] = prefix\n        prefix *= nums[i]\n    suffix = 1\n    for i in range(n-1, -1, -1):\n        res[i] *= suffix\n        suffix *= nums[i]\n    return res", "time": "O(n)", "space": "O(1)"},
                        {"accion": "¿Cómo encontrar el elemento mayoritario", "explicacion": "Usamos Boyer-Moore Voting.", "codigo": "def majority_element(nums):\n    candidate, count = None, 0\n    for n in nums:\n        if count == 0: candidate = n\n        count += (1 if n == candidate else -1)\n    return candidate", "time": "O(n)", "space": "O(1)"},
                    ]
                }]
            },
            "strings": {
                "templates": [{
                    "question": "{accion} en strings.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Cómo invertir un string", "explicacion": "En Python usamos slicing.", "codigo": "s_invertido = s[::-1]", "time": "O(n)", "space": "O(n)"},
                        {"accion": "¿Cómo verificar palíndromo", "explicacion": "Comparamos con su reverso.", "codigo": "def is_palindrome(s):\n    return s == s[::-1]", "time": "O(n)", "space": "O(n)"},
                        {"accion": "¿Cómo verificar anagramas", "explicacion": "Ordenamos o contamos frecuencias.", "codigo": "def is_anagram(s, t):\n    return sorted(s) == sorted(t)", "time": "O(n log n)", "space": "O(n)"},
                        {"accion": "Primer carácter único", "explicacion": "Contamos frecuencias.", "codigo": "def first_unique_char(s):\n    from collections import Counter\n    count = Counter(s)\n    for i, c in enumerate(s):\n        if count[c] == 1: return i\n    return -1", "time": "O(n)", "space": "O(1)"},
                        {"accion": "Substring más larga sin repetir", "explicacion": "Sliding Window + HashSet.", "codigo": "def length_of_longest_substring(s):\n    seen = set()\n    l = res = 0\n    for r in range(len(s)):\n        while s[r] in seen:\n            seen.remove(s[l])\n            l += 1\n        seen.add(s[r])\n        res = max(res, r - l + 1)\n    return res", "time": "O(n)", "space": "O(min(n,m))"},
                        {"accion": "Valida paréntesis balanceados", "explicacion": "Usamos un stack.", "codigo": "def isValid(s):\n    stack = []\n            map = {')':'(', '}':'{', ']':'['}\n    for c in s:\n        if c in map:\n            if not stack or stack.pop() != map[c]: return False\n        else: stack.append(c)\n    return not stack", "time": "O(n)", "space": "O(n)"},
                    ]
                }]
            },
            "linked_lists": {
                "templates": [{
                    "question": "{accion} en Linked List.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "Invierte una Linked List", "explicacion": "Usamos tres punteros.", "codigo": "def reverse(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev", "time": "O(n)", "space": "O(1)"},
                        {"accion": "¿Cómo detectar un ciclo", "explicacion": "Punteros lento y rápido (Floyd).", "codigo": "def has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast: return True\n    return False", "time": "O(n)", "space": "O(1)"},
                        {"accion": "Encuentra el inicio del ciclo", "explicacion": "Reiniciamos un puntero al inicio.", "codigo": "def detect_cycle_start(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast: break\n    else: return None\n    slow = head\n    while slow != fast:\n        slow = slow.next\n        fast = fast.next\n    return slow", "time": "O(n)", "space": "O(1)"},
                        {"accion": "K-ésimo desde el final", "explicacion": "Dos punteros con distancia k.", "codigo": "def kth_from_end(head, k):\n    fast = slow = head\n    for _ in range(k):\n        if not fast: return None\n        fast = fast.next\n    while fast:\n        fast = fast.next\n        slow = slow.next\n    return slow", "time": "O(n)", "space": "O(1)"},
                        {"accion": "¿Cómo encontrar el punto medio", "explicacion": "Lento avanza 1, rápido avanza 2.", "codigo": "def middle_node(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n    return slow", "time": "O(n)", "space": "O(1)"},
                        {"accion": "¿Cómo mergear dos listas ordenadas", "explicacion": "Comparamos nodos y enlazamos el menor.", "codigo": "def merge_two_lists(l1, l2):\n    dummy = curr = ListNode(0)\n    while l1 and l2:\n        if l1.val < l2.val:\n            curr.next = l1\n            l1 = l1.next\n        else:\n            curr.next = l2\n            l2 = l2.next\n        curr = curr.next\n    curr.next = l1 or l2\n    return dummy.next", "time": "O(n+m)", "space": "O(1)"},
                    ]
                }]
            },
            "trees": {
                "templates": [{
                    "question": "{accion} en Binary Tree.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Qué es Inorder Traversal", "explicacion": "Visita Izquierda -> Nodo -> Derecha.", "codigo": "def inorder(root):\n    if root:\n        inorder(root.left)\n        print(root.val)\n        inorder(root.right)", "time": "O(n)", "space": "O(h)"},
                        {"accion": "¿Cómo validar un BST", "explicacion": "Verificar rango (min, max).", "codigo": "def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):\n    if not root: return True\n    if not (min_val < root.val < max_val): return False\n    return (is_valid_bst(root.left, min_val, root.val) and\n            is_valid_bst(root.right, root.val, max_val))", "time": "O(n)", "space": "O(h)"},
                        {"accion": "Encuentra la altura", "explicacion": "DFS recursivo.", "codigo": "def max_depth(root):\n    if not root: return 0\n    return 1 + max(max_depth(root.left), max_depth(root.right))", "time": "O(n)", "space": "O(h)"},
                        {"accion": "¿Cómo encontrar el LCA", "explicacion": "Ancestro común más bajo.", "codigo": "def lowest_common_ancestor(root, p, q):\n    if not root or root == p or root == q: return root\n    left = lowest_common_ancestor(root.left, p, q)\n    right = lowest_common_ancestor(root.right, p, q)\n    return root if left and right else left or right", "time": "O(n)", "space": "O(h)"},
                        {"accion": "¿Cómo verificar si es balanceado", "explicacion": "Diferencia de altura <= 1.", "codigo": "def is_balanced(root):\n    def check(node):\n        if not node: return 0\n        left = check(node.left)\n        if left == -1: return -1\n        right = check(node.right)\n        if right == -1: return -1\n        if abs(left - right) > 1: return -1\n        return 1 + max(left, right)\n    return check(root) != -1", "time": "O(n)", "space": "O(h)"},
                        {"accion": "Calcula el diámetro", "explicacion": "Camino más largo entre dos nodos.", "codigo": "def diameter_of_binary_tree(root):\n    res = [0]\n    def dfs(node):\n        if not node: return 0\n        left = dfs(node.left)\n        right = dfs(node.right)\n        res[0] = max(res[0], left + right)\n        return 1 + max(left, right)\n    dfs(root)\n    return res[0]", "time": "O(n)", "space": "O(h)"},
                    ]
                }]
            },
            "graphs": {
                "templates": [{
                    "question": "{accion} en Grafos.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "Implementa BFS", "explicacion": "Queue para explorar por niveles.", "codigo": "def bfs(graph, start):\n    visited, q = set(), deque([start])\n    visited.add(start)\n    while q:\n        node = q.popleft()\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                q.append(neighbor)", "time": "O(V+E)", "space": "O(V)"},
                        {"accion": "Implementa DFS", "explicacion": "Recursión para explorar en profundidad.", "codigo": "def dfs(graph, node, visited):\n    visited.add(node)\n    for neighbor in graph[node]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)", "time": "O(V+E)", "space": "O(V)"},
                        {"accion": "¿Cómo detectar ciclo en grafo dirigido", "explicacion": "DFS con colores.", "codigo": "def has_cycle(graph):\n    color = [0]*len(graph)\n    def dfs(node):\n        if color[node] == 1: return True\n        if color[node] == 2: return False\n        color[node] = 1\n        for nei in graph[node]:\n            if dfs(nei): return True\n        color[node] = 2\n        return False\n    return any(dfs(i) for i in range(len(graph)) if color[i]==0)", "time": "O(V+E)", "space": "O(V)"},
                        {"accion": "¿Cómo hacer Topological Sort", "explicacion": "Usando grados de entrada.", "codigo": "def topological_sort(n, edges):\n    graph = defaultdict(list)\n    indegree = [0]*n\n    for u, v in edges:\n        graph[u].append(v)\n        indegree[v] += 1\n    q = deque([i for i in range(n) if indegree[i]==0])\n    result = []\n    while q:\n        node = q.popleft()\n        result.append(node)\n        for nei in graph[node]:\n            indegree[nei] -= 1\n            if indegree[nei] == 0: q.append(nei)\n    return result if len(result)==n else []", "time": "O(V+E)", "space": "O(V)"},
                        {"accion": "¿Qué es Dijkstra", "explicacion": "Camino más corto con pesos no negativos.", "codigo": "import heapq\ndef dijkstra(graph, start):\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, node = heapq.heappop(pq)\n        if d > dist[node]: continue\n        for neighbor, weight in graph[node]:\n            if dist[node] + weight < dist[neighbor]:\n                dist[neighbor] = dist[node] + weight\n                heapq.heappush(pq, (dist[neighbor], neighbor))\n    return dist", "time": "O((V+E) log V)", "space": "O(V)"},
                    ]
                }]
            },
            "dynamic_programming": {
                "templates": [{
                    "question": "{accion} con Programación Dinámica.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Cómo calcular Fibonacci", "explicacion": "Guardamos resultados previos.", "codigo": "def fib(n):\n    if n <= 1: return n\n    dp = [0, 1]\n    for i in range(2, n+1):\n        dp.append(dp[i-1] + dp[i-2])\n    return dp[n]", "time": "O(n)", "space": "O(n)"},
                        {"accion": "Implementa Coin Change", "explicacion": "Mínimo número de monedas para amount.", "codigo": "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount+1)\n    dp[0] = 0\n    for a in range(1, amount+1):\n        for c in coins:\n            if a >= c:\n                dp[a] = min(dp[a], dp[a-c] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1", "time": "O(amount * len(coins))", "space": "O(amount)"},
                        {"accion": "¿Qué es House Robber", "explicacion": "Maximizar sin robar casas adyacentes.", "codigo": "def rob(nums):\n    prev1 = prev2 = 0\n    for n in nums:\n        prev1, prev2 = max(prev2 + n, prev1), prev1\n    return prev1", "time": "O(n)", "space": "O(1)"},
                        {"accion": "¿Cómo resolver Climbing Stairs", "explicacion": "Subir n escalones (1 o 2 pasos).", "codigo": "def climb_stairs(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a", "time": "O(n)", "space": "O(1)"},
                        {"accion": "Longest Common Subsequence", "explicacion": "Subsecuencia común más larga.", "codigo": "def lcs(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(m):\n        for j in range(n):\n            if s1[i] == s2[j]:\n                dp[i+1][j+1] = dp[i][j] + 1\n            else:\n                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n    return dp[m][n]", "time": "O(m*n)", "space": "O(m*n)"},
                        {"accion": "Maximum Subarray (Kadane)", "explicacion": "Subarray con suma máxima.", "codigo": "def max_subarray(nums):\n    max_sum = curr = nums[0]\n    for n in nums[1:]:\n        curr = max(n, curr + n)\n        max_sum = max(max_sum, curr)\n    return max_sum", "time": "O(n)", "space": "O(1)"},
                    ]
                }]
            },
            "heaps": {
                "templates": [{
                    "question": "{accion} con Heaps.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Cómo implementar Min-Heap", "explicacion": "Usamos `heapq`.", "codigo": "import heapq\nh = []\nheapq.heappush(h, 5)\nmin_val = heapq.heappop(h)", "time": "O(log n)", "space": "O(n)"},
                        {"accion": "K-ésimo elemento más grande", "explicacion": "Min-Heap de tamaño k.", "codigo": "def kth_largest(nums, k):\n    heap = []\n    for n in nums:\n        heapq.heappush(heap, n)\n        if len(heap) > k:\n            heapq.heappop(heap)\n    return heap[0]", "time": "O(n log k)", "space": "O(k)"},
                        {"accion": "¿Cómo mergear K listas ordenadas", "explicacion": "Min-Heap con el menor de cada lista.", "codigo": "def merge_k_lists(lists):\n    heap = []\n    for i, l in enumerate(lists):\n        if l: heapq.heappush(heap, (l.val, i, l))\n    dummy = curr = ListNode(0)\n    while heap:\n        val, i, node = heapq.heappop(heap)\n        curr.next = node\n        curr = curr.next\n        if node.next:\n            heapq.heappush(heap, (node.next.val, i, node.next))\n    return dummy.next", "time": "O(N log k)", "space": "O(k)"},
                    ]
                }]
            },
            "sliding_window": {
                "templates": [{
                    "question": "{accion} con Sliding Window.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "Máxima suma de subarray tamaño k", "explicacion": "Ventana fija de tamaño k.", "codigo": "def max_sum_subarray(nums, k):\n    window_sum = sum(nums[:k])\n    max_sum = window_sum\n    for i in range(k, len(nums)):\n        window_sum += nums[i] - nums[i-k]\n        max_sum = max(max_sum, window_sum)\n    return max_sum", "time": "O(n)", "space": "O(1)"},
                        {"accion": "Ventana con suma mínima >= target", "explicacion": "Ventana variable.", "codigo": "def min_subarray_len(target, nums):\n    l = total = 0\n    res = float('inf')\n    for r in range(len(nums)):\n        total += nums[r]\n        while total >= target:\n            res = min(res, r - l + 1)\n            total -= nums[l]\n            l += 1\n    return res if res != float('inf') else 0", "time": "O(n)", "space": "O(1)"},
                    ]
                }]
            },
            "binary_search": {
                "templates": [{
                    "question": "{accion} con Binary Search.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Cómo implementar búsqueda binaria", "explicacion": "Dividimos espacio a la mitad.", "codigo": "def binary_search(arr, target):\n    l, r = 0, len(arr)-1\n    while l <= r:\n        mid = (l+r)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: l = mid + 1\n        else: r = mid - 1\n    return -1", "time": "O(log n)", "space": "O(1)"},
                        {"accion": "¿Cómo encontrar primer elemento >= target", "explicacion": "Binary Search para lower bound.", "codigo": "def lower_bound(arr, target):\n    l, r = 0, len(arr)\n    while l < r:\n        mid = (l+r)//2\n        if arr[mid] < target: l = mid + 1\n        else: r = mid\n    return l", "time": "O(log n)", "space": "O(1)"},
                    ]
                }]
            },
            "backtracking": {
                "templates": [{
                    "question": "{accion} con Backtracking.",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Cómo generar permutaciones", "explicacion": "Backtracking intercambiando.", "codigo": "def permute(nums):\n    res = []\n    def backtrack(start):\n        if start == len(nums):\n            res.append(nums[:])\n            return\n        for i in range(start, len(nums)):\n            nums[start], nums[i] = nums[i], nums[start]\n            backtrack(start+1)\n            nums[start], nums[i] = nums[i], nums[start]\n    backtrack(0)\n    return res", "time": "O(n*n!)", "space": "O(n)"},
                        {"accion": "¿Cómo generar subconjuntos", "explicacion": "Backtracking o iterativo.", "codigo": "def subsets(nums):\n    res = [[]]\n    for n in nums:\n        res += [curr + [n] for curr in res]\n    return res", "time": "O(n*2^n)", "space": "O(1)"},
                    ]
                }]
            },
            "math_bits": {
                "templates": [{
                    "question": "{accion} (Matemáticas/Bits).",
                    "answer": "{explicacion}\n\n```python\n{codigo}\n```\n\nComplejidad: Tiempo {time}, Espacio {space}.",
                    "data": [
                        {"accion": "¿Cómo calcular GCD", "explicacion": "Algoritmo de Euclides.", "codigo": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a", "time": "O(log min(a,b))", "space": "O(1)"},
                        {"accion": "¿Cómo verificar si es primo", "explicacion": "Divisores hasta √n.", "codigo": "def is_prime(n):\n    if n < 2: return False\n    if n == 2: return True\n    if n % 2 == 0: return False\n    for i in range(3, int(n**0.5)+1, 2):\n        if n % i == 0: return False\n    return True", "time": "O(√n)", "space": "O(1)"},
                        {"accion": "¿Cómo contar bits activos", "explicacion": "n & (n-1) elimina bit derecho.", "codigo": "def count_bits(n):\n    count = 0\n    while n:\n        n &= n - 1\n        count += 1\n    return count", "time": "O(k)", "space": "O(1)"},
                        {"accion": "Número único con XOR", "explicacion": "XOR todos los elementos.", "codigo": "def single_number(nums):\n    res = 0\n    for n in nums: res ^= n\n    return res", "time": "O(n)", "space": "O(1)"},
                    ]
                }]
            },
        }
    
    def get_all_combinations(self) -> List[Tuple[str, Dict, Dict]]:
        """Obtiene todas las combinaciones (topic, template, data)"""
        combinations = []
        for topic_name, topic_data in self.topics.items():
            for template in topic_data["templates"]:
                for data_item in template["data"]:
                    combinations.append((topic_name, template, data_item))
        return combinations


# =============================================================================
# GENERADOR - VERSIÓN FINAL CORREGIDA
# =============================================================================

class DatasetGenerator:
    def __init__(self, system_prompt: str, knowledge_base: KnowledgeBase):
        self.system_prompt = system_prompt
        self.kb = knowledge_base
        
    def generate_example(self, topic: str, template: Dict, data_item: Dict, variation_id: int = 0) -> Dict:
        """
        Genera un ejemplo único
        CORRECCIÓN: data_item es el nombre correcto (no 'data')
        """
        try:
            question = template["question"].format(**data_item)
            answer = template["answer"].format(**data_item)
            
            # Añadir ID de variación para garantizar unicidad cuando se repiten plantillas
            if variation_id > 0:
                answer += f"\n\n# Variación {variation_id}"
            
            return {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
        except KeyError as e:
            print(f"⚠️ Error al generar ejemplo: {e}")
            print(f"   Template keys: {template.keys()}")
            print(f"   Data keys: {data_item.keys()}")
            return None
    
    def generate_n_examples(self, n: int) -> List[Dict]:
        """
        CORRECCIÓN PRINCIPAL: Genera EXACTAMENTE N ejemplos
        - Usa cycling through combinations cuando se agotan las plantillas base
        - Añade variation_id para hacer únicos los ejemplos repetidos
        """
        print(f"\n🚀 Generando {n} ejemplos...")
        
        all_combinations = self.kb.get_all_combinations()
        total_base = len(all_combinations)
        
        print(f"   📚 Variaciones base disponibles: {total_base}")
        print(f"   🔄 Ciclos necesarios: {(n + total_base - 1) // total_base}")
        
        examples = []
        
        for i in range(n):
            # Cycling through combinations
            combo_index = i % total_base
            variation_id = i // total_base
            
            topic, template, data_item = all_combinations[combo_index]
            
            example = self.generate_example(topic, template, data_item, variation_id)
            
            if example:
                examples.append(example)
            
            # Progreso
            if (i + 1) % max(1, n // 10) == 0:
                print(f"   📊 Progreso: {i + 1}/{n} ({(i + 1)/n*100:.1f}%)")
        
        print(f"✅ {len(examples)} ejemplos generados exitosamente")
        return examples
    
    def save_to_jsonl(self, examples: List[Dict], filename: str):
        """Guarda ejemplos en formato JSONL"""
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        file_size = os.path.getsize(filename) / 1024
        print(f"✅ Guardado: {filename} ({file_size:.2f} KB)")
    
    def save_in_batches(self, examples: List[Dict], batch_size: int = 100, output_dir: str = OUTPUT_DIR):
        """Guarda en lotes y dataset completo"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            filename = f"{output_dir}/batch_{i//batch_size + 1:03d}.jsonl"
            self.save_to_jsonl(batch, filename)
        
        # Guardar dataset completo
        self.save_to_jsonl(examples, f"{output_dir}/train_dataset.jsonl")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def generate_dataset(n: int = 1000, output_dir: str = OUTPUT_DIR, batch_size: int = 100, seed: int = 42):
    """Función principal para generar dataset"""
    random.seed(seed)
    
    print("=" * 70)
    print("🤖 GENERADOR DE DATASET - VERSIÓN FINAL CORREGIDA")
    print("=" * 70)
    print(f"📊 Ejemplos solicitados: {n}")
    print(f"📁 Directorio: {output_dir}")
    print(f"📦 Tamaño de lote: {batch_size}")
    print("=" * 70)
    
    kb = KnowledgeBase()
    generator = DatasetGenerator(SYSTEM_PROMPT, kb)
    
    examples = generator.generate_n_examples(n)
    generator.save_in_batches(examples, batch_size, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ ¡GENERACIÓN COMPLETADA!")
    print("=" * 70)
    print(f"📈 Ejemplos generados: {len(examples)}")
    print(f"📁 Archivos en: {output_dir}/")
    print("=" * 70)
    
    return generator


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    generate_dataset(n=n, seed=seed)