# Resolución de la mochila 0/1 mediante branch and bound
# con poda usando cota superior fraccionaria

class Item:
    def __init__(self, peso, valor):
        self.peso = peso
        self.valor = valor
        # Relación valor/peso para ordenar
        self.ratio = valor / peso

def knapsack_fraccionaria(items, capacidad):
    """
    Calcula el valor óptimo del problema de la mochila FRACCIONARIA
    (cota superior para un conjunto de ítems dado).
    """
    # Ordenar por ratio descendente
    items_ordenados = sorted(items, key=lambda x: x.ratio, reverse=True)
    valor_total = 0.0
    capacidad_restante = capacidad
    for item in items_ordenados:
        if capacidad_restante >= item.peso:
            valor_total += item.valor
            capacidad_restante -= item.peso
        else:
            # tomar fracción
            valor_total += item.valor * (capacidad_restante / item.peso)
            break
    return valor_total

def greedy_binario(items, capacidad):
    """
    Heurística greedy para mochila 0/1 (solución factible inicial).
    Ordena por ratio valor/peso y toma el ítem completo si cabe.
    """
    items_ordenados = sorted(items, key=lambda x: x.ratio, reverse=True)
    peso_actual = 0
    valor_actual = 0
    for item in items_ordenados:
        if peso_actual + item.peso <= capacidad:
            peso_actual += item.peso
            valor_actual += item.valor
    return valor_actual

class Nodo:
    def __init__(self, nivel, valor_actual, peso_actual, items_tomados):
        self.nivel = nivel          # índice del ítem que se decide en este nivel (0..n-1)
        self.valor_actual = valor_actual
        self.peso_actual = peso_actual
        self.items_tomados = items_tomados[:]  # lista de 0/1

def branch_and_bound(items, capacidad):
    # Ordenar ítems por ratio valor/peso descendente (necesario para la cota)
    items = sorted(items, key=lambda x: x.ratio, reverse=True)
    n = len(items)
    
    # Cota inferior inicial (solución factible)
    mejor_valor = greedy_binario(items, capacidad)
    mejor_solucion = None
    
    # Pila DFS (se puede usar cola también)
    pila = []
    nodo_raiz = Nodo(nivel=0, valor_actual=0, peso_actual=0, items_tomados=[])
    pila.append(nodo_raiz)
    
    while pila:
        nodo = pila.pop()
        
        # Si ya hemos procesado todos los ítems (nodo hoja)
        if nodo.nivel == n:
            if nodo.valor_actual > mejor_valor:
                mejor_valor = nodo.valor_actual
                mejor_solucion = nodo.items_tomados
            continue
        
        # Próximo ítem a considerar
        item = items[nodo.nivel]
        
        # --- Rama 1: tomar el ítem (si cabe) ---
        if nodo.peso_actual + item.peso <= capacidad:
            nuevo_valor = nodo.valor_actual + item.valor
            nuevo_peso = nodo.peso_actual + item.peso
            nuevos_tomados = nodo.items_tomados + [1]
            hijo_tomar = Nodo(nodo.nivel + 1, nuevo_valor, nuevo_peso, nuevos_tomados)
            
            # Calcular cota superior para este hijo (incluye ítems restantes)
            items_restantes = items[nodo.nivel + 1:]
            # La capacidad restante es la capacidad total - nuevo_peso
            capacidad_restante = capacidad - nuevo_peso
            # La cota superior es valor_actual + valor fraccionario de los restantes
            if items_restantes and capacidad_restante > 0:
                cota = nuevo_valor + knapsack_fraccionaria(items_restantes, capacidad_restante)
            else:
                cota = nuevo_valor
            
            # Poda: solo explorar si la cota es mayor que el mejor valor conocido
            if cota > mejor_valor:
                pila.append(hijo_tomar)
        
        # --- Rama 2: no tomar el ítem ---
        nuevos_tomados_no = nodo.items_tomados + [0]
        hijo_no_tomar = Nodo(nodo.nivel + 1, nodo.valor_actual, nodo.peso_actual, nuevos_tomados_no)
        
        # Cota superior para la rama de no tomar
        items_restantes_no = items[nodo.nivel + 1:]
        capacidad_restante_no = capacidad - nodo.peso_actual
        if items_restantes_no and capacidad_restante_no > 0:
            cota_no = nodo.valor_actual + knapsack_fraccionaria(items_restantes_no, capacidad_restante_no)
        else:
            cota_no = nodo.valor_actual
        
        if cota_no > mejor_valor:
            pila.append(hijo_no_tomar)
    
    return mejor_valor, mejor_solucion

# Ejemplo de uso
if __name__ == "__main__":
    # Ítems: (peso, valor)
    datos = [(1, 2), (2, 3), (3, 4), (4, 5)]
    items = [Item(p, v) for p, v in datos]
    capacidad = 7
    
    valor_optimo, solucion = branch_and_bound(items, capacidad)
    print(f"Valor óptimo: {valor_optimo}")
    print(f"Solución (tomar/no tomar según orden de ratio): {solucion}")