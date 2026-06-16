# KGeometric

Estrategia de k-partición que extiende el enfoque geométrico (caminos en hipercubo + costos de transición) para k ≥ 2.

## Problema

Dado un sistema con `u` nodos **mecanismo** y `v` nodos **efecto**, encontrar una
k-partición `(M₁, A₁), …, (Mₖ, Aₖ)` que minimice la **EMD** (Earth Mover's
Distance) entre la distribución marginal intacta del subsistema y la
distribución marginal inducida por la partición.

Cada bloque asocia un conjunto de mecanismos con un conjunto de efectos. La
distribución marginal del efecto `j` bajo la partición se computa promediando
sobre los estados de los mecanismos en su bloque, fijando los demás en su
estado inicial.

---

## Enfoque general

KGeometric combina tres estrategias según el tamaño del espacio de búsqueda:

| Condición | Estrategia | Exactitud |
|---|---|---|
| k = 2 | Caminos del hipercubo (Geometric clásico) | Exacta |
| k ≥ 3, ≤ 500K particiones | Enumeración de set partitions no etiquetadas | Exacta |
| k ≥ 3, > 500K particiones | Clustering geométrico + Familias A+B fallback | Aproximada |

---

## Algoritmo principal

```
_search_kgt2(inicio)
│
├─ 1. u = |presentes|, v = |futuros|
├─ 2. Si k > u+v → retornar ∞ (no válido)
│
├─ 3. count = count_k_partitions_unlabeled(u, v, k)
│    └─ Si count ≤ 500K → _search_exact()
│
├─ 4. influence = _build_influence_matrix(u, v)
│    (costos nivel-1 del hipercubo, sin construirlo)
│
├─ 5. cluster_result = _search_geometric_clustering(influence)
│    (método primario para k≥3)
│
├─ 6. Si cluster_result no tiene EMD=0:
│    ├─ cache de distribuciones para conjuntos mech 0..k-1
│    ├─ _run_family_A() — k-1 singletons
│    └─ Si factible: _run_family_B() — k-1 pares 1:1
│
└─ 7. Retornar mejor solución encontrada
```

---

## Matriz de influencia geométrica

### Cálculo

Mide cuánto cambia cada efecto `j` cuando **un solo** mecanismo `i` invierte su
estado, con el resto fijo en el estado inicial. Esto corresponde a las
transiciones de **nivel 1** del hipercubo.

```
influence[i][j] = 0.5 × |TPM(s_init)[j] − TPM(s_flip_i)[j]|
```

Donde:
- `s_init` = vector de estado inicial de todos los mecanismos
- `s_flip_i` = igual a `s_init` pero con el mecanismo `i` invertido
- El factor `0.5` = `1 / 2^Hamming(s_init, s_flip_i)` = `1/2^1`

### Propiedades

- `influence[i][j] ∈ [0, 0.5]`: 0 significa que el mecanismo `i` no afecta al
  efecto `j`; valores altos indican una relación causal fuerte.
- Cada mecanismo `i` tiene un **vector de influencia** de dimensión `v`.
  Mecanismos con vectores similares afectan a los mismos efectos con intensidad
  parecida.

### Ventaja

Se calcula en `O(u · v)` accesos directos a la TPM, sin construir el hipercubo
completo (`O(2ⁿ)`). Esto evita el OOM (Out Of Memory) para n ≥ 26.

---

## Clustering geométrico

Método **primario** para k ≥ 3 con espacio grande (>500K particiones).

### Idea

Agrupa mecanismos con patrones de influencia similares. Cada grupo (cluster) se
convierte en un bloque de la partición. Los efectos se asignan al cluster que
mejor los explica.

### Algoritmo

```
1. Inicializar: cada mecanismo es su propio cluster
2. Evaluar agrupación actual si clusters ≤ k:
   a. Asignar cada efecto al cluster con mayor Σ influence[i][j]
                                              i∈C
   b. Completar con clusters vacíos hasta k bloques
   c. Computar EMD
3. Fusionar los 2 clusters más similares
4. Repetir 2-3 hasta que quede 1 cluster
5. Retornar la mejor EMD encontrada (o None si no hay solución)
```

### Distancia entre clusters

Se usa distancia **L1 negativa** entre los centroides de influencia:

```
centroide(C) = (1/|C|) · Σ influence[i]     # vector promedio de dimensión v
                       i∈C

distancia(C₁, C₂) = −Σ |centroide(C₁)[j] − centroide(C₂)[j]|
                    j
```

En cada iteración se fusionan los dos clusters con **mayor distancia** (más
cercanos).

### Asignación de efectos

Dados `g ≤ k` clusters de mecanismos (el resto se completa con clusters vacíos
hasta `k`):

```
para cada efecto j:
    asignar j al cluster g con mayor Σ influence[i][j]
                                  i∈C₉
```

La asignación por **máxima** influencia asegura que los mecanismos que más
afectan a cada efecto queden en su mismo bloque. Un cluster vacío (sin
mecanismos) indica un bloque independiente cuyo efecto no depende de ningún
mecanismo.

### Evaluación

Para cada efecto `j` asignado al cluster C:

```
dist[j] = P(efecto_j = 1 | mecanismos en C varían, otros fijos en inicial)
```

Se computa vía marginalización directa (slicing numpy, sin NCube.marginalizar):

```python
index = []
for cada dimensión d:
    if d ∈ C:
        index.append(slice(None))   # promediar
    else:
        index.append(int(initial[d]))  # fijar
dist[j] = float(data[tuple(index)].mean())
```

La EMD se computa entre `dist` y la distribución marginal intacta:
```
EMD = emd_efecto(dist, intact)
```

---

## Familias A+B

Método **fallback** cuando el clustering geométrico no encuentra EMD = 0.

### Familia A: Singletons

Toma `k−1` ítems del total de `u+v` nodos. Cada ítem se convierte en un bloque
simple, y el resto va al **sink** (último bloque que concentra la complejidad
restante).

```
Ejemplo: u=3 (a,b,c), v=3 (A,B,C), k=4
Selección: a, c, B
  bloque 1: ({a}, ∅)
  bloque 2: ({c}, ∅)
  bloque 3: (∅, {B})
  sink:     ({b}, {A,C})
```

Candidatos: `C(u+v, k−1)`.

### Familia B: Pares 1:1

Toma `k−1` mecanismos y `k−1` efectos, los empareja en todas las permutaciones
posibles, y el resto va al sink.

```
Ejemplo: u=3 (a,b,c), v=3 (A,B,C), k=4
Selección de mecanismos: a, b
Selección de efectos: C, A
Permutación: a→C, b→A
  bloque 1: ({a}, {C})
  bloque 2: ({b}, {A})
  sink:     ({c}, {B})
```

Candidatos: `C(u, k−1) × C(v, k−1) × (k−1)!`.

Solo se ejecuta si este número no excede 500K.

### Cuándo se aplica cada familia

- **Familia A**: siempre, si se entra al modo heurístico (barrera económica:
  `C(u+v, k−1)` candidatos)
- **Familia B**: solo si `best_EMD > 0` (A no encontró solución perfecta) y el
  espacio de búsqueda es manejable (≤500K)

### Cache compartido

Ambas familias comparten un cache de distribuciones marginales precomputadas
para todos los conjuntos de mecanismos de tamaño 0 a `k−1`. La Familia B además
necesita conjuntos de tamaño `u−1` (todos los mecanismos menos uno, para los
pares). Esto evita recalcular `_marg_value` para el mismo conjunto múltiples
veces.

### Limitación

Ambas familias asumen que la complejidad se concentra en el sink. Fallan cuando
la partición óptima distribuye la complejidad en múltiples bloques, por ejemplo:

```
({a,b}, {C}), ({c}, {A,B})    # multi-elemento en varios bloques
```

En esos casos el clustering geométrico captura la estructura porque agrupa
mecanismos por su perfil de influencia, sin depender de un sink.

---

## Enumeración exacta

Usa `all_k_partitions_unlabeled` para generar todas las k-particiones no
etiquetadas de `u+v` nodos. Cada partición se evalúa con el mismo mecanismo
de marginalización directa.

Se activa solo si `count_k_partitions_unlabeled(u, v, k) ≤ 500.000`.

### Regiones de factibilidad

| n | k | Particiones | Exacto |
|---|---|---|---|
| ≤5 | cualquiera | ≤ 42.525 | Sí |
| 6 | 3 | 86.526 | Sí |
| 6 | ≥4 | > 611K | No |
| ≥8 | ≥3 | > 7M | No |

---

## k = 2

Reimplementación del algoritmo Geometric clásico.

### Algoritmo

```
_search_k2()
├─ 1. _compute_geometric_data()
│    (caminos del hipercubo + tabla de costos de transición)
├─ 2. _identify_bipartitions()
│    (candidatos desde niveles del hipercubo)
├─ 3. Para cada candidato:
│    ├─ System.bipartir(futuros, presentes)
│    └─ EMD con distribución marginal
└─ 4. Retornar mejor bipartición
```

Para `n > 14` usa `_search_k2_selection()` (fallback rápido que prueba cada
mecanismo o efecto como singleton candidato).

### Componentes geométricos

- `_caminos[nivel]`: lista de estados en cada nivel del hipercubo desde
  `s_inicial` hasta `s_final`
- `_tabla_transiciones[(s₁, s₂)][j]`: costo de transición del efecto `j` cuando
  los mecanismos pasan de `s₁` a `s₂`, acumulado como `0.5^d · Σ|diff|`
- `_identify_bipartitions()`: en cada nivel, selecciona los mecanismos que
  difieren del estado inicial como "futuros" y los que no como "presentes"

---

## Optimizaciones de memoria

| Optimización | Descripción | Impacto |
|---|---|---|
| Influence directa | Costos nivel-1 sin hipercubo completo | O(2ⁿ) → O(u·v) tiempo y RAM |
| Compactación | Cubos float64 → float32 | 2× menos RAM de cubos |
| Liberación TPM | `self.tpm = None` tras cargar | Libera copia de la TPM |
| Carga uint8 | TPMs binarias n>20 como uint8 | 4× menos RAM de TPM |

Para n=26 estas optimizaciones reducen el consumo de ~14 GB (float64 + hipercubo)
a ~1.74 GB (uint8 + clustering).

---

## Complejidad

| k | Componente | Tiempo | Memoria |
|---|---|---|---|
| 2 | Caminos hipercubo | O(2ⁿ) | O(2ⁿ) |
| ≥3 | Matriz de influencia | O(u·v) | O(u·v) |
| ≥3 | Clustering | O(u²·v) | O(u·v) |
| ≥3 | Cache fallback | O(C(u, 0..k−1)·v) | O(C(u, 0..k−1)·v) |
| ≥3 | Familia A | O(C(u+v, k−1)·v) | O(v) |
| ≥3 | Enumeración exacta | O(#particiones·v) | O(v) |

### Tiempos empíricos

| Sistema | k | Tiempo | Método |
|---|---|---|---|
| N4A | 5 | 0.04s | Exacto |
| N6A | 3 | 5.9s | Exacto |
| N6A | 4 | 0.03s | Clustering + A |
| N8A | 5 | 0.08s | Clustering + A |
| N14A | 5 | 1.6s | Clustering + A |
| N20A | 5 | 3.1s | Clustering + A |
