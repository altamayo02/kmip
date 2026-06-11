"""Verify N3A refinement with correct empty-block partitions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.loader import TpmLoader
from src.models.system import System
from src.functions.emd import emd_efecto
from tests.test_kpartition_refinement import all_k_partitions, k_partition_distribution, refines

tpm = TpmLoader.cargar(3, 'A')
init = np.array([1, 0, 0], dtype=np.int8)  # '100' como usa BruteForce
system = System(tpm, init)
labels = ['A', 'B', 'C']

intact = system.distribucion_marginal()
print(f'Intacta (estado 100): [{intact[0]:.2f}, {intact[1]:.2f}, {intact[2]:.2f}]')
print()

# Find opt k=2 and k=3
opt2 = []
for kp in all_k_partitions(3, 3, 2):
    #print('2-Part:', kp)
    dist = k_partition_distribution(system, kp)
    emd = emd_efecto(dist, intact)
    opt2.append((emd, kp, dist))
opt2.sort(key=lambda x: x[0])
best2 = opt2[0][0]

print()
opt3 = []
for kp in all_k_partitions(3, 3, 3):
    dist = k_partition_distribution(system, kp)
    emd = emd_efecto(dist, intact)
    opt3.append((emd, kp, dist))
opt3.sort(key=lambda x: x[0])
best3 = opt3[0][0]

print(f'Optimal k=2: EMD={best2:.4f}')
for emd, kp, dist in opt2:
    if abs(emd - best2) > 1e-10: break
    blocks = '  |  '.join(
        'M={' + ','.join(labels[m] for m in sorted(m)) + '}->A={' + ','.join(labels[a] for a in sorted(a)) + '}'
        for m,a in kp
    )
    print(f'  [{blocks}]  P=[{dist[0]:.2f},{dist[1]:.2f},{dist[2]:.2f}]')

print()
print(f'Optimal k=3: EMD={best3:.4f}')
opt3_best = []
for emd, kp, dist in opt3:
    if abs(emd - best3) > 1e-10: continue
    opt3_best.append((kp, dist))
    blocks = '  |  '.join(
        'M={' + ','.join(labels[m] for m in sorted(m)) + '}->A={' + ','.join(labels[a] for a in sorted(a)) + '}'
        for m,a in kp
    )
    print(f'  [{blocks}]  P=[{dist[0]:.2f},{dist[1]:.2f},{dist[2]:.2f}]')

print()
print('Verificacion: las k=3 optimas refinan a k=2 optima?')
kp2_opt = [kp for emd, kp, _ in opt2 if abs(emd - best2) < 1e-10]
for kp3, dist3 in opt3_best:
    refines_any = any(refines(kp3, kp2) for kp2 in kp2_opt)
    b3 = '  |  '.join(
        'M={' + ','.join(labels[m] for m in sorted(m)) + '}->A={' + ','.join(labels[a] for a in sorted(a)) + '}'
        for m,a in kp3
    )
    status = 'VIOLA' if not refines_any else 'ok'
    print(f'  [{b3}]  {status}')

print()
# Explain the crossing in estado-nodo terms
print('Por que cruza?')
kp2 = kp2_opt[0]  # M={A,B,C}->{A,C} | M={}->{B}
kp3 = opt3_best[0][0]  # first optimal k=3

# For the crossing block
print()
print('En k=2 optima:')
for bi, (m, a) in enumerate(kp2):
    m_str = ','.join(labels[x] for x in sorted(m)) if m else 'EMPTY'
    a_str = ','.join(labels[x] for x in sorted(a)) if a else 'EMPTY'
    print(f'  Bloque {bi}: M={{{m_str}}}  A={{{a_str}}}')

print()
print('En k=3 optima:')
for bi, (m, a) in enumerate(kp3):
    m_str = ','.join(labels[x] for x in sorted(m)) if m else 'EMPTY'
    a_str = ','.join(labels[x] for x in sorted(a)) if a else 'EMPTY'
    print(f'  Bloque {bi}: M={{{m_str}}}  A={{{a_str}}}')

print()
print('El cruce:')
for bi, (m, a) in enumerate(kp3):
    if not m or not a:
        continue
    m0 = next(iter(m))
    a0 = list(a)[0]
    for bj, (m2, a2) in enumerate(kp2):
        if m0 in m2: bm = bj
        if a0 in a2: ba = bj
    if bm != ba:
        print(f'  Bloque {bi} de k=3: {labels[m0]}->{labels[a0]}')
        print(f'    mecanismo {labels[m0]} esta en k=2 bloque {bm}')
        print(f'    alcance   {labels[a0]} esta en k=2 bloque {ba}')
        print(f'    -> CRUCE: estan en bloques distintos!')
