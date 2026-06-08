"""Recompute N3A analysis with the correct initial state '100'."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.loader import TpmLoader
from src.models.system import System
from src.functions.emd import emd_efecto
from tests.test_kpartition_refinement import all_k_partitions, k_partition_distribution, refines

tpm = TpmLoader.cargar(3, 'A')
labels = ['A','B','C']

init_str = '100'
estado = np.array([int(c) for c in init_str], dtype=np.int8)
system = System(tpm, estado)

intact = system.distribucion_marginal()
vals = ', '.join(f'{v:.4f}' for v in intact)
print(f'Estado inicial: {init_str}  (poco-endian: A*1 + B*2 + C*4 = fila {estado[0]*1 + estado[1]*2 + estado[2]*4})')
print(f'Distribucion intacta: [{vals}]')
print()

print('Todas las biparticiones (k=2) ordenadas por EMD:')
res2 = []
for kp in all_k_partitions(3, 3, 2):
    dist = k_partition_distribution(system, kp)
    emd = emd_efecto(dist, intact)
    res2.append((emd, kp, dist))
res2.sort(key=lambda x: x[0])
for emd, kp, dist in res2:
    blocks = '  '.join(
        'M={' + ','.join(labels[m] for m in sorted(m)) + '}->A={' + ','.join(labels[a] for a in sorted(a)) + '}'
        for m,a in kp
    )
    dstr = ', '.join(f'{v:.4f}' for v in dist)
    tag = '  <<< OPT' if emd < 0.26 else ''
    print(f'  EMD={emd:.4f}  {blocks}  [{dstr}]{tag}')

print()
print('Triparticion (k=3, unica):')
for kp in all_k_partitions(3, 3, 3):
    dist = k_partition_distribution(system, kp)
    emd = emd_efecto(dist, intact)
    blocks = '  '.join(
        'M={' + ','.join(labels[m] for m in sorted(m)) + '}->A={' + ','.join(labels[a] for a in sorted(a)) + '}'
        for m,a in kp
    )
    dstr = ', '.join(f'{v:.4f}' for v in dist)
    print(f'  EMD={emd:.4f}  {blocks}  [{dstr}]')

print()
print('Verificacion de refinamiento:')
kp3 = list(all_k_partitions(3, 3, 3))[0]
for emd, kp2, dist in res2:
    r = refines(kp3, kp2)
    optag = 'OPT' if emd < 0.26 else ''
    ref = 'REFINED' if r else '-'
    blocks2 = '  '.join(
        'M={' + ','.join(labels[m] for m in sorted(m)) + '}->A={' + ','.join(labels[a] for a in sorted(a)) + '}'
        for m,a in kp2
    )
    print(f'  EMD={emd:.2f}  {blocks2}  {optag:5s}  refina={r} {ref}')
