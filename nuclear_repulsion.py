#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

nuc_charges = dict(
    H=1,
    He=2,
    Li=3,
    Be=4,
    B=5,
    C=6,
    N=7,
    O=8,
    F=9,
    Ne=10
)

atoms = ["C", "H", "H", "C", "H", "H"]
coordinates = [
    np.array([0., 0., 0.660022]),
    np.array([0., 0.910176, 1.227515]),
    np.array([0., -0.910176, 1.227515]),
    np.array([0., 0., -0.660022]),
    np.array([0., -0.910176, -1.227515]),
    np.array([0., 0.910176, -1.227515])
    ]

ncp = 0
for idx in range(len(atoms)):
    for jdx in (range(idx+1,len(atoms))):
        product = nuc_charges[atoms[idx]]*nuc_charges[atoms[jdx]]
        distance = np.linalg.norm(1.88973*(coordinates[idx]-coordinates[jdx]))
        ncp += product/distance

print(ncp)

