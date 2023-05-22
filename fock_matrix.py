#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import orbitals
import integrals
from numba import jit

def S(ao_list):
    n_orbitals = len(ao_list)
    overlap_matrix = np.zeros((n_orbitals, n_orbitals))

    for idx, a0 in enumerate(ao_list):
        for jdx, a1 in enumerate(ao_list):
            overlap_matrix[idx, jdx] = integrals.orbital_overlap(a0, a1)

    return overlap_matrix

def T(ao_list):
    n_orbitals = len(ao_list)
    kinetic_matrix = np.zeros((n_orbitals, n_orbitals))

    for idx, a0 in enumerate(ao_list):
        for jdx, a1 in enumerate(ao_list):
            kinetic_matrix[idx, jdx] = integrals.kinetic(a0, a1)

    return kinetic_matrix

def C(ao_list, atomlist, coordinates):
    n_orbitals = len(ao_list)
    coulombAttractionMatrix = np.zeros((n_orbitals,n_orbitals))
    for idx, a0 in enumerate(ao_list):
        for jdx, a1 in enumerate(ao_list):
            coulombAttractionMatrix[idx][jdx] = integrals.coulombicAttraction(ao_list[idx], ao_list[jdx], atomlist, coordinates)
    return coulombAttractionMatrix

def CT(ao_list):
    n = len(ao_list)
    coulombRepulsionTensor = np.zeros((n,n,n,n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    print(f"{i}{j}{k}{l}")
                    coulombRepulsionTensor[i][j][k][l] = integrals.ERI(ao_list[i],ao_list[j],ao_list[k],ao_list[l])
    return coulombRepulsionTensor

@jit(nopython=True)
def build_JK(n, norb, C, V_ee):
    JK = np.zeros((n,n))
    for j in range(n):
        for k in range(n):
            for l in range(n):
                for m in range(n):
                    for o in range(norb):
                        JK[j][k] += (C[l][o]*C[m][o] * 
                                    (2*V_ee[j][k][l][m]-
                                    V_ee[j][m][k][l]
                                    ))
    return JK

def main():
    return

if __name__== "__main__":
    main()