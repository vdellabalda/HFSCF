#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Name:      Della Balda, Vincente
   Email:     vincente.dellabalda@uzh.ch
   Date:      24 April, 2023
   Kurs:      ESC203
   Semester:  FS23
   Week:      Project
   Thema:     Hartree-Fock self-consistent field solver
"""

import numpy as np
import orbitals
import integrals
from integrals import factorial2
import fock_matrix
import plotly.graph_objects as go
from scipy.linalg import eigh
from numba import jit

# ================================== 
#
#    Constant data
# 
# ==================================

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

orbitaltypes = ["S", "P", "SP"]

# ==================================
#
#    Classes
# 
# ==================================

class Gaussian1d:
    def __init__(self, center, angular, exponent):
        self.center = center
        self.angular = angular
        self.exponent = exponent
    
    def __call__(self, x):
        return (x - self.center)**self.angular*np.exp(-self.exponent*(x-self.center)**2)

class Ao:
    def __init__(self, center, angular, contraction):
        self.center = center
        self.angular = angular
        self.exponents = [0.0 for i in range(contraction)]
        self.coefficients = [0.0 for i in range(contraction)]
        self.norm = [0.0 for i in range(contraction)]

    def normalize(self):
        l, m, n = self.angular
        for idx, exponent in enumerate(self.exponents):
            self.norm[idx] = np.power(2*exponent/np.pi, 0.75)*\
                        np.power(4*exponent, np.sum(self.angular)/2)/\
                        np.power(factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1), 0.5)

    def __call__(self, x, y, z):
        value = 0
        x0, y0, z0 = self.center
        lx, ly, lz = self.angular
        for idx, coefficient in enumerate(self.coefficients):
            exponent = self.exponents[idx]
            g1dx = Gaussian1d(x0, lx, exponent)
            g1dy = Gaussian1d(y0, ly, exponent)
            g1dz = Gaussian1d(z0, lz, exponent)
            value += self.norm[idx]*coefficient*g1dx(x)*g1dy(y)*g1dz(z)
        return value

class Mo:
    def __init__(self, ao_list, coeffs):
        self.ao_list = ao_list
        self.coeffs = coeffs
        self.norm = []

    def normalize(self):
        for i in range(len(self.ao_list)):
            ao = self.ao_list[i]
            l, m, n = ao.angular
            exponent = ao.exponent
            res += self.coeffs[i]*ao(x, y, z)
        
    def __repr__(self):
        return str(self.coeffs) + repr(self.ao_list)
    
    def __call__(self, x, y, z):
        res = 0
        for i in range(len(self.ao_list)):
            ao = self.ao_list[i]
            res += self.coeffs[i]*ao(x, y, z)
        return res

# ==================================
#
#    Functions
# 
# ==================================

def SCF(n, norb, S, C, T, V_ee, V_ek, tol, maxiter):
    prev = float('inf')
    SVAL, SVEC = np.linalg.eig(S)  
    SVAL_minhalf = (np.diag(SVAL**(-0.5)))  
    S_minhalf = np.dot(SVEC,np.dot(SVAL_minhalf,np.transpose(SVEC)))

    fock_list = []
    error_list = []

    # initial guess
    H = T + V_ek
    C = np.divide(H, np.sum(H, axis=0))

    D = np.zeros_like(C)
    for mu in range(D.shape[0]):  
        for nu in range(D.shape[1]):
            for m in range(norb):  
                D[mu,nu] += 2*C[mu,m]*C[nu,m]
    D_new = np.zeros_like(C)

    for cycle in range(maxiter):
        
        # current JK matrix
        JK = fock_matrix.build_JK(n, norb, C, V_ee)

        # current Fock matrix
        F = H + JK

        # DIIS
        error_1 = np.matmul(np.matmul(S,D),F)
        error_2 = np.matmul(np.matmul(F,D),S)
        error = error_1 - error_2
        if(len(error_list)) > 7:
            max_error = 0
            for idx, error in enumerate(error_list):
                temp_error = np.sum(np.abs(error))
                if temp_error > max_error:
                    max_idx = idx
            error_list.pop(max_idx)
            fock_list.pop(max_idx)
        error_list.append(error.copy())
        fock_list.append(F.copy())        
        if len(fock_list) > 1:
            error_length = len(error_list)
            error_matrix = -1*np.ones((error_length+1, error_length+1))
            error_matrix[-1, -1] = 0
            for idx in range(error_length):
                for jdx in range(error_length):
                    error_matrix[idx, jdx] = np.trace(np.dot(error_list[idx], error_list[jdx]))
            rhs = np.zeros(error_length+1)
            rhs[-1] = -1
            coefficients = np.linalg.solve(error_matrix, rhs)
            diis_F = np.zeros_like(F)
            for idx, coefficient in enumerate(coefficients[:-1]):
                diis_F += coefficient*fock_list[idx]
        else:
            diis_F = F

        # transform F to orthogonal AO basis
        diis_F_prime = np.dot(np.transpose(S_minhalf),np.dot(diis_F,S_minhalf))
        # solve eigenvalue equation
        energy, C_prime = np.linalg.eigh(diis_F_prime)
        # retrieve coefficient matrix 
        C_new = np.dot(S_minhalf,C_prime)  

        # build density matrix
        for mu in range(D_new.shape[0]):  
            for nu in range(D_new.shape[1]):
                D_new[mu,nu] = 0
                for m in range(norb):  
                    D_new[mu,nu] += 2*C_new[mu,m]*C_new[nu,m]

        EHF = 0
        for idx in range(len(energy)):
            for jdx in range(len(energy)):
                EHF += 0.5*D_new[idx,jdx]*(H[idx,jdx] + diis_F[idx,jdx])

        delta = EHF - prev
        density_delta = np.sqrt(np.sum((D - D_new)**2))
        # check convergence
        if abs(delta) < tol and density_delta < tol:
            print(f'SCF Converged after {cycle+1} steps')
            break

        prev = EHF
        C = C_new.copy()
        D = D_new.copy()
        print('EHF:'+str(EHF)+" "+'prev:'+str(prev)+' '+'delta:'+str(delta)+' '+'density delta:'+str(density_delta))

    return EHF, energy, C_new

# ================================== 
#
#    Main
# 
# ==================================

def main():

    molecules = dict(
        formaldehyde=(
            ["H", "H", "C", "O"],

            [1.88973*np.array([0.000000,    0.920117,   -1.102442]),
            1.88973*np.array([-0.000000,   -0.920117,   -1.102442]),
            1.88973*np.array([0.000000,    0.000000,   -0.534011]),
            1.88973*np.array([-0.000000,  -0.000000,    0.676119])]
            ),
        water=(
            ["O", "H", "H"],

            [1.88973*np.array([0., 0., 0.116321]),
            1.88973*np.array([0., 0.751155, -0.465285]),
            1.88973*np.array([0., -0.751155, -0.465285])
            ]
        ),
        ethene=(
            ["C", "H", "H", "C", "H", "H"],

            [1.88973*np.array([0., 0., 0.660022]),
            1.88973*np.array([0., 0.910176, 1.227515]),
            1.88973*np.array([0., -0.910176, 1.227515]),
            1.88973*np.array([0., 0., -0.660022]),
            1.88973*np.array([0., -0.910176, -1.227515]),
            1.88973*np.array([0., 0.910176, -1.227515])
            ]),
        ethanol=(
            ["H", "H", "H", "C", "C", "H", "H", "O", "H"],
            [
            1.88973*np.array([1.252061,   -0.865351,    0.877772]),
            1.88973*np.array([2.081899,    0.423065,   -0.000031]),
            1.88973*np.array([1.252039,   -0.865523,   -0.877553]),
            1.88973*np.array([1.219651,   -0.233443,    0.000049]),
            1.88973*np.array([-0.075905,    0.569943,   -0.000042]),
            1.88973*np.array([-0.122215,    1.205134,   -0.878830]),
            1.88973*np.array([-0.122329,    1.205198,    0.878698]),
            1.88973*np.array([-1.149376,   -0.395839,   -0.000105]),
            1.88973*np.array([-2.008922,    0.045191,    0.000742])
            ])            
        )

    molecule = "water"
    basis = "STO-3G"
    filename = f"{basis}.txt"
    ao_dict = orbitals.create_ao_dict(filename)

    ao_list = []
    
    atoms = molecules[molecule][0]
    coordinates = molecules[molecule][1]

    for atom, coordinate in zip(atoms, coordinates):
        for atomic_orbital in ao_dict[atom]:
            new_orbital = Ao(coordinate, atomic_orbital.angular, len(atomic_orbital.coefficients))
            new_orbital.exponents = atomic_orbital.exponents.copy()
            new_orbital.coefficients = atomic_orbital.coefficients.copy()
            new_orbital.normalize()
            ao_list.append(new_orbital)

    S = fock_matrix.S(ao_list)
    T = fock_matrix.T(ao_list)
    V_ek = fock_matrix.C(ao_list, atoms, coordinates)
    V_ee = fock_matrix.CT(ao_list)

    #V_ee = np.load(f"V_ee_{molecule}_{basis}.dat.npy")

    n = len(ao_list)
    n_electrons = 0
    for atom in atoms:
        n_electrons += nuc_charges[atom]
    norb = n_electrons // 2
    maxiter = 100
    C = np.eye(n) #coeffcient matrix
    tol = 1e-12
    EHF, energy, C = SCF(n, norb, S, C, T, V_ee, V_ek, tol, maxiter)

    ncp = 0
    for idx in range(len(atoms)):
        for jdx in (range(idx+1,len(atoms))):
            product = nuc_charges[atoms[idx]]*nuc_charges[atoms[jdx]]
            distance = np.linalg.norm((coordinates[idx]-coordinates[jdx]))
            ncp += product/distance

    print(f"Nuclear repulsion energy: {ncp} Hartree")
    print(f"Hartree Fock energy: {EHF+ncp} Hartree")

    np.savetxt(f'coefficients_{molecule}_{basis}.csv', C, delimiter=",")
    np.savetxt(f'energies_{molecule}_{basis}.csv', energy, delimiter=",")
    np.save(f"V_ee_{molecule}_{basis}.dat", V_ee)
    np.save(f"V_ek_{molecule}_{basis}.dat", V_ek)
    np.save(f"S_{molecule}_{basis}.dat", S)
    np.save(f"T_{molecule}_{basis}.dat", T)
     
    return

if __name__== "__main__":
    main()
