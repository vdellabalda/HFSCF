#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numba
import orbitals
from scipy.special import hyp1f1
import numba_special


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

@numba.jit(nopython=True)
def factorial2(n):
    if n == 1 or n == 0 or n == -1:
        return 1 
    res = 1
    for i in range(n, -1, -2):
        if(i == 0 or i == 1):
            return res
        else:
            res *= i

@numba.jit(nopython=True)
def norm_coeff(exponent, angular):
    l, m, n = angular
    N = np.power(2*exponent/np.pi, 0.75)*\
        np.power(4*exponent, np.sum(angular)/2)/\
        np.power(factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1), 0.5)
    return N

@numba.jit(nopython=True)
def recurrence(center0, center1, angular0, angular1, exponent0, exponent1):
    center01 = (exponent0*center0+exponent1*center1)/(exponent0+exponent1)

    if angular1 != 0:
        term1 = recurrence(center0, center1, angular0+1, angular1-1, exponent0, exponent1)
        term2 = recurrence(center0, center1, angular0, angular1-1, exponent0, exponent1)
        return term1+(center0-center1)*term2

    elif angular0 > 1:
        term1 = recurrence(center0, center1, angular0-1, 0, exponent0, exponent1)
        term2 = recurrence(center0, center1, angular0-2, 0, exponent0, exponent1)
        return -(center0-center01)*term1 + (angular0-1)/(2*(exponent0+exponent1))*term2

    elif angular0 == 0 and angular1 == 0:
        return 1

    elif angular0 == 1 and angular1 == 0:
        return -(center0-center01)

    else:
        print("Error")

@numba.jit(nopython=True)
def gaussian_1d_overlap(center0, center1, angular0, angular1, exponent0, exponent1):
    overlap = recurrence(center0, center1, angular0, angular1, exponent0, exponent1)
    return overlap


def orbital_overlap(a0, a1):
    overlap = 0

    center0_x, center0_y, center0_z = a0.center
    angular0_x, angular0_y, angular0_z = a0.angular
    exponents0 = a0.exponents
    coeffs0 = a0.coefficients

    center1_x, center1_y, center1_z = a1.center
    angular1_x, angular1_y, angular1_z = a1.angular
    exponents1 = a1.exponents
    coeffs1 = a1.coefficients

    for coeff0, exponent0 in zip(coeffs0, exponents0):
        norm_coeff0 = norm_coeff(exponent0, a0.angular)
        for coeff1, exponent1 in zip(coeffs1, exponents1):
            prefactor = np.power(np.pi/(exponent0+exponent1), 1.5)
            EAB = np.exp(-(exponent0*exponent1/(exponent0+exponent1))*np.sum((a0.center-a1.center)**2))
            norm_coeff1 = norm_coeff(exponent1, a1.angular)
            overlap_x = gaussian_1d_overlap(
                    a0.center[0], a1.center[0],
                    a0.angular[0], a1.angular[0],
                    exponent0, exponent1)
            overlap_y = gaussian_1d_overlap(
                    a0.center[1], a1.center[1],
                    a0.angular[1], a1.angular[1],
                    exponent0, exponent1)
            overlap_z = gaussian_1d_overlap(
                    a0.center[2], a1.center[2],
                    a0.angular[2], a1.angular[2],
                    exponent0, exponent1)
            overlap += EAB*prefactor*norm_coeff0*norm_coeff1*coeff0*coeff1*overlap_x*overlap_y*overlap_z
    
    return overlap

@numba.jit(nopython=True)
def kinetic_1d(center0, center1, angular0, angular1, exponent0, exponent1):
    if angular0 == 0 and angular1 == 0:
        term = 2*exponent0*exponent1*gaussian_1d_overlap(center0, center1, 1, 1, exponent0, exponent1)
        return term

    elif angular1 == 0:
        term = -angular0*exponent1*gaussian_1d_overlap(center0, center1, angular0-1, 1, exponent0, exponent1)+\
            2*exponent0*exponent1*gaussian_1d_overlap(center0, center1, angular0+1, 1, exponent0, exponent1)
        return term

    elif angular0 == 0:
        term = -angular1*exponent0*gaussian_1d_overlap(center0, center1, 1, angular1-1, exponent0, exponent1)+\
            2*exponent0*exponent1*gaussian_1d_overlap(center0, center1, 1, angular1+1, exponent0, exponent1)
        return term

    else:
        term = 0.5*(angular0*angular1*gaussian_1d_overlap(center0, center1, angular0-1, angular1-1, exponent0, exponent1)-\
            2*exponent0*angular1*gaussian_1d_overlap(center0, center1, angular0+1, angular1-1, exponent0, exponent1)-\
            2*angular0*exponent1*gaussian_1d_overlap(center0, center1, angular0-1, angular1+1, exponent0, exponent1)+\
            4*exponent0*exponent1*gaussian_1d_overlap(center0, center1, angular0+1, angular1+1, exponent0, exponent1))
        return term

def kinetic(a0, a1):
    T = 0
    center0_x, center0_y, center0_z = a0.center
    angular0_x, angular0_y, angular0_z = a0.angular
    exponents0 = a0.exponents
    coeffs0 = a0.coefficients

    center1_x, center1_y, center1_z = a1.center
    angular1_x, angular1_y, angular1_z = a1.angular
    exponents1 = a1.exponents
    coeffs1 = a1.coefficients

    for coeff0, exponent0 in zip(coeffs0, exponents0):
        norm_coeff0 = norm_coeff(exponent0, a0.angular)
        for coeff1, exponent1 in zip(coeffs1, exponents1):
            prefactor = np.power(np.pi/(exponent0+exponent1), 1.5)
            EAB = np.exp(-(exponent0*exponent1/(exponent0+exponent1))*np.sum((a0.center-a1.center)**2))
            norm_coeff1 = norm_coeff(exponent1, a1.angular)
            T += EAB*prefactor*norm_coeff0*norm_coeff1*coeff0*coeff1*\
                kinetic_1d(center0_x, center1_x, angular0_x, angular1_x, exponent0, exponent1)*\
                gaussian_1d_overlap(center0_y, center1_y, angular0_y, angular1_y, exponent0, exponent1)*\
                gaussian_1d_overlap(center0_z, center1_z, angular0_z, angular1_z, exponent0, exponent1)
            T += EAB*prefactor*norm_coeff0*norm_coeff1*coeff0*coeff1*\
                gaussian_1d_overlap(center0_x, center1_x, angular0_x, angular1_x, exponent0, exponent1)*\
                kinetic_1d(center0_y, center1_y, angular0_y, angular1_y, exponent0, exponent1)*\
                gaussian_1d_overlap(center0_z, center1_z, angular0_z, angular1_z, exponent0, exponent1)
            T += EAB*prefactor*norm_coeff0*norm_coeff1*coeff0*coeff1*\
                gaussian_1d_overlap(center0_x, center1_x, angular0_x, angular1_x, exponent0, exponent1)*\
                gaussian_1d_overlap(center0_y, center1_y, angular0_y, angular1_y, exponent0, exponent1)*\
                kinetic_1d(center0_z, center1_z, angular0_z, angular1_z, exponent0, exponent1)
    
    return T

@numba.jit(nopython=True)
def boys(n,T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

@numba.jit(nopython=True)
def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

@numba.jit(nopython=True)
def E(i,j,t,Qx,a,b):
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t  
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)

@numba.jit(nopython=True)
def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    l1,m1,n1 = lmn1 
    l2,m2,n2 = lmn2
    p = a + b
    P = (a*A+b*B)/(a+b) # Gaussian composite center
    RPC = np.linalg.norm(P-C)

    val = 0.0
    for t in range(l1+l2+1):
        E1 = E(l1,l2,t,A[0]-B[0],a,b)
        for u in range(m1+m2+1):
            E2 = E(m1,m2,u,A[1]-B[1],a,b)
            for v in range(n1+n2+1):
                E3 = E(n1,n2,v,A[2]-B[2],a,b)
                val += E1*E2*E3* \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    val *= 2*np.pi/p 
    return val


def V(a0,a1,C):
    v = 0.0

    center0_x, center0_y, center0_z = a0.center
    angular0_x, angular0_y, angular0_z = a0.angular
    exponents0 = a0.exponents
    coeffs0 = a0.coefficients

    center1_x, center1_y, center1_z = a1.center
    angular1_x, angular1_y, angular1_z = a1.angular
    exponents1 = a1.exponents
    coeffs1 = a1.coefficients

    for coeff0, exponent0 in zip(coeffs0, exponents0):
        norm_coeff0 = norm_coeff(exponent0, a0.angular)
        for coeff1, exponent1 in zip(coeffs1, exponents1):
            norm_coeff1 = norm_coeff(exponent1, a1.angular)

            v += norm_coeff0*norm_coeff1*coeff0*coeff1*\
                     nuclear_attraction(exponent0,a0.angular,a0.center,
                     exponent1,a1.angular,a1.center,C)
    return v


def coulombicAttraction(a0, a1, atomlist, coordinates):
    res = 0
    for i, atom in enumerate(atomlist):
        center = coordinates[i]
        Z = nuc_charges[atom]
        res += (-Z)*V(a0, a1, center)
    return res

@numba.jit(nopython=True)
def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
    q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
    alpha = p*q/(p+q)
    P = (a*A+b*B)/(a+b) # A and B composite center
    Q = (c*C+d*D)/(c+d) # C and D composite center
    RPQ = np.linalg.norm(P-Q)

    val = 0.0
    for t in range(l1+l2+1):
        E1 = E(l1,l2,t,A[0]-B[0],a,b)
        for u in range(m1+m2+1):
            E2 = E(m1,m2,u,A[1]-B[1],a,b)
            for v in range(n1+n2+1):
                E3 = E(n1,n2,v,A[2]-B[2],a,b)
                for tau in range(l3+l4+1):
                    E4 = E(l3,l4,tau,C[0]-D[0],c,d)
                    for nu in range(m3+m4+1):
                        E5 = E(m3,m4,nu ,C[1]-D[1],c,d)
                        for phi in range(n3+n4+1):
                            E6 = E(n3,n4,phi,C[2]-D[2],c,d)
                            val += E1*E2*E3*E4*E5*E6* \
                                   np.power(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
    return val

def ERI(a0,a1,a2,a3):
    eri = 0.0

    center0_x, center0_y, center0_z = a0.center
    angular0_x, angular0_y, angular0_z = a0.angular
    exponents0 = a0.exponents
    coeffs0 = a0.coefficients

    center1_x, center1_y, center1_z = a1.center
    angular1_x, angular1_y, angular1_z = a1.angular
    exponents1 = a1.exponents
    coeffs1 = a1.coefficients

    center2_x, center2_y, center2_z = a2.center
    angular2_x, angular2_y, angular2_z = a2.angular
    exponents2 = a2.exponents
    coeffs2 = a2.coefficients

    center3_x, center3_y, center3_z = a3.center
    angular3_x, angular3_y, angular3_z = a3.angular
    exponents3 = a3.exponents
    coeffs3 = a3.coefficients

    for coeff0, exponent0 in zip(coeffs0, exponents0):
        norm_coeff0 = norm_coeff(exponent0, a0.angular)
        for coeff1, exponent1 in zip(coeffs1, exponents1):
            norm_coeff1 = norm_coeff(exponent1, a1.angular)
            for coeff2, exponent2 in zip(coeffs2, exponents2):
                norm_coeff2 = norm_coeff(exponent2, a2.angular)
                for coeff3, exponent3 in zip(coeffs3, exponents3):
                    norm_coeff3 = norm_coeff(exponent3, a3.angular)

                    eri += norm_coeff0*norm_coeff1*norm_coeff2*norm_coeff3*\
                             coeff0*coeff1*coeff2*coeff3*\
                             electron_repulsion(exponent0,a0.angular,a0.center,\
                                                exponent1,a1.angular,a1.center,\
                                                exponent2,a2.angular,a2.center,\
                                                exponent3,a3.angular,a3.center)
    return eri