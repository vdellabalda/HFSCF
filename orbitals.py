#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

orbitaltypes = ["S", "P", "SP"]

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
        self.angular = np.array(angular)
        self.exponents = [0.0 for i in range(contraction)]
        self.coefficients = [0.0 for i in range(contraction)]

    def __call__(self, x, y, z):
        value = 0
        x0, y0, z0 = self.center
        lx, ly, lz = self.angular
        for idx, coefficient in enumerate(self.coefficients):
            exponent = self.exponents[idx]
            g1dx = Gaussian1d(x0, lx, exponent)
            g1dy = Gaussian1d(y0, ly, exponent)
            g1dz = Gaussian1d(z0, lz, exponent)
            value += coefficient*g1dx(x)*g1dy(y)*g1dz(z)
        return value


def orbitaltype_to_quantumnumber(orbitaltype):
    if orbitaltype == "S":
        return 0, 0, 0
    if orbitaltype == "Px":
        return 1, 0, 0
    if orbitaltype == "Py":
        return 0, 1, 0
    if orbitaltype == "Pz":
        return 0, 0, 1
    if orbitaltype == "Dx2":
        return 2, 0, 0
    if orbitaltype == "Dy2":
        return 0, 2, 0
    if orbitaltype == "Dz2":
        return 0, 0, 2
    if orbitaltype == "Dxy":
        return 1, 1, 0
    if orbitaltype == "Dyz":
        return 0, 1, 1
    if orbitaltype == "Dzx":
        return 1, 0, 1
    
def quantumnumber_to_orbitaltype(angular):
    if angular == (0,0,0):
        return "S"
    if angular == (1,0,0):
        return "Px"
    if angular == (0,1,0):
        return "Py"
    if angular == (0,0,1):
        return "Pz"
    if angular == (2,0,0):
        return "Dx2"
    if angular == (0,2,0):
        return "Dy2"
    if angular == (0,0,2):
        return "Dz2"
    if angular == (1,1,0):
        return "Dxy"
    if angular == (0,1,1):
        return "Dyz"
    if angular == (1,0,1):
        return "Dzx"   


def create_ao_dict(basis_set_filename):
    ao_dict = dict()

    new_atom = True
    atom = False

    
    with open(basis_set_filename, "r") as f:
        line = f.readline().strip().split()
        while line:
            if new_atom:
                ao_dict[line[0]] = []
                atomtype = line[0]
                new_atom = False
                atom = True
                line = f.readline().strip().split()
            while atom:
                if line[0] in orbitaltypes:
                    orbitaltype = line[0]
                    exponents = []
                    s_coefficients = []
                    p_coefficients = []
                    primitive_count = int(line[1])
                    if orbitaltype == "S":
                        angular = orbitaltype_to_quantumnumber("S")
                        aos = Ao(np.zeros(3), angular, primitive_count)
                    elif orbitaltype == "SP":
                        angs = orbitaltype_to_quantumnumber("S")
                        angpx = orbitaltype_to_quantumnumber("Px")
                        angpy = orbitaltype_to_quantumnumber("Py")
                        angpz = orbitaltype_to_quantumnumber("Pz")
                        aos = Ao(np.zeros(3), angs, primitive_count)
                        aopx = Ao(np.zeros(3), angpx, primitive_count)
                        aopy = Ao(np.zeros(3), angpy, primitive_count)
                        aopz = Ao(np.zeros(3), angpz, primitive_count)
                    line = f.readline().strip().split()

                line_count = 0
                while line_count < primitive_count:
                    if orbitaltype == "S":
                        exponents.append(float(line[0].replace('D', 'E')))
                        s_coefficients.append(float(line[1].replace('D', 'E')))
                    elif orbitaltype == "SP":
                        exponents.append(float(line[0].replace('D', 'E')))
                        s_coefficients.append(float(line[1].replace('D', 'E')))
                        p_coefficients.append(float(line[2].replace('D', 'E')))
                    line = f.readline().strip().split()
                    line_count += 1

                if orbitaltype == "S":
                    aos.coefficients = np.array(s_coefficients)
                    aos.exponents = np.array(exponents)
                    ao_dict[atomtype].append(aos)

                elif orbitaltype == "SP":
                    aos.coefficients = np.array(s_coefficients)
                    aos.exponents = np.array(exponents)
                    aopx.coefficients = np.array(p_coefficients)
                    aopx.exponents = np.array(exponents)
                    aopy.coefficients = np.array(p_coefficients)
                    aopy.exponents = np.array(exponents)
                    aopz.coefficients = np.array(p_coefficients)
                    aopz.exponents = np.array(exponents)
                    ao_dict[atomtype].append(aos)
                    ao_dict[atomtype].append(aopx)
                    ao_dict[atomtype].append(aopy)
                    ao_dict[atomtype].append(aopz)

                if line[0] == "****":
                    atom = False
                    new_atom = True
                    line = f.readline().strip().split()

    return ao_dict

# ================================== 
#
#    Main
# 
# ==================================

def main():
    return


if __name__== "__main__":
    main()