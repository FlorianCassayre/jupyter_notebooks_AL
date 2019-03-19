#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:14:28 2019

@author: jecker
"""

import AL_Fct as al
import numpy as np
import matplotlib.pyplot as plt

print("Entrez le nombre de variables n=") 
n=input()
n=al.EnterInt(n)
print("Entrez le nombre d'équations m=") 
m=input()
m=al.EnterInt(m)
print("Votre système est de la forme")
al.printSyst(m,n, '')



MatCoeff=[[0] * n for i in range(m) ]
for i in range(1,m+1):
    print("Entrez les ", n+1," coefficients de l'équation", i)
    entry=input()
    MatCoeff[i-1]=al.EnterListReal(n,entry)
MatCoeff=np.array(MatCoeff)
print("Votre système est de la forme")
al.printSyst(m,n, MatCoeff)

print("La matrice correspondante est")
al.printA(MatCoeff)

al.echelonMatA(MatCoeff)