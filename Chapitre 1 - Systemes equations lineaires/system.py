#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:53:28 2019

@author: jecker
"""
import AL_Fct as al
import matplotlib.pyplot as plt


MatCoeff=np.array([[1, 2,0], [3, -4,9], [3, -4,5]])
al.printSyst(len(MatCoeff), len(MatCoeff[0,:])-1,MatCoeff)
al.printSyst(4,5,'')


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


print("Entrez la solution sous la forme d'une suite de ", n," nombres réels")
entry=input()
sol=al.EnterListReal(n-1,entry)

sol=np.asarray(sol[0:len(sol)])
isSol=[al.SolOfEq(sol, MatCoeff[i,:],i+1) for i in range(0,m)]
if all(isSol[i]==True for i in range(len(isSol))):
    print("C'est une solution du système")
else:
    print("Ce n'est pas une solution du système")

x=np.arange(-5,5,0.1)
plt.plot(x, 3-x)
plt.show()