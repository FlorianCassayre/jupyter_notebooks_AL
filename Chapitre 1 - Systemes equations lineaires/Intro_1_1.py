#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:13:39 2019

@author: jecker
"""
import numpy as np

import AL_newFct as al

print("Entrez le nombre de variables n=") #function EnterInt but we might need to specify what int we want: variable, nbr equation,..
n=input()
n=al.EnterInt(n)

print("Votre équation est de la forme")
al.printEq(n, '')

print("Entrez les ", n+1," coefficients de l'équations sous le format a1, a2, ..., b")
entry=input()
coeff=al.EnterListReal(n,entry)

print("Votre equation est")
al.printEq(n, coeff)
      
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Entrez la solution sous la forme d'une suite de ", n," nombres réels")
entry=input()
sol=al.EnterListReal(n-1,entry)

sol=np.asarray(sol[0:len(sol)])
al.SolOfEq(sol, coeff)
#%%%%%%%%%%
    
   






