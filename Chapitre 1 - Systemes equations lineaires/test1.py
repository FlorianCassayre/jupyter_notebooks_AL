#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:13:39 2019

@author: jecker
"""
import numpy as np

def printEq(n,coeff):
    textEq='$'
    if coeff=='': # equation is a1x1+ ...+anxn
        if n==1:
            textEq=textEq + 'a_1x_1'
        elif n==2:
            textEq=textEq + 'a_1x_1 + a_2x_2'
        else:
            textEq=textEq + 'a_1x_1 + \ldots + ' + 'a_' + str(n) + 'x_'+ str(n) + '=b'
    else :
         for i in range(0,n-1):
            textEq=textEq + str(coeff[i]) + 'x_' + str(i+1) +' + '
         textEq=textEq + str(coeff[len(coeff)-2]) + 'x_' + str(n) + '=' + str(coeff[len(coeff)-1]) 
    texEq =textEq+ '$'
    # print(texEq)
    display(Latex(texEq))
    
    

print("Entrez le nombre variables n=")
n=input()
while type(n)!=int:
    try:
        n=int(n)
        if n<=0:
            print("Le nombre de variable ne peut pas être négatif!") 
            print("Entrez le nombre variables n=")
            n=input()
    except:
        print("Ce n'est pas un entier!")
        print("Entrez le nombre variables n=")
        n=input()

n=int(n)
print("Votre équation est de la forme")

printEq(n, '')

coeff=''
while type(coeff)!=list and len(coeff)!=n+1:
    print("Entrez les ", n," coefficients de l'équations sous le format a1, a2, ..., b")
    entry=input()
    try: 
        coeff=[int(x) for x in entry.split(',')]     
    except:
        print("Les coefficients ne sont pas dans le bon format ou vous n'avez pas entré le bon nombre de coefficients!")
 
   
print("Votre equation est")
printEq(n, coeff)
      
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sol=''
while type(sol)!=list and len(sol)!=n:
    print("Entrez la solution sous la forme d'une suite de ", n," nombres réels")
    entry=input()
    try: 
        sol=[int(x) for x in entry.split(',')]     
    except:
        print("La suite n'est pas dans le bon format ou n'est pas de la bonne longueur!")

sol=np.asarray(sol[0:len(sol)])
A = np.asarray(coeff[0:len(coeff)-1])

if np.dot(A,sol)==coeff[len(coeff)-1]:
    print("La suite entrée est une solution de votre équation!")
else:
    print("La suite entrée n'est pas une solution de votre équation!")

 
   






