#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:42:29 2019

@author: jecker
"""

def printEq(n,coeff):# Latex Print of one equation in format a1x1+ ...+anxn or with coefficients.
    textEq='$'
    if coeff=='': 
        if n==1:
            textEq=textEq + 'a_1x_1 = b'
        elif n==2:
            textEq=textEq + 'a_1x_1 + a_2x_2 = b'
        else:
            textEq=textEq + 'a_1x_1 + \ldots + ' + 'a_' + str(n) + 'x_'+ str(n) + '=b'
    else :
         textEq=textEq + str(coeff[0] if coeff[0] % 1 else int(coeff[0]))+ 'x_'+str(1)
         for i in range(1,n-1):
            textEq=textEq + ' + ' + str(coeff[i] if coeff[i] %1 else int(coeff[i])) + 'x_' + str(i+1) 
         textEq=textEq +  ' + ' +str(coeff[len(coeff)-2] if coeff[len(coeff)-2] % 1 else int(coeff[len(coeff)-2])) + 'x_' + str(n) + '=' + str(coeff[len(coeff)-1] if coeff[len(coeff)-1] % 1 else int(coeff[len(coeff)-1])) 
    texEq =textEq+ '$'
    # print(texEq)
    display(Latex(texEq))
    
    

def EnterInt(n): #function enter integer. 
    while type(n)!=int:
        try:
            n=int(n)
            if n<=0:
                print("Le nombre ne peut pas être négatif!") 
                print("Entrez à nouveau : ")
                n=input()
        except:
            print("Ce n'est pas un entier!")
            print("Entrez à nouveau :")
            n=input()
    n=int(n)
    return n

def EnterListReal(n,coeff): #function enter list of real numbers.
    while type(coeff)!=list:
        try: 
            coeff=[float(x) for x in coeff.split(',')]   
            if len(coeff)!=n+1:
                print("Vous n'avez pas entré le bon nombre de réels!") 
                print("Entrez à nouveau : ")
                coeff=input() 
        except:
            print("Ce n'est pas le bon format!")
            print("Entrez à nouveau")
            coeff=input() 
    return coeff

def SolOfEq(sol,coeff): #Verify if sol is the solution of the equation with coefficients coeff
    A = np.asarray(coeff[0:len(coeff)-1])
    if np.dot(A,sol)==coeff[len(coeff)-1]:
        print("La suite entrée est une solution de votre équation!")
    else:
        print("La suite entrée n'est pas une solution de votre équation!")
