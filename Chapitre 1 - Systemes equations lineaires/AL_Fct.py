#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:42:29 2019

@author: jecker
"""
import numpy as np
from IPython.display import display, Latex

def Eq(n,coeff):# Latex str of one equation in format a1x1+ ...+anxn or with coefficients.
    Eq=''
    if str(coeff)=='': 
        if n==1:
            Eq=Eq + 'a_1x_1 = b'
        elif n==2:
            Eq=Eq + 'a_1x_1 + a_2x_2 = b'
        else:
            Eq=Eq + 'a_1x_1 + \ldots + ' + 'a_' + str(n) + 'x_'+ str(n) + '=b'
    else :
        if n==1:
            Eq=Eq + str(coeff[0] if coeff[0] % 1 else int(coeff[0]))+ 'x_'+str(1) + '='+ str(coeff[len(coeff)-1] if coeff[len(coeff)-1] % 1 else int(coeff[len(coeff)-1])) 
        else:
             Eq=Eq + str(coeff[0] if coeff[0] % 1 else int(coeff[0]))+ 'x_'+str(1)
             for i in range(1,n-1):
                Eq=Eq + ' + ' + str(coeff[i] if coeff[i] %1 else int(coeff[i])) + 'x_' + str(i+1) 
             Eq=Eq +  ' + ' +str(coeff[len(coeff)-2] if coeff[len(coeff)-2] % 1 else int(coeff[len(coeff)-2])) + 'x_' + str(n) + '=' + str(coeff[len(coeff)-1] if coeff[len(coeff)-1] % 1 else int(coeff[len(coeff)-1])) 
    return Eq

def printEq(n,coeff):# Latex Print of one equation in format a1x1+ ...+anxn or with coefficients.
    texEq='$'
    texEq=texEq+ Eq(n,coeff)
    texEq=texEq + '$'
    display(Latex(texEq))
    
def printSyst(m,n,MatCoeff):# Latex Print of one system of m equation in format ai1x1+ ...+ainxn=bi or with coeff in MatCoeff.
    textSyst='$\\begin{cases}'
    Eq_list=[]
    for i in range(m):          
        if str(MatCoeff)=='':
            Eq_i=''
            if n==1:
                Eq_i=Eq_i + 'a_{' + str(i+1) + '1}' + 'x_1 = b_' + str(i+1)
            elif n==2:
                Eq_i=Eq_i + 'a_{' + str(i+1) + '1}' + 'x_1 + ' +  'a_{' + str(i+1) + '2}' + 'x_2 = b_' + str(i+1)
            else:
                Eq_i=Eq_i + 'a_{' + str(i+1) + '1}' + 'x_1 + \ldots +' + 'a_{' +  str(i+1) +str(n) + '}' + 'x_'+ str(n) + '=b_'  + str(i+1)
        else:
            Eq_i=Eq(n,MatCoeff[i,:])
        Eq_list.append(Eq_i)
        textSyst=textSyst+  Eq_list[i] + '\\\\'
    texSyst =textSyst+ '\\end{cases}$'
    display(Latex(texSyst))  

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

def SolOfEq(sol,coeff, i): #Verify if sol is the solution of the equation with coefficients coeff
    A = np.asarray(coeff[0:len(coeff)-1])
    if np.dot(A,sol)==coeff[len(coeff)-1]:
        print("La suite entrée est une solution de l'équation", i)
    else:
        print("La suite entrée n'est pas une solution de l'équation",i)
    return np.dot(A,sol)==coeff[len(coeff)-1]