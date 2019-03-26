#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:13:39 2019

@author: jecker
"""

import AL_Fct as al
import math
#print("Entrez le nombre de variables n=") #function EnterInt but we might need to specify what int we want: variable, nbr equation,..
#n=input()
#n=al.EnterInt(n)
#
#print("Votre équation est de la forme")
#al.printEq(n, '')
#
#def EnterListReal(n): #function enter list of real numbers.
#    coeff=input()
#    while type(coeff)!=list:
#        try: 
#            coeff=[float(eval(x)) for x in coeff.split(',')]   
#            if len(coeff)!=n+1:
#                print("Vous n'avez pas entré le bon nombre de réels!") 
#                print("Entrez à nouveau : ")
#                coeff=input() 
#        except:
#            print("Ce n'est pas le bon format!")
#            print("Entrez à nouveau")
#            coeff=input() 
#    #coeff[abs(coeff)<1e-15]=0 #ensures that 0 is 0.  
#    return coeff
#
#print("Entrez les ", n+1," coefficients de l'équations sous le format a1, a2, ..., b")
#coeff=al.EnterListReal(n)
#
#print("Votre equation est")
#al.printEq(n, coeff)
#      
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#print("Entrez la solution sous la forme d'une suite de ", n," nombres réels")
#entry=input()
#sol=al.EnterListReal(n-1,entry)
#
#sol=np.asarray(sol[0:len(sol)])
#al.SolOfEq(sol, coeff,1)
#%%%%%%%%%%
#OK works with fractions as well.
   




MatCoeff =[ [1, -3,4] , [-1,4,5] ]
MatCoeff=np.array(MatCoeff)

al.printSyst(m,n, MatCoeff)
