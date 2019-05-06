#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:42:29 2019

@author: jecker
"""
from __future__ import division

import numpy as np
from IPython.display import display, Latex
import matplotlib.pyplot as plt
import math
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
from IPython.core.magic import register_cell_magic
from IPython.display import HTML, display

@register_cell_magic
def bgc(color, cell=None):
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)

    display(HTML('<img src onerror="{}">'.format(script)))



#PRINTS Equations, systems, matrix
def strEq(n,coeff):# Latex str of one equation in format a1x1+ ...+anxn or with coefficients.
    Eq=''
    if str(coeff)=='' or coeff==[]: 
        if n==1:
            Eq=Eq + 'a_1x_1 = b'
        elif n==2:
            Eq=Eq + 'a_1x_1 + a_2x_2 = b'
        else:
            Eq=Eq + 'a_1x_1 + \ldots + ' + 'a_' + str(n) + 'x_'+ str(n) + '=b'
    else :
        if n==1:
            Eq=Eq + str(round(coeff[0],3) if coeff[0] % 1 else int(coeff[0]))+ 'x_'+str(1) +\
            '='+ str(round(coeff[len(coeff)-1],3) if coeff[len(coeff)-1] % 1 else int(coeff[len(coeff)-1])) 
        else:
             Eq=Eq + str(round(coeff[0],3) if coeff[0] % 1 else int(coeff[0]))+ 'x_'+str(1)
             for i in range(1,n-1):
                Eq=Eq + ' + ' + str(round(coeff[i],3) if coeff[i] %1 else int(coeff[i])) + 'x_' + str(i+1) 
             Eq=Eq +  ' + ' +str(round(coeff[len(coeff)-2],3) if coeff[len(coeff)-2] % 1 else int(coeff[len(coeff)-2]))\
             + 'x_' + str(n) + '=' + str(round(coeff[len(coeff)-1],3) if coeff[len(coeff)-1] % 1 else int(coeff[len(coeff)-1])) 
    return Eq

def printEq(n,coeff):# Latex Print of one equation in format a1x1+ ...+anxn or with coefficients.
    texEq='$'
    texEq=texEq+ strEq(n,coeff)
    texEq=texEq + '$'
    display(Latex(texEq))
    
def printSyst(m,n,MatCoeff):# Latex Print of one system of m equation in format ai1x1+ ...+ainxn=bi or with coeff in MatCoeff.
    texSyst='$\\begin{cases}'
    Eq_list=[]
    MatCoeff=np.array(MatCoeff) #just in case it's not
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
            Eq_i=strEq(n,MatCoeff[i,:])
        Eq_list.append(Eq_i)
        texSyst=texSyst+  Eq_list[i] + '\\\\'
    texSyst =texSyst+ '\\end{cases}$'
    display(Latex(texSyst))  


def texMatrix(A): #return tex expression of one matrix
    texApre ='\\left(\\begin{array}{'
    texA = ''
    for i in np.asarray(A) :
        texALigne = ''
        texALigne = texALigne + str(round(i[0],3) if i[0] %1 else int(i[0]))
        if texA == '' :
            texApre = texApre + 'c'
        for j in i[1:] :
            if texA == '' :
                texApre = texApre + 'c'
            texALigne = texALigne + ' & ' + str(round(j,3) if j %1 else int(j))
        texALigne = texALigne + ' \\\\'
        texA = texA + texALigne
    texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)'
    return texA

def printA(A) : #Print matrix 
    texA='$'+ texMatrix(A) + '$'
    # print(texA)
    display(Latex(texA))  
    
def printEquMatrices(listOfMatrices): #list of matrices is M=[M1, M2, ..., Mn]
    texEqu='$' + texMatrix(listOfMatrices[0])
    for i in range(1,len(listOfMatrices)):
        texEqu=texEqu+ '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i])
    texEqu=texEqu+ '$'
    display(Latex(texEqu))

#%% Functions to enter smth

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

def EnterListReal(n): #function enter list of real numbers.
    coeff=input()
    while type(coeff)!=list:
        try: 
            coeff=[float(eval(x)) for x in coeff.split(',')]   
            if len(coeff)!=n+1:
                print("Vous n'avez pas entré le bon nombre de réels!") 
                print("Entrez à nouveau : ")
                coeff=input() 
        except:
            print("Ce n'est pas le bon format!")
            print("Entrez à nouveau")
            coeff=input() 
    #coeff[abs(coeff)<1e-15]=0 #ensures that 0 is 0.  
    return coeff

#%%Verify if sol is the solution of the equation with coefficients coeff
def SolOfEq(sol,coeff, i):
    A = np.asarray(coeff[0:len(coeff)-1])
    if abs(np.dot(A,sol)-coeff[len(coeff)-1])<1e-10:
        print("La suite entrée est une solution de l'équation", i)
    else:
        print("La suite entrée n'est pas une solution de l'équation",i)
    return abs(np.dot(A,sol)-coeff[len(coeff)-1])<1e-10

def SolOfSyst(solution,MatriceCoeff,m):
    MatriceCoeff=np.array(MatriceCoeff)
    isSol=[SolOfEq(solution, MatriceCoeff[i,:],i+1) for i in range(0,m)]
    if all(isSol[i]==True for i in range(len(isSol))):
        print("C'est une solution du système")
    else:
        print("Ce n'est pas une solution du système")
        
#%%Plots using plotly
def Plot2DSys(xL,xR,p,MatCoeff): # small values for p allows for dots to be seen
    MatCoeff=np.array(MatCoeff)
    x=np.linspace(xL,xR,p)
    legend=[]
    data=[]
    for i in range(1,len(MatCoeff)+1):
            trace=go.Scatter(x=x,  y= (MatCoeff[i-1,2]-MatCoeff[i-1,0]*x)/MatCoeff[i-1,1], name='Droite %d'%i)
            data.append(trace)
    fig = go.Figure(data=data)
    plotly.offline.iplot(fig)
    
def Plot3DSys(xL,xR,p,MatCoeff): # small values for p allows for dots to be seen
    MatCoeff=np.array(MatCoeff)
    x=np.linspace(xL,xR,p)
    legend=[]
    data=[]
    for i in range(1,len(MatCoeff)+1):
            trace=go.Scatter(x=x,  y= (MatCoeff[i-1,2]-MatCoeff[i-1,0]*x)/MatCoeff[i-1,1], name='Droite %d'%i)
            data.append(trace)
    fig = go.Figure(data=data)
    plotly.offline.iplot(fig) 
    
#%%Echelonnage
   
def echZero(indice, M): #echelonne la matrice pour mettre les zeros dans les lignes du bas. M (matrice ou array) et Mat (list) pas le même format.
    Mat=M[indice==False,:].ravel()
    Mat=np.concatenate([Mat,M[indice==True,:].ravel()])
    Mat=Mat.reshape(len(M), len(M[0,:])) 
    return Mat

def Eij(M, i,j): #matrice elementaire, echange la ligne i avec la ligne j
    M[[i,j],:]=M[[j,i],:]
    return M

def Ealpha(M, i, alpha): # matrice elementaire, multiple la ligne i par le scalaire alpha
    M[i,:]=alpha*M[i,:]
    return M

def Eijalpha(M, i,j, alpha): # matrice elementaire, AJOUTE à la ligne i alpha *ligne j. Attention alpha + ou -
    M[i,:]=M[i,:] +  alpha *M[j,:]
    return M 

def echelonMat(MatCoeff):
    Mat=np.array(MatCoeff)
    Mat=Mat.astype(float) # in case the array in int instead of float.

    numPivot=0
    for i in range(len(Mat)):
        j=i
        while all(abs(Mat[j:,i])<1e-15) and j!=len(Mat[0,:])-1: #if column (or rest of) is zero, take next column
             j+=1 
        if j==len(Mat[0,:])-1:
            print("La matrice est sous la forme échelonnée")
            printEquMatrices([MatCoeff, Mat])
            break     
        if abs(Mat[i,j])<1e-15:
                Mat[i,j]=0
                zero=abs(Mat[i:,j])<1e-15
                M= echZero(zero,Mat[i:,:] )
                Mat[i:,:]=M
        Mat=Ealpha(Mat, i,1/Mat[i,j] ) #normalement Mat[i,i]!=0
        for k in range(i+1,len(MatCoeff)):
            Mat=Eijalpha(Mat, k,i, -Mat[k,j])
            #Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
    
        numPivot+=1
        Mat[abs(Mat)<1e-15]=0
        printA(np.asmatrix(Mat))    
