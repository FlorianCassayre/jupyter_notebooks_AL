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
#import chart_studio.plotly as py
#import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
plotly.offline.init_notebook_mode(connected=True)
from IPython.core.magic import register_cell_magic
from IPython.display import HTML
import ipywidgets as widgets
import random

from ipywidgets import interact, interactive, fixed, interact_manual
@register_cell_magic
def bgc(color, cell=None):
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)

    display(HTML('<img src onerror="{}">'.format(script)))
###############################################################################

## PRINTS Equations, systems, matrix

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

def printEq(coeff,b, *args):# Latex Print of one equation in format a1x1+ ...+anxn or with coefficients.
    if len(args)==1:
        n=args[0]
    else:
        n=len(coeff)
    coeff=coeff+b
    texEq='$'
    texEq=texEq+ strEq(n,coeff)
    texEq=texEq + '$'
    display(Latex(texEq))
    
def printSyst(A,b, *args):# Latex Print of one system of m equation in format ai1x1+ ...+ainxn=bi or with coeff in MatCoeff.
    if (len(args)==2) or (len(A)==len(b)):  #ensures that MatCoeff has proper dimensions 
        if len(args)==2:
            m=args[0]
            n=args[1]
        else:
            m=len(A)
            n=len(A[0])
        texSyst='$\\begin{cases}'
        Eq_list=[]
        if type(b[0])==list:
            b=np.matrix(b).astype(float)
            A=np.concatenate((A,b), axis=1)
        else:
            A = [A[i]+[b[i]]for i in range(0,m)] #becomes augmented matrix
        A=np.array(A) #just in case it's not
        for i in range(m):          
            if str(A)=='':
                Eq_i=''
                if n==1:
                    Eq_i=Eq_i + 'a_{' + str(i+1) + '1}' + 'x_1 = b_' + str(i+1)
                elif n==2:
                    Eq_i=Eq_i + 'a_{' + str(i+1) + '1}' + 'x_1 + ' +  'a_{' + str(i+1) + '2}' + 'x_2 = b_' + str(i+1)
                else:
                    Eq_i=Eq_i + 'a_{' + str(i+1) + '1}' + 'x_1 + \ldots +' + 'a_{' +  str(i+1) +str(n) + '}' + 'x_'+ str(n) + '=b_'  + str(i+1)
            else:
                Eq_i=strEq(n,A[i,:]) #attention A is (A|b)
            Eq_list.append(Eq_i)
            texSyst=texSyst+  Eq_list[i] + '\\\\'
        texSyst =texSyst+ '\\end{cases}$'
        display(Latex(texSyst))  
    else: 
        print("La matrice des coefficients n'a pas les bonnes dimensions")


def texMatrix(*args): #return tex expression of one matrix of A or (A|b) -- or (A|B) VERIFIED OK
    if len(args)==2: # matrice augmentée
        A=np.matrix(args[0]).astype(float)
        m=A.shape[1]    #??????    
        b=args[1]
        #b=[b[i] for i in range(m)]
        b=np.matrix(b).astype(float)
        A=np.concatenate((A,b), axis=1)
        texApre ='\\left(\\begin{array}{'
        texA = ''
        for i in np.asarray(A) :
            texALigne = ''
            texALigne = texALigne + str(round(i[0],3) if i[0] %1 else int(i[0]))
            if texA == '' :
                texApre = texApre + 'c'
            for j in i[1:m] :
                if texA == '' :
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j,3) if j %1 else int(j))
            if texA == '' :
                texApre = texApre + '| c'
            for j in i[m:] :   
                if texA == '' :
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j,3) if j %1 else int(j))
            texALigne = texALigne + ' \\\\'
            texA = texA + texALigne
        texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)'
    elif len(args)==1: #matrice des coefficients
        A=np.matrix(args[0]).astype(float)
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
    else: 
        print("Ce n'est pas une matrice des coefficients ni une matrice augmentée")
    return texA


def printA(*args) : #Print matrix VERIFIED OK
    texA='$'+ texMatrix(*args) + '$'
    # print(texA)
    display(Latex(texA))  


def printEquMatrices(*args): #list of matrices is M=[M1, M2, ..., Mn] where Mi=(Mi|b) 
    if len(args)==2:
        listOfMatrices=args[0]
        listOfRhS=args[1]
        texEqu='$' + texMatrix(listOfMatrices[0],listOfRhS[0])
        for i in range(1,len(listOfMatrices)):
            texEqu=texEqu+ '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i], listOfRhS[i])
        texEqu=texEqu+ '$'
        display(Latex(texEqu))
    else:
        listOfMatrices=args[0]
       # print(listOfMatrices)
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

def SolOfSyst(solution,A,b):
    A = [A[i]+[b[i]] for i in range(0,len(A))]
    A=np.array(A)
    isSol=[SolOfEq(solution, A[i,:],i+1) for i in range(0,len(A))]
    if all(isSol[i]==True for i in range(len(isSol))):
        print("C'est une solution du système")
    else:
        print("Ce n'est pas une solution du système")
        
#%%Plots using plotly
def drawLine(p, d): #p,d=vectors p-->"point" and d-->"direction",
    blue='rgb(51, 214, 255)'
    colors=[blue]
    colorscale=[[0.0,colors[0]],
               [0.1, colors[0]],
               [0.2, colors[0]],
               [0.3, colors[0]],
               [0.4, colors[0]],
               [0.5, colors[0]],
               [0.6, colors[0]],
               [0.7, colors[0]],
               [0.8, colors[0]],
               [0.9, colors[0]],
               [1.0, colors[0]]]
    vec=0.9*np.array(d)
    if len(p)==2:
        data=[]
        t=np.linspace(-5,5,51)
        s=np.linspace(0,1,10)
        trace=Scatter(x=p[0]+t*d[0],  y= p[1] + t*d[1],name='Droite')
        peak=Scatter(x= d[0],  y=  d[1],marker=dict(symbol=6,size=12,color = colors[0]),showlegend=False)
        vector=Scatter(x=p[0]+s*d[0],  y= p[1] + s*d[1],mode='lines',
                         line=dict(width = 5,
            color = colors[0]), name='Vecteur directeur')
        data.append(trace)
        data.append(vector)
        data.append(peak)
        fig = FigureWidget(data=data)
        plotly.offline.iplot(fig)
    elif len(p)==3:
        data=[
            {
            'type': 'cone',
            'x': [1], 'y': vec[1], 'z': vec[2],
            'u': d[0], 'v': d[1], 'w': d[2],
            "sizemode": "absolute",
            'colorscale': colorscale,
            'sizeref': 1,
            "showscale": False,
              'hoverinfo':'none'  
            }
            ]
        t=np.linspace(-5,5,51)
        s=np.linspace(0,1,10)
        trace=Scatter3d(x=p[0]+t*d[0],  y= p[1] + t*d[1], z=p[2]+ t*d[2], mode='lines', name='Droite')
        zero=Scatter3d(x=t*0,  y= t*0, z=t*0, name='Origine', marker=dict(size=5),showlegend=False)
        if all(dd==[0] for dd in d):
            vector=Scatter3d(x=p[0]+s*d[0],  y= p[1] + s*d[1], z=p[2]+ s*d[2], marker=dict(size=5),name='Point') 
        else:
            vector=Scatter3d(x=p[0]+s*d[0],  y= p[1] + s*d[1], z=p[2]+ s*d[2], mode='lines',
                         line=dict(width = 5,
            color = colors[0], dash='solid'), name='Vecteur directeur', hoverinfo='none')
        data.append(zero)
        data.append(vector)
        data.append(trace)
        layout = {
            'scene': {
              'camera': {
                'eye': {'x': -0.76, 'y': 1.8, 'z': 0.92}
              }
            }
        }
        fig = FigureWidget(data=data, layout=layout)
        plotly.offline.iplot(fig)    
    return fig
        

def Plot2DSys(xL,xR,p,A,b): # small values for p allows for dots to be seen
        A = [A[i]+[b[i]] for i in range(0,len(A))]
        A=np.array(A)
        t=np.linspace(xL,xR,p)
        legend=[]
        data=[]
        for i in range(1,len(A)+1):
            if (abs(A[i-1,1])) > abs (A[i-1,0]) :
                #p0=[0,A[i-1,2]/A[i-1,1]]
                #p1=[1,(A[i-1,2]-A[i-1,0])/A[i-1,1]]
                trace=go.Scatter(x=t,  y= (A[i-1,2]-A[i-1,0]*t)/A[i-1,1], name='Droite %d'%i)
            else :
                trace=go.Scatter(x=(A[i-1,2]-A[i-1,1]*t)/A[i-1,0],  y= t, name='Droite %d'%i)
            data.append(trace)
        fig = go.Figure(data=data)
        plotly.offline.iplot(fig)


        
def Plot3DSys(xL,xR,p,A,b): # small values for p allows for dots to be seen
    A = [A[i]+[b[i]] for i in range(0,len(A))]
    A=np.array(A)
    gr='rgb(102,255,102)'
    org='rgb(255,117,26)'
    red= 'rgb(255,0,0)'
    blue='rgb(51, 214, 255)'
    colors =[blue, gr, org]
    s = np.linspace(xL, xR ,p)
    t = np.linspace(xL, xR, p)
    tGrid, sGrid = np.meshgrid(s, t)
    data=[]
    for i in range(0, len(A)):
        colorscale=[[0.0,colors[i]],
               [0.1, colors[i]],
               [0.2, colors[i]],
               [0.3, colors[i]],
               [0.4, colors[i]],
               [0.5, colors[i]],
               [0.6, colors[i]],
               [0.7, colors[i]],
               [0.8, colors[i]],
               [0.9, colors[i]],
               [1.0, colors[i]]]
        j=i+1 
        if (abs(A[i,2])) > abs (A[i,1]) : # z en fonction de x,y
            x = sGrid
            y = tGrid
            surface = go.Surface(x=x, y=y, z=(A[i,3]-A[i,0]*x -A[i,1]*y)/A[i,2],
                             showscale=False, colorscale=colorscale, opacity=1, name='Plan %d' %j)
        elif (A[i,2]==0 and A[i,1]==0): # x =b
            y = sGrid
            z = tGrid
            surface = go.Surface(x=A[i,3]-A[i,1]*y, y=y, z=z,
                             showscale=False, colorscale=colorscale, opacity=1, name='Plan %d' %j)
        else:# y en fonction de x,z
            x = sGrid
            z = tGrid
            surface = go.Surface(x=x, y=(A[i,3]-A[i,0]*x -A[i,2]*z)/A[i,1], z=z,
                             showscale=False, colorscale=colorscale, opacity=1, name='Plan %d' %j)
            
        data.append(surface)
        layout = go.Layout(
        showlegend=True,  #not there WHY????
        legend=dict(orientation="h"),
        autosize=True,
        width=800,
        height=800, 
            scene=go.Scene(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
        zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    return


def Ex3Chapitre1_7():
    systa=widgets.Select(
            options=['Point', 'Droite', 'Plan', 'Incompatible'],
            description='Système a):',
            #layout=Layout(width='auto'),
            disabled=False,
        )
    systb=widgets.Select(
            options=['Point', 'Droite', 'Plan', 'Incompatible'],
            description='Système b):',
            disabled=False
        )
    systc=widgets.Select(
            options=['Point', 'Droite', 'Plan','Espace', 'Incompatible'],
            description='Système c):',
            disabled=False
        )
    systd=widgets.Select(
            options=['Point', 'Droite', 'Plan','Espace',  'Incompatible'],
            description='Système d):',
            disabled=False
        )
    choice=widgets.Dropdown(
        options=['a)', 'b)', 'c)', 'd)'],
        value='a)',
        description='Système:',
        disabled=False,
    )
    def plot(c):
        if c=='a)':
            drawLine([[0],[0]],[[4],[1]])
        if c=='b)':
            print("Le système est incompatible, donc il n'y a pas de solutions")
        if c=='c)':
            drawLine([[-17],[5],[-10]],[[0],[0],[0]])
        if c=='d)':
            drawLine([[1],[0],[0]],[[0],[-1],[1]])

    def correction(a,b,c,d): 
        if 'Droite' in a and 'Incompatible' in b and 'Point' in c and 'Droite' in d:
            print("C'est correct!")
            out=interact_manual(plot,c=choice)
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    out=interact_manual(correction,a=systa,b=systb, c=systc, d=systd)   
    
#%%Echelonnage
   
def echZero(indice, M): #echelonne la matrice pour mettre les zeros dans les lignes du bas. M (matrice ou array) et Mat (list) pas le même format.
    Mat=M[indice==False,:].ravel()
    Mat=np.concatenate([Mat,M[indice==True,:].ravel()])
    Mat=Mat.reshape(len(M), len(M[0,:])) 
    return Mat

def Eij(M, i,j): #matrice elementaire, echange la ligne i avec la ligne j
    M=np.array(M)
    M[[i,j],:]=M[[j,i],:]
    return M

def Ealpha(M, i, alpha): # matrice elementaire, multiple la ligne i par le scalaire alpha
    M=np.array(M)
    M[i,:]=alpha*M[i,:]
    return M

def Eijalpha(M, i,j, alpha): # matrice elementaire, AJOUTE à la ligne i alpha *ligne j. Attention alpha + ou -
    M=np.array(M)
    M[i,:]=M[i,:] +  alpha *M[j,:]
    return M 

def echelonMat(ech,*args):  #Nous donne la matrice echelonnée 'E' ou reduite 'ER' d'une matrice des coeffs. ou augmentée. 
    if len(args)==2: # matrice augmentée
        A=np.matrix(args[0]).astype(float)
        m=A.shape[0] 
        n=A.shape[1] 
        b=args[1]
        if type(b[0])==list:
            b=np.matrix(b).astype(float)
            A=np.concatenate((A,b), axis=1)
        else:
            b=[b[i] for i in range(m)]
            A = [A[i]+[b[i]]for i in range(0,m)]
    else: # matrice coeff
        A=np.matrix(args[0]).astype(float)
        m=A.shape[0] 
        n=A.shape[1] 
        b=np.zeros((m,1))
        A=np.concatenate((A,b), axis=1)
        
        
    if ech=='E': #Echelonnée
        Mat=np.array(A)
        Mat=Mat.astype(float) # in case the array in int instead of float.
        numPivot=0
        for i in range(len(Mat)):
            j=i
            while all(abs(Mat[j:,i])<1e-15) and j!=len(Mat[0,:])-1: #if column (or rest of) is 0, take next 
                 j+=1 
            if j==len(Mat[0,:])-1:
                if len(Mat[0,:])>j:
                    Mat[i+1:len(Mat),:]=0
                print("La matrice est sous la forme échelonnée")
                if len(args)==2:
                    printEquMatrices([A[:,0:n], Mat[:,0:n]],[A[:,n:],Mat[:,n:]])
                else:
                    printEquMatrices([A,Mat])
                break     
            if abs(Mat[i,j])<1e-15:
                    Mat[i,j]=0
                    zero=abs(Mat[i:,j])<1e-15
                    M= echZero(zero,Mat[i:,:] )
                    Mat[i:,:]=M
            Mat=Ealpha(Mat, i,1/Mat[i,j] ) #normalement Mat[i,j]!=0
            for k in range(i+1,len(A)):
                Mat=Eijalpha(Mat, k,i, -Mat[k,j])
                #Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
            numPivot+=1
            Mat[abs(Mat)<1e-15]=0
            #printA(np.array(Mat)) 
    elif ech=='ER': # Echelonnée réduite
        Mat=np.array(A)
        Mat=Mat.astype(float) # in case the array in int instead of float.
        numPivot=0
        for i in range(len(Mat)):
            j=i
            while all(abs(Mat[j:,i])<1e-15) and j!=len(Mat[0,:])-1: #if column (or rest of) is zero, take next column
                 j+=1 
            if j==len(Mat[0,:])-1:
                #ADD ZERO LINES BELOW!!!!!!
                if len(Mat[0,:])>j:
                    Mat[i+1:len(Mat),:]=0
                print("La matrice est sous la forme échelonnée")
                if len(args)==2:
                    printEquMatrices([A[:,0:n], Mat[:,0:n]],[A[:,n:], Mat[:,n:]])      
                else:
                    printEquMatrices([np.asmatrix(A), np.asmatrix(Mat)])
                break     
            if abs(Mat[i,j])<1e-15:
                    Mat[i,j]=0
                    zero=abs(Mat[i:,j])<1e-15
                    M= echZero(zero,Mat[i:,:] )
                    Mat[i:,:]=M
            Mat=Ealpha(Mat, i,1/Mat[i,j] ) #normalement Mat[i,j]!=0
            for k in range(i+1,len(A)):
                Mat=Eijalpha(Mat, k,i, -Mat[k,j])
                #Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
            numPivot+=1
            Mat[abs(Mat)<1e-15]=0
        Mat=np.array(Mat)
        MatAugm=np.concatenate((A,b), axis=1)
        # MatAugm = [A[i]+[b[i]] for i in range(0,len(A))]
        i=(len(Mat)-1)
        while i>=1:
            while all(abs(Mat[i,:len(Mat[0])-1])<1e-15) and i!=0:#if ligne (or rest of) is zero, take next ligne
                i-=1
            #we have a lign with one non-nul element
            j=i #we can start at pos ij at least the pivot is there
            if abs(Mat[i,j])<1e-15: #if element Aij=0 take next one --> find pivot
                j+=1
            #Aij!=0 and Aij==1 if echelonMat worked
            for k in range(i): #put zeros above pivot (which is 1 now)
                Mat=Eijalpha(Mat, k,i, -Mat[k,j])
            i-=1
        print("La matrice est sous la forme échelonnée réduite")
        if len(args)==2:
            printEquMatrices([A[:,0:n], Mat[:,0:n]],[A[:,n:], Mat[:,n:]])             
        else:
            printEquMatrices([A,Mat])
    return np.asmatrix(Mat)



#Generate random matrix
def randomA():
    n=random.randint(1,10)
    m=random.randint(1,10)
    A=np.empty([m,n],dtype=int)
    A=[[random.randint(-100,100) for i in range(n)] for j in range(m)]
    printA(A)
    return np.matrix(A)


def dimensionA(A):
    m=widgets.IntText(
    value=1,
    step=1,
    description='m:',
    disabled=False
    )
    n=widgets.IntText(
        value=1,
        step=1,
        description='n:',
        disabled=False
    )

    display(m)
    display(n)
    def f():
        if m.value==A.shape[0] and n.value==A.shape[1]:
            print('Correcte!')
        else:
            print('Incorrecte, entrez de nouvelles valeurs')
    out=interact_manual(f)


def manualEch(*args):
    if len(args)==2: # matrice augmentée
        A=np.matrix(args[0]).astype(float)
        m=A.shape[0]        
        b=args[1]
        #b=[b[i] for i in range(m)]
        if type(b[0])==list:
            b=np.matrix(b).astype(float)
            A=np.concatenate((A,b), axis=1)
        else:
            b=[b[i] for i in range(m)]
            A = [A[i]+[b[i]]for i in range(0,m)]
    else:
        A=np.matrix(args[0]).astype(float)
        m=A.shape[0]
    A=np.array(A) #just in case it's not
    j=widgets.BoundedIntText(
    value=1,
    min=1,
    max=m,
    step=1,
    description='Ligne j:',
    disabled=False
    )
    i=widgets.BoundedIntText(
        value=1,
        min=1,
        max=m,
        step=1,
        description='Ligne i:',
        disabled=False
    )

    r=widgets.RadioButtons(
        options=['Eij', 'Ei(alpha)', 'Eij(alpha)'],
        description='Opération:',
        disabled=False
    )


    alpha=widgets.Text(
        value='1',
        description='Coeff. alpha:',
        disabled=False
    )
    print("Régler les paramètres et évaluer la cellule suivante")
    print("Répéter cela jusqu'à obtenir une forme échelonnée réduite")
    display(r)
    display(i)
    display(j)
    display(alpha)
    return i,j,r,alpha

def echelonnage(i,j,r,alpha,A,m,*args): #1.5-1.6 Matrice echelonnées
    if alpha.value==0:
        print('Le coefficient alpha doit être non-nul!')
    if r.value=='Eij':
        m=Eij(m,i.value-1,j.value-1)
    if r.value=='Ei(alpha)':
        m=Ealpha(m,i.value-1,eval(alpha.value))
    if r.value=='Eij(alpha)':
        m=Eijalpha(m,i.value-1,j.value-1,eval(alpha.value))    
    if len(args)==2: 
        A=np.asmatrix(A)
        MatriceList=args[0]
        RhSList=args[1]
        MatriceList.append(m[:,0:A.shape[1]])#??????????
        RhSList.append(m[:,A.shape[1]:])#??????????
        printEquMatrices(MatriceList,RhSList)
    else:
        MatriceList=args[0]
        A=np.asmatrix(A)
        MatriceList.append(m[:,0:A.shape[1]]) #??????????
        printEquMatrices(MatriceList)   
    return m
    
def manualOp(*args):
    if len(args)==2: # matrice augmentée
        A=np.matrix(args[0]).astype(float)
        m=A.shape[0]        
        b=args[1]
        #b=[b[i] for i in range(m)]
        if type(b[0])==list:
            b=np.matrix(b).astype(float)
            A=np.concatenate((A,b), axis=1)
        else:
            b=[b[i] for i in range(m)]
            A = [A[i]+[b[i]]for i in range(0,m)]
    else:
        A=np.matrix(args[0]).astype(float)
        m=A.shape[0]
    A=np.array(A) #just in case it's not
    j=widgets.BoundedIntText(
    value=1,
    min=1,
    max=m,
    step=1,
    description='Ligne j:',
    disabled=False
    )
    i=widgets.BoundedIntText(
        value=1,
        min=1,
        max=m,
        step=1,
        description='Ligne i:',
        disabled=False
    )

    r=widgets.RadioButtons(
        options=['Eij', 'Ei(alpha)', 'Eij(alpha)'],
        description='Opération:',
        disabled=False
    )


    alpha=widgets.Text(
        value='1',
        description='Coeff. alpha:',
        disabled=False
    )
    print("Régler les paramètres et cliquer sur RUN INTERACT pour effectuer votre opération")
    def f(r,i,j,alpha):
        m=np.concatenate((A,b), axis=1)
        MatriceList=[A]
        RhSList=[b]    
        if alpha==0:
            print('Le coefficient alpha doit être non-nul!')
        if r=='Eij':
            m=Eij(m,i-1,j-1)
        if r=='Ei(alpha)':
            m=Ealpha(m,i.value-1,eval(alpha))
        if r=='Eij(alpha)':
            m=Eijalpha(m,i-1,j-1,eval(alpha))
        MatriceList.append(m[:,0:len(A[0])])
        RhSList.append(m[:,len(A[0]):])    
        printEquMatricesAug(MatriceList,RhSList)
    interact_manual(f,r=r,i=i,j=j,alpha=alpha)
    



########################################OBSOLETE
def printEquMatricesAug(listOfMatrices, listOfRhS): #list of matrices is M=[M1, M2, ..., Mn] where Mi=(Mi|b) 
    texEqu='$' + texMatrixAug(listOfMatrices[0],listOfRhS[0])
    for i in range(1,len(listOfMatrices)):
        texEqu=texEqu+ '\\quad \\sim \\quad' + texMatrixAug(listOfMatrices[i], listOfRhS[i])
    texEqu=texEqu+ '$'
    display(Latex(texEqu))

def echelonMatCoeff(A): #take echelonMAt but without b. 
    b= [0 for i in range(len(A))]
    Mat = [A[i]+[b[i]] for i in range(0,len(A))]
    Mat=np.array(Mat)
    Mat=Mat.astype(float) # in case the array in int instead of float.
    numPivot=0
    for i in range(len(Mat)):
        j=i
        while all(abs(Mat[j:,i])<1e-15) and j!=len(Mat[0,:])-1: #if column (or rest of) is zero, take next column
             j+=1 
        if j==len(Mat[0,:])-1:
            #ADD ZERO LINES BELOW!!!!!!
            if len(Mat[0,:])>j:
                Mat[i+1:len(Mat),:]=0
            print("La matrice est sous la forme échelonnée")
            printEquMatrices(np.asmatrix(A), np.asmatrix(Mat[:, :len(A[0])]))
            break     
        if abs(Mat[i,j])<1e-15:
                Mat[i,j]=0
                zero=abs(Mat[i:,j])<1e-15
                M= echZero(zero,Mat[i:,:] )
                Mat[i:,:]=M
        Mat=Ealpha(Mat, i,1/Mat[i,j] ) #normalement Mat[i,j]!=0
        for k in range(i+1,len(A)):
            Mat=Eijalpha(Mat, k,i, -Mat[k,j])
            #Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
        numPivot+=1
        Mat[abs(Mat)<1e-15]=0
        #printA(np.asmatrix(Mat[:, :len(A[0])])) 
    return np.asmatrix(Mat)

def echelonRedMat(A, b):
    Mat=echelonMat('ER',A,b)
    Mat=np.array(Mat)
    MatAugm=np.concatenate((A,b), axis=1)
   # MatAugm = [A[i]+[b[i]] for i in range(0,len(A))]
    i=(len(Mat)-1)
    while i>=1:
        while all(abs(Mat[i,:len(Mat[0])-1])<1e-15) and i!=0:#if ligne (or rest of) is zero, take next ligne
            i-=1
        #we have a lign with one non-nul element
        j=i #we can start at pos ij at least the pivot is there
        if abs(Mat[i,j])<1e-15: #if element Aij=0 take next one --> find pivot
            j+=1
        #Aij!=0 and Aij==1 if echelonMat worked
        for k in range(i): #put zeros above pivot (which is 1 now)
            Mat=Eijalpha(Mat, k,i, -Mat[k,j])
        i-=1
        printA(Mat)
    print("La matrice est sous la forme échelonnée réduite")
    printEquMatrices(MatAugm, Mat)
    return np.asmatrix(Mat)


def printEquMatricesOLD(listOfMatrices): #list of matrices is M=[M1, M2, ..., Mn]
    texEqu='$' + texMatrix(listOfMatrices[0])
    for i in range(1,len(listOfMatrices)):
        texEqu=texEqu+ '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i])
    texEqu=texEqu+ '$'
    display(Latex(texEqu))
def texMatrixAug(A,b): #return tex expression of one matrix (A|b) where b can also be a matrix
    m=len(A[0])
    A=np.concatenate((A,b), axis=1)
    texApre ='\\left(\\begin{array}{'
    texA = ''
    for i in np.asarray(A) :
        texALigne = ''
        texALigne = texALigne + str(round(i[0],3) if i[0] %1 else int(i[0]))
        if texA == '' :
            texApre = texApre + 'c'
        for j in i[1:m] :
            if texA == '' :
                texApre = texApre + 'c'
            texALigne = texALigne + ' & ' + str(round(j,3) if j %1 else int(j))
        if texA == '' :
            texApre = texApre + '| c'
        for j in i[m:] :   
            if texA == '' :
                texApre = texApre + 'c'
            texALigne = texALigne + ' & ' + str(round(j,3) if j %1 else int(j))
        texALigne = texALigne + ' \\\\'
        texA = texA + texALigne
    texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)'
    return texA
def printAAug(A,b) : #Print matrix (A|b)
    texA='$'+ texMatrixAug(A,b) + '$'
    # print(texA)
    display(Latex(texA))  

    
def printEquMatricesOLD(*args): # M=[M1, M2, ..., Mn] n>1 VERIFIED OK
    texEqu='$' + texMatrix(args[0])
    for i in range(1,len(args)):
        texEqu=texEqu+ '\\quad \\sim \\quad' + texMatrix(args[i])
    texEqu=texEqu+ '$'
    display(Latex(texEqu))