#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:12:37 2019

@author: jecker
"""
import AL_Fct as al

#ligne=len(MatCoeff), colonne= len(MatCoeff[0,:])


#%% 
MatCoeff=np.array([[0, 2,1,2], [3, -4,10,8], [3, -4,10,8]])

#A = np.asmatrix(MatCoeff)
al.printA(MatCoeff)

al.echelonMat(MatCoeff)

#%%TRASH
#def diviseLgn(coeff, Mat, i,j):
#    Mat[i,:]=Mat[i,:]/Mat[i,j] # Attention si Mat[i,j] est zero...il faut prendre la colonne d aprés. Add exception
#    for k in range(1,len(MatCoeff)):
#        Mat[i,:]=Mat[i,:]-Mat[i,j]*Mat[i,:]#-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        
#Mat[0,:]=Mat[0,:]/Mat[0,0] 
#for i in range(1,len(MatCoeff)):
#    Mat[i,:]=Mat[i,:]-Mat[i,0]*Mat[0,:]
#al.printA(np.asmatrix(Mat)) #deviendra une fonction 
#   
##on ne touche plus Mat[0,:] 1ere ligne ----> loop sur j les colonnes!!
#zero=Mat[1:len(Mat),1]==0
#M= ech_zero(zero,Mat[1:len(Mat),:] )
#
#Mat[1:len(Mat),:]=M
#al.printA(np.asmatrix(Mat))
#diviseLgn(coeff, M, i,j)
#
#if MatCoeff[0,]
#MatCoeff[0,:]
#
#def ech_zero(indice, M): #echelonne la matrice pour mettre les zeros dans les lignes du bas. M (matrice ou array) et Mat (list) pas le même format.
#    Mat=M[indice==False,:].ravel()
#    Mat=np.concatenate([Mat,M[indice==True,:].ravel()])
#    Mat=Mat.reshape(len(M), len(M[0,:])) 
#    return Mat
#
#def Eij(M, i,j): #matrice elementaire, echange la ligne i avec la ligne j
#    M[[i,j],:]=M[[j,i],:]
#    return M
#
#def Ealpha(M, i, alpha): # matrice elementaire, multiple la ligne i par le scalaire alpha
#    M[i,:]=alpha*M[i,:]
#    return M
#
#def Eijalpha(M, i,j, alpha): # matrice elementaire, AJOUTE à la ligne i alpha *ligne j. Attention alpha + ou -
#    M[i,:]=M[i,:] +  alpha *M[j,:]
#    return M
#def texA_rrefA(A, rrefA): #latex of A and rrefA as A~rrefA
#    texApre = '$\\left(\\begin{array}{'
#    texA = ''
#    for i in np.asarray(A) :
#        texALigne = ''
#        texALigne = texALigne + str(round(i[0],3) if i[0] %1 else int(i[0]))
#        if texA == '' :
#            texApre = texApre + 'c'
#        for j in i[1:] :
#            if texA == '' :
#                texApre = texApre + 'c'
#            texALigne = texALigne + ' & ' + str(round(j,3) if j %1 else int(j))
#        texALigne = texALigne + ' \\\\'
#        texA = texA + texALigne
#    texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)'
#    texApre = texA+ '\\quad \\sim \\, \ldots\\, \\sim \\quad \\left(\\begin{array}{'
#    texA = ''
#    for i in np.asarray(rrefA) :
#        texALigne = ''
#        texALigne = texALigne + str(round(i[0],3) if i[0] %1 else int(i[0]))
#        if texA == '' :
#            texApre = texApre + 'c'
#        for j in i[1:] :
#            if texA == '' :
#                texApre = texApre + 'c'
#            texALigne = texALigne + ' & ' + str(round(j,3) if j %1 else int(j))
#        texALigne = texALigne + ' \\\\'
#        texA = texA + texALigne
#    texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)$'   
#     
#    display(Latex(texA))
#    