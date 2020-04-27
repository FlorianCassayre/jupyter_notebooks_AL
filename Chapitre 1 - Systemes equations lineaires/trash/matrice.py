#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:01:03 2019

@author: jecker
"""

import numpy as np

def printA(A) :
    texApre = '$\\left(\\begin{array}{'
    texA = ''
    for i in np.asarray(A) :
        texALigne = ''
        texALigne = texALigne + str(i[0])
        if texA == '' :
            texApre = texApre + 'c'
        for j in i[1:] :
            if texA == '' :
                texApre = texApre + 'c'
            texALigne = texALigne + ' & ' + str(j)
        texALigne = texALigne + ' \\\\'
        texA = texA + texALigne
    texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)$'
    
    # print(texA)
    display(Latex(texA))
# done printing
    

A_array = np.array([[1, 2, 3], [3, 4, 5]])
A = np.asmatrix(A_array)
del A_array

#printA(A+A)
#printA(np.matmul(np.transpose(A),A))

printA(A)