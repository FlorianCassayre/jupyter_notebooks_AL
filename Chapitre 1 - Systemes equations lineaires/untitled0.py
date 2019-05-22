#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:41:46 2019

@author: jecker
"""

import random 

def generateRandomConnectedGraph( V):

    initialSet = set()

    visitedSet = set()

    vertices = set()

    edges = set()

    #generate the set of names for the vertices

    for i in range(V):

        initialSet.add(str(i))

        vertices.add(str(i))

    #set the intial vertex to be connected

    curVertex = random.sample(initialSet, 1).pop()

    initialSet.remove(curVertex)

    visitedSet.add(curVertex)

    #loop through all the vertices, connecting them randomly

    while initialSet:

        adjVertex = random.sample(initialSet, 1).pop()

        edge = (random.randint(0,3), curVertex, adjVertex)

        edges.add(edge)

        initialSet.remove(adjVertex)

        visitedSet.add(adjVertex)

        curVertex = adjVertex

    return vertices, edges


def generateCombination(n):

    com = []

    for i in range(n):

        for j in range(i+1, n):

            com.append([i,j])

            com.append([j,i])

    return com

V,E=generateRandomConnectedGraph( 5)