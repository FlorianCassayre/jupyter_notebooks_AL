#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:26:30 2019

@author: jecker
"""
from bokeh.plotting import figure, output_file, show


def syracuse(N):
    suite=[N]
    i=0
    ind=[i]
    while N !=1:
        i+=1
        ind.append(i)
        if N%2==0: 
            N=N/2
        else:
            N=3*N+1
        suite.append(N)
    return (suite, ind)


suite, ind=syracuse(121)
p = figure(plot_width=400, plot_height=400)

p.line(ind, suite, line_width=2)
#p.circle(ind, suite, fill_color="white", size=8)

show(p)