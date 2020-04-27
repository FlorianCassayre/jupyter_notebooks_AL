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
   

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

from matplotlib.backends.backend_gtk3agg import (
    FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit)
win.set_default_size(400, 300)
win.set_title("Embedding in GTK")

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)
t = np.arange(0.0, 3.0, 0.01)
s = np.sin(2*np.pi*t)
a.plot(t, s)

sw = Gtk.ScrolledWindow()
win.add(sw)
# A scrolled window border goes outside the scrollbars and viewport
sw.set_border_width(10)

canvas = FigureCanvas(f)  # a Gtk.DrawingArea
canvas.set_size_request(800, 600)
sw.add_with_viewport(canvas)

win.show_all()
Gtk.main()