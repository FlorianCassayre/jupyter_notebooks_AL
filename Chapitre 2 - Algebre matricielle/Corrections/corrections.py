import sys
sys.path.insert(0, './../')

import plotly
plotly.offline.init_notebook_mode(connected=True)
import ipywidgets as widgets
from IPython.display import display, Latex
from ipywidgets import interact_manual, Layout
from Librairie.AL_Fct import printA


def Ex2Chapitre2_1():
    """Provides the correction of exercise 2 of notebook 2_1
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Il existe \(\lambda\in \mathbb{R}\) tel que \((A-\lambda B)^T\) soit échelonnée réduite',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Il existe \(\lambda\in \mathbb{R}\) tel que \((A-\lambda B)^T\) soit échelonnée (mais pas réduite)',
        disabled=False,
        layout=Layout(width='80%', height='30px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r"Il n'existe pas de \(\lambda\in \mathbb{R}\) tel que \((A-\lambda B)^T\) soit échelonnée",
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )

    def correction(a, b, c):
        if a and not c and not b:
            display(Latex("C'est correct! Pour $\lambda=-3$ La matrice échelonnée réduite est:"))
            A = [[1, 0, -2, 3], [0, 1, -1, 7]]
            printA(A)
        else:
            print("C'est faux.")

    interact_manual(correction, a=a, b=b, c=c)

    return


def Ex3Chapitre2_1():
    """Provides the correction of exercise 3 of notebook 2_1
    """

    a = widgets.Checkbox(
        value=False,
        description=r'\(C_{32}\) vaut -14',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'\(C_{32}\) vaut 14',
        disabled=False,
        layout=Layout(width='80%', height='30px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'\(C_{32}\) vaut -10',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"\(C_{32}\) n'existe pas",
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )

    def correction(a, b, c, d):
        if c and not a and not b and not d:
            display(Latex("C'est correct! La matrice C vaut:"))
            A = [[-6, 64], [-32, -22], [28, -10], [-2, 6]]
            printA(A)
        else:
            print("C'est faux.")

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return

