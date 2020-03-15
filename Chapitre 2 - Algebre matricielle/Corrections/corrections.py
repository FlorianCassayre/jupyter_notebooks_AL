import sys
sys.path.insert(0, './../')
import numpy as np
import plotly
plotly.offline.init_notebook_mode(connected=True)
import ipywidgets as widgets
from IPython.display import display, Latex
from ipywidgets import interact_manual, Layout
from Librairie.AL_Fct import printA, texMatrix, isDiag, isSym


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


def Ex1Chapitre2_2():
    """Provides the correction of exercise 1 of notebook 2_2
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le produit $A \cdot B$ vaut: <br>'
                    r'\begin{equation*} \qquad A \cdot B = \begin{bmatrix}-1 & 4\\-3& -3\\2 & 0\end{bmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='110px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Le produit $A \cdot B$ vaut: <br>'
                    r'\begin{equation*} \qquad A \cdot B =\begin{bmatrix}-1 & 8\\-3& 3\\-2 & 4\end{bmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='110px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le produit $A \cdot B$ vaut: <br>'
                    r'\begin{equation*} \qquad A \cdot B =\begin{bmatrix}5 & -4\\1 & 0\end{bmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='90px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"Le produit $A \cdot B$ n'est pas définie",
        disabled=False,
        layout=Layout(width='100%', height='60px')
    )

    def correction(a, b, c, d):
        if b and not a and not c and not d:
            print("C'est correct!")
        else:
            print("C'est faux.")

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex2Chapitre2_2():
    """Provides the correction of exercise 2 of notebook 2_2
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le produit $A \cdot B$ appartient à $M_{3 \times 3}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Le produit $A \cdot B$ appartient à $M_{3 \times 2}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le produit $A \cdot B$ appartient à $M_{2 \times 1}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"$A \cdot B$ n'est pas définie",
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )

    def correction(a, b, c, d):
        if c and not a and not b and not d:
            A = [[14], [6]]
            texA = '$' + texMatrix(A) + '$'
            display(Latex(r"C'est correct! Le produit $ A \cdot B$ vaut: $A \cdot B$ = " + texA))
        else:
            print("C'est faux.")

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex3Chapitre2_2():
    """Provides the correction of exercise 3 of notebook 2_2
    """
    display(Latex("Insert the values of a and b as floating point numbers"))
    a = widgets.FloatText(
        value=0.0,
        step=0.1,
        description='a:',
        disabled=False
    )
    b = widgets.FloatText(
        value=0.0,
        step=0.1,
        description='b:',
        disabled=False
    )

    display(a)
    display(b)

    def f():
        A = np.array([[-1, 2], [5, -2]])
        B = np.array([[-1, 1], [a.value, b.value]])

        AB = np.dot(A,B)
        texAB = '$' + texMatrix(AB) + '$'
        BA = np.dot(B,A)
        texBA = '$' + texMatrix(BA) + '$'

        if a.value == 5/2 and b.value == -3/2:
            display(Latex(r"Correcte! Le produits $A \cdot B$ et $B \cdot A$ valent chacun: " + texAB))
        else:
            display(Latex(r"Incorrecte! Le produit $A \cdot B$ vaut " + texAB + r"et par contre le produit "
                          r"$B \cdot A$ vaut " + texBA + r" Entrez de nouvelles valeurs!"))

    interact_manual(f)

    return


def Ex1Chapitre2_3(A, B, C):
    """Provides the correction to exercise 1 of notebook 2_3

    :param A: original matrix
    :type A: list[list] or numpy.ndarray
    :param B: matrix such that A+B should be diagonal
    :type B: list[list] or numpy.ndarray
    :param C: matrix such that A+C should be symmetric and not diagonal
    :type C: list[list] or numpy.ndarray
    :return:
    :rtype:
    """

    if not type(A) is np.ndarray:
        A = np.array(A)
    if not type(B) is np.ndarray:
        B = np.array(B)
    if not type(C) is np.ndarray:
        C = np.array(C)

    ans1 = isDiag(A+B)
    ans2 = isSym(A+C) and not isDiag(A+C)

    if ans1 and ans2:
        print('Correcte!')
    else:
        print('Incorrecte! Entrez des nouvelles valeurs pur le matrices B et C!\n')

    if ans1:
        print("A+B est bien diagonale!")
    else:
        print("A+B est n'est pas diagonale!")
    texAB = '$' + texMatrix(A+B) + '$'
    display(Latex(r"A+B=" + texAB))

    if ans2:
        print("A+C est bien symétrique et non diagonale!")
    elif isSym(A + C) and isDiag(A + C):
        print("A+C est bien symétrique mais elle est aussi diagonale!")
    else:
        print("A + C n'est pas symétrique")
    texAC = '$' + texMatrix(A + C) + '$'
    display(Latex(r"A+C=" + texAC))

    return


def Ex2Chapitre2_3():
    """Provides the correction to exercise 2 of notebook 2_3
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires supérieures mais différentes',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires inférieures mais différentes',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires inférieures et identiques',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires supérieures et identiques',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d):
        if d and not a and not c and not b:
            A = np.array(([-1, 0, 0], [3, 1/2, 0], [1, 2, 1]))
            res = np.transpose(np.linalg.inv(A))
            texAres = '$' + texMatrix(res) + '$'
            display(Latex("C'est correct! $(A^T)^{-1}$ est donnée par: $\quad$" + texAres))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex1Chapitre2_4():
    """Provides the correction to exercise 1 of notebook 2_4
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ exists and the system admits multiple solutions, whichever $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ exists and the system admits either a unique solution or multiple solutions '
                    r'depending on the value of $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$A$ is invertible and the system admits a unique solution, whichever $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ exists and the system admits at least one solution, whichever $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r'$A$ is not invertible and the system admits either a unique or multiple solutions, depending on '
                    r'the value of $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r'$A$ is not invertible and the system admits a unique solution, whichever $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r'The system admits a unique solution and $A$ is not invertible',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    h = widgets.Checkbox(
        value=False,
        description=r'The system admits a unique solution and $A$ is invertible',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if c and d and h and not a and not b and not e and not f and not g:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return


def Ex2Chapitre2_4():
    """Provides the correction to exercise 2 of notebook 2_4
    """

    a = widgets.Checkbox(
        value=False,
        description=r'The system admits a unique solution and it is:'
                    r'$$\qquad x = \begin{bmatrix} 1&4/3&4/3\end{bmatrix}^T$$',
        disabled=False,
        layout=Layout(width='80%', height='70px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'The system does not admit any solution',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    c = widgets.Checkbox(
        value=False,
        description=r'The system admits a unique solution and it is:'
                    r'$$\qquad x = \begin{bmatrix} 1&4/3&8/3\end{bmatrix}^T$$',
        disabled=False,
        layout=Layout(width='80%', height='70px')

    )
    d = widgets.Checkbox(
        value=False,
        description=r'The system admits multiple solutions',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r'$A$ is invertible and its inverse is: <br>'
                    r'$$\qquad A^{-1} = \begin{bmatrix} 1/2&0&1/2\\1/2&-1/3&5/3\\1/2&-2/3&5/6\end{bmatrix}$$',
        disabled=False,
        layout=Layout(width='80%', height='100px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r'$A$ is not invertible',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r'$A$ is invertible and its inverse is: <br>'
                    r'$$\qquad A^{-1} = \begin{bmatrix} 1/2&0&1/2\\1/2&-1/3&5/3\\1/2&-2/3&-1/2\end{bmatrix}$$',
        disabled=False,
        layout=Layout(width='80%', height='100x')
    )

    def correction(a, b, c, d, e, f, g):
        if c and e and not a and not b and not d and not f and not g:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g)

    return


def Ex3Chapitre2_4():
    """Provides the correction to exercise 3 of notebook 2_4
    """

    a = widgets.Checkbox(
        value=False,
        description=r'The system admits unique solution only if $\alpha < 2$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'The system admits unique solution only if $\alpha \geq 2$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    c = widgets.Checkbox(
        value=False,
        description=r'The system admits unique solution $\forall \alpha \in \mathbb{R}$',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    d = widgets.Checkbox(
        value=False,
        description=r'The system does not admit any solution if $\alpha < 2$, while it admits unique solution'
                    r' if $\alpha \geq 2$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r'The system admits multiple solutions if $\alpha \neq 2$, while it admits unique solution '
                    r'for $\alpha == 2$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r'The system never admits a unique solution, whichever $\alpha \in \mathbb{R}$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f):
        if f and not a and not b and not c and not d and not e:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f)

    return



