import sys
sys.path.insert(0, './../')
import numpy as np
import plotly
plotly.offline.init_notebook_mode(connected=True)
import ipywidgets as widgets
from IPython.display import display, Latex
from ipywidgets import interact_manual, Layout, interactive, HBox, VBox, widgets, interact, FloatSlider
from Librairie.AL_Fct import printA, texMatrix, isDiag, isSym
import plotly.graph_objs as go


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
        description=r'Il existe \(\lambda\in \mathbb{R}\) tel que \((A-\lambda B)^T\) '
                    r'soit échelonnée (mais pas réduite)',
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
            display(Latex("C'est faux."))
    
   
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
            C = [[-6, 64], [-32, -22], [28, -10], [-2, 6]]
            printA(C)
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex1Chapitre2_2():
    """Provides the correction of exercise 1 of notebook 2_2
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ vaut: <br>'
                    r'\begin{equation*} \qquad AB = \begin{pmatrix}-1 & 4\\-3& -3\\2 & 0\end{pmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='110px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ vaut: <br>'
                    r'\begin{equation*} \qquad AB =\begin{pmatrix}-1 & 8\\-3& 3\\-2 & 4\end{pmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='110px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ vaut: <br>'
                    r'\begin{equation*} \qquad AB =\begin{pmatrix}5 & -4\\1 & 0\end{pmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='90px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"Le produit $AB$ n'est pas défini",
        disabled=False,
        layout=Layout(width='100%', height='60px')
    )

    def correction(a, b, c, d):
        if b and not a and not c and not d:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))
   
    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex2Chapitre2_2():
    """Provides the correction of exercise 2 of notebook 2_2
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ appartient à $\mathcal{M}_{3 \times 3}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ appartient à $\mathcal{M}_{3 \times 2}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ appartient à $\mathcal{M}_{2 \times 1}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"$AB$ n'est pas définie",
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )

    def correction(a, b, c, d):
        if c and not a and not b and not d:
            A = [[14], [6]]
            texA = '$' + texMatrix(A) + '$'
            display(Latex(r"C'est correct! Le produit $ AB$ vaut: $AB$ = " + texA))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex3Chapitre2_2():
    """Provides the correction of exercise 3 of notebook 2_2
    """
    display(Latex("Insérez les valeurs de a et b"))
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
            display(Latex(r"Correct! Le produits $AB$ et $BA$ valent chacun: " + texAB))
        else:
            display(Latex(r"Incorrect! Le produit $AB$ vaut " + texAB + r"et par contre le produit "
                          r"$BA$ vaut " + texBA + r". Entrez de nouvelles valeurs!"))

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
        display(Latex('Correcte!'))
    else:
        display(Latex('Incorrect! Entrez des nouvelles valeurs pur le matrices B et C!\n'))

    if ans1:
        display(Latex("A+B est bien diagonale!"))
    else:
        display(Latex("A+B est n'est pas diagonale!"))
    texAB = '$' + texMatrix(A+B) + '$'
    display(Latex(r"A+B=" + texAB))

    if ans2:
        display(Latex("A+C est bien symétrique et non diagonale!"))
    elif isSym(A + C) and isDiag(A + C):
        display(Latex("A+C est bien symétrique mais elle est aussi diagonale!"))
    else:
        display(Latex("A + C n'est pas symétrique"))
    texAC = '$' + texMatrix(A + C) + '$'
    display(Latex(r"A+C=" + texAC))

    return


def Ex2Chapitre2_3():
    """Provides the correction to exercise 2 of notebook 2_3
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires supérieures mais différentes.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires inférieures mais différentes.',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires inférieures et identiques.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires supérieures et identiques.',
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
    """Provides the correction to exercise 2 of notebook 2_4
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le système admet une solution unique et elle est:'
                    r'$$\qquad \qquad x = \begin{pmatrix} 1&4/3&4/3\end{pmatrix}^T.$$',
        disabled=False,
        layout=Layout(width='50%', height='70px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet aucune solution.",
        disabled=False,
        layout=Layout(width='50%', height='20px')
    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le système admet une solution unique et elle est:'
                    r'$$\qquad \qquad x = \begin{pmatrix} 1&4/3&8/3\end{pmatrix}^T.$$',
        disabled=False,
        layout=Layout(width='50%', height='70px')

    )
    d = widgets.Checkbox(
        value=False,
        description=r'Le système admet plusieurs solutions.',
        disabled=False,
        layout=Layout(width='50%', height='20px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r'$A$ est inversible et son inverse est: <br>'
                    r'$$\qquad \qquad A^{-1} = \begin{pmatrix} 1/2&0&1/2\\1/2&-1/3&1/6\\1/2&-2/3&5/6\end{pmatrix}.$$',
        disabled=False,
        layout=Layout(width='50%', height='100px')
    )
    
    f = widgets.Checkbox(
        value=False,
        description=r'$A$ est inversible et son inverse est:'
                    r'$$\qquad \qquad A^{-1} = \begin{pmatrix} 1/2&0&1/2\\1/2&-1/3&-1/3\\1/2&-2/3&-1/2\end{pmatrix}.$$',
        disabled=False,
        layout=Layout(width='50%', height='100px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r"$A$ n'est pas inversible.",
        disabled=False,
        layout=Layout(width='50%', height='20px')
    )

    def correction(a, b, c, d, e, f, g):
        if c and e and not a and not b and not d and not f and not g:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g)

    return


def Ex2Chapitre2_4():
    """Provides the correction to exercise 3 of notebook 2_4
    """

    a = widgets.Checkbox(
        value=False,
        description=r"Le système admet une solution unique seulement si $\alpha < -7$.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Le système admet une unique solution seulement si $\alpha \geq -7$.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le système admet une unique solution $\forall \alpha \in \mathbb{R}.$',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    d = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet aucune solution si $\alpha < -7$, alors qu'il admet une solution unique si "
                    r"$\alpha \geq -7.$",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet aucune solution si $\alpha \neq -7$, alors qu'il admet une solution unique si"
                    r" $\alpha = -7$.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r"Aucun des précédents.",
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


def Ex1aChapitre2_5():
    """Provides the correction to exercise 1a of notebook 2_5
    """

    a = widgets.Checkbox(
        value=False,
        description=r'\(E_1E_2\) multiplie la ligne 4 par -6 et échange les lignes 2 et 3.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'\(E_1E_2\) ajoute 6 fois la ligne 4 à la ligne 2 et échange les lignes 1 et 3.',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'\(E_1E_2\) échange les lignes 1 et 3 et ajoute -6 fois la ligne 4 à la ligne 2.',
        disabled=False,
       layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"\(E_1E_2\) ajoute -6 fois la ligne 4 à la ligne 2 et échange les lignes 1 et 2.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d):
        if c and not(a) and not(d) and not(b):
            display(Latex("C'est correct! Par exemple, si on applique le produit à la matrice ci-dessous"))
            A=[[1,-1,0,0], [0,0,0,1], [1,2,1,2], [1,0,0,1]]
            B=[[1,0,0,0], [0,1,0,-6], [0,0,1,0], [0,0,0,1]]
            C=[[0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1]]
            BCA = np.linalg.multi_dot([B,C,A])
            texA = '$' + texMatrix(A) + '$'
            texBCA = '$' + texMatrix(BCA) + '$'
            display(Latex('$\qquad A = $' + texA))
            display(Latex("on obtient"))
            display((Latex('$\qquad \hat{A} = $' + texBCA)))
        else:
            display(Latex("C'est faux."))


    interact_manual(correction,a=a,b=b,c=c,d=d)

    return


def Ex1bChapitre2_5(inv):
    """Provides the correction to exercise 1b of notebook 2_5

    :param inv: inverse of the matrix to be calculated
    :type inv: list[list]
    """

    if inv == [[0, 0, 1, 0], [0, 1, 0, 6], [1, 0, 0, 0], [0, 0, 0, 1]]:
        display(Latex("C'est correct!"))
    else:
         display(Latex("C'est faux."))

    return


def Ex2aChapitre2_5(A, B, T, D, L):
    """Provides the correction to exercise 2a of notebook 2_5

    :param A: starting matrix
    :type A: list[list]
    :param B: target matrix
    :type B: list[list]
    :param T: permutation (type I) matrix
    :type T: list[list]
    :param D: scalar multiplication (type II) matrix
    :type D: list[list]
    :param L: linear combination (type III) matrix
    :type L: list[list]
    """

    if ~(B - np.linalg.multi_dot([L, D, T, A])).any():
        display(Latex("C'est correct!"))
    else:
        display(Latex("C'est faux."))
        str = 'Il faut entrer la/les matrice(s) {'
        if (np.asmatrix(T) - np.asmatrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])).any():
            str = str + ' T, '
        if (np.asmatrix(D) - np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 5]])).any():
            str = str + ' D, '
        if (np.asmatrix(L) - np.asmatrix([[1, 0, 0], [-4, 1, 0], [0, 0, 1]])).any():
            str = str + ' L, '
        str = str + '}. Le produit des matrices entrées vaut:'
        display(Latex(str))
        tmp = np.linalg.multi_dot([L, D, T, A])
        texM = '$' + texMatrix(tmp) + '$'
        display(Latex('$\qquad \hat{B} = $' + texM))

    return


def Ex2bChapitre2_5(inv):
    """Provides the correction to exercise 2b of notebook 2_5

    :param inv: inverse of the matrix to be calculated
    :type inv: list[list]
    """

    if inv == [[0, 0, 1/5], [4, 1, 0], [1, 0, 0]]:
        display(Latex("C'est correct!"))
    else:
        display(Latex("C'est faux."))


    return


def Ex1Chapitre2_6_7():
    """Provides the correction to exercise 1 of notebook 2_6-7
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ existe et le système admet plusieurs solutions, quelle que soit la valeur de $b$.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ existe et le système admet une solution unique ou plusieurs solutions en fonction '
                    r'de la valeur de $b$.',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$A$ est inversible et le système admet une solution unique, quelle que soit la valeur de $b$.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ existe et le système admet au moins une solution, quelle que soit la valeur de $b$.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"$A$ n'est pas inversible et le système admet une unique solution ou plusieurs solutions, "
                    r"selon la valeur de $b$. ",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r"$A$ n'est pas inversible et le système admet une solution unique, "
                    r"quelle que soit la valeur de $b$.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r"Le système admet une solution unique et $A$ n'est pas inversible.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    h = widgets.Checkbox(
        value=False,
        description=r'Le système admet une solution unique et $A$ est inversible.',
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


def Ex2Chapitre2_6_7():
    """Provides the correction to exercise 2 of notebook 2_6-7
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$A_1$ est inversible.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$A_2$ est inversible.',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$A_3$ est inversible.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c):
        if a and not b and not c:
            A1 = np.array([[2, 0, 1], [0, 6, 4], [2, 2, 1]])
            A1_inv = np.linalg.inv(A1)
            texA1inv = '$' + texMatrix(A1_inv) + '$'
            display(Latex("C'est correct! $A_1$ est la seule matrice inversible et son inverse est: $\quad A_1^{-1} = $"
                          + texA1inv))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c)

    return


def Ex3Chapitre2_6_7():
    """Provides the correction to exercise 3 of notebook 2_6-7
    """

    style = {'description_width': 'initial'}
    a = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha = 4$ et $\beta = 2$, alors $A$ n'est pas inversible et le système linéaire "
                    r"admet une infinité de solutions.",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=8$ et $\beta=-1$, alors $A$ n'est pas inversible et le système linéaire n'admet pas de"
                    r" solutions.",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=-2$ et $\beta=-4$, alors $A$ n'est pas inversible et le système linéaire admet "
                    r"une infinité de solutions.",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=8$ et $\beta=1$, alors $A$ n'est pas inversible et le système linéaire admet "
                    r"une infinité de solutions.",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=-4$ et $\beta=-2$, alors $A$ n'est pas inversible et le système linéaire n'admet pas "
                    r"de solutions.",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r'Si $\alpha=4$ et $\beta=1$, alors $A$ est inversible et le système linéaire admet une infinité '
                    r'de solutions.',
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r'Si $\alpha=4$ et $\beta=1$, alors $A$ est inversible et le système linéaire admet une solution'
                    r' unique.',
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    h = widgets.Checkbox(
        value=False,
        description=r"$A$ n'est pas inversible pour infinite de valeurs de $\alpha$ et $\beta$. Mais il existe un unique couple $(\alpha, \beta)$ pour lequel"
                    r"le système admet une infinité de solutions.",
        disabled=False,
        style=style,
        layout=Layout(width='100%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if c and e and g and h and not a and not b and not d and not f:
            display(Latex(r"C'est correct! En effet $A$ n'est pas inversible si $\alpha = \dfrac{8}{\beta}$ (résultat "
                          r"obtenu en divisant par élément les lignes de A les unes par les autres et en imposant que "
                          r"les résultats des divisions soient les mêmes). Si $\alpha = -2$ et $\beta = -4$, "
                          r"alors le système admet une infinité de solutions, puisque les rapports obtenus par la "
                          r"division est $-\dfrac{1}{2}$, qui est égal au rapport entre les éléments du vecteur de droite "
                          r"$b$."))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return


def graphe_Ex3Chapitre2_6_7():
    """Provides the graph of exercice 3 of nnotebook 2_6-7
    """
    np.seterr(divide='ignore', invalid='ignore')

    A=[[2, 0], [0, -4]] # we initialize the problem. The values of alpha and beta are fixed
    b=[1, -2]

    m=len(A)
    MatCoeff = [A[i]+[b[i]]for i in range(m)] #becomes augmented matrix
    MatCoeff=np.array(MatCoeff)
    data=[]
    x=np.linspace(-15,15,101)
    y=np.linspace(-10,10,101)
    MatCoeff=np.array(MatCoeff)
    for i in range(len(MatCoeff)):
        trace=go.Scatter(x=x,  y= (MatCoeff[i,2]-MatCoeff[i,0]*x)/MatCoeff[i,1], name='d) Droite %d'%(i+1))
        data.append(trace)

    f=go.FigureWidget(data=data,
        layout=go.Layout(xaxis=dict(
            range=[-15, 15]
        ),
        yaxis=dict(
            range=[-10, 10]
        ) )                  
    )

    def update_y(alpha, beta):
        MatCoeff= [[2, -alpha, 1],[beta, -4, -2]]
        MatCoeff=np.array(MatCoeff)
        if MatCoeff[0,1] == 0:
            MatCoeff[0,1] += 1e-3
        f.data[0].y = (MatCoeff[0,2]-MatCoeff[0,0]*x)/MatCoeff[0,1]
        if MatCoeff[1,1] == 0:
            MatCoeff[1,1] += 1e-3
        f.data[1].y=(MatCoeff[1,2]-MatCoeff[1,0]*x)/MatCoeff[1,1]

    freq_slider = interactive(update_y, alpha=(-20, 20, 1/2), beta=(-20, 20, 1/2))

    vb = VBox((f, freq_slider))
    vb.layout.align_items = 'center'
    vb    

    return vb


def Ex4Chapitre2_6_7():
    """Provides the correction of exercise 4 of notebook 2_6-7
    """

    display(Latex("Insérez les valeurs de a et b"))
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
        A = np.array([[0.5, a.value, 1], [0, 2, -1], [-2, 1, b.value]])
        B = np.array([[-6, -2, -2], [4, 2, 1], [8, 3, 2]])

        AB = np.dot(A, B)
        texAB = '$' + texMatrix(AB) + '$'
        BA = np.dot(B, A)
        texBA = '$' + texMatrix(BA) + '$'

        if a.value == -1 and b.value == -2:
            display(Latex(r"Correct! Le produits $AB$ et $BA$ valent chacun: $I$ = " + texAB))
        else:
            display(Latex(r"Incorrect! Le produit $AB$ vaut  " + texAB + r" et le produit "
                          r"$BA$ vaut  " + texBA + r",  donc $A$ ne peut pas être l'inverse de $B$. "
                          r"Entrez de nouvelles valeurs!"))

    interact_manual(f)

    return


def Ex1Chapitre2_8_9(E1, E2, E3, E4):
    """Provides the correction of exercise 2 of notebook 2_8_9

    :param E1:
    :type E1:
    :param E2:
    :type E2:
    :param E3:
    :type E3:
    :param E4:
    :type E4:
    :return:
    :rtype:
    """

    # MATRIX A1
    E_pre_1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    E_post_1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]]

    # MATRIX A2
    E_pre_2 = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    E_post_2 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/2, 0], [0, 0, 0, -1]]

    # MATRIX A3
    E_pre_3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    E_post_3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -2], [0, 0, 0, 1]]

    # MATRIX A4
    E_pre_4 = [[1/2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    E_post_4 = [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, -1]]

    E_bool = np.zeros(4).astype(bool)
    E_bool[0] = E1[0] == E_pre_1 and E1[1] == E_post_1
    E_bool[1] = E2[0] == E_pre_2 and E2[1] == E_post_2
    E_bool[2] = E3[0] == E_pre_3 and E3[1] == E_post_3
    E_bool[3] = E4[0] == E_pre_4 and E4[1] == E_post_4

    correct = set(np.where(E_bool)[0]+1)
    wrong = set(np.arange(1,5)) - correct

    if wrong:
        if correct:
            display(Latex(f"Corrects: {correct}"))
        else:
            display((Latex("Corrects: {}")))
        display(Latex(f"Manqué: {wrong}"))
    else:
        display(Latex("C'est correct."))

    return


def Ex3Chapitre2_8_9():
    """Provides the correction of exercise 3 of notebook 2_8_9
    """
    
    
    a_1 = widgets.Checkbox(
        value=False,
        description=r'La matrice $A_1$ admet une décomposition LU.',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    
    a_2 = widgets.Checkbox(
        value=False,
        description=r'La matrice $A_2$ admet une décomposition LU.',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    
    a_3 = widgets.Checkbox(
        value=False,
        description=r'La matrice $A_3$ admet une décomposition LU.',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    
    def correction(a_1, a_2, a_3):
        if not a_1 and a_2 and not a_3:
            display(Latex("C'est correct! Plus précisément, la matrice $A_1$ n'admet pas de décomposition LU car elle n'est pas inversible, la matrice $A_2$ admet décomposition LU et la matrice $A_3$ n'admet pas décomposition LU car elle ne peut pas être réduite sans échanger deux lignes pendant la méthode d'élimination de Gauss"))         
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a_1=a_1, a_2=a_2, a_3=a_3)

    return


def Ex1Chapitre2_10():
    """Provides the correction to exercise 1 of notebook 2_10
    """

    a = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire n'admet aucune solution.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire admet une solution unique.",
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire admet deux solutions distinctes.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire admet une infinité de solutions.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"La décomposition LU de $A$ ne peut pas être calculée.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r"La dernière colonne de $U$ est entièrement composée de zéros.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r'La dernière ligne de $U$ est entièrement composée de zéros.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    h = widgets.Checkbox(
        value=False,
        description=r'La première entrée de la première ligne de $L$ est égale à 1.',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if not a and not b and not c and d and not e and not f and g and not h:
            display(Latex("C'est correct. En effet, $A$ n'est clairement pas inversible, car la dernière ligne est "
                          "égale à la seconde moins la première, et il en va de même pour le vecteur de droite $b$. "
                          "Par conséquent, la dernière ligne de $U$ est entièrement composée de zéros (réponse 7) et la "
                          "dernière entrée du vecteur de droite $b$, après l'application de la méthode d'élimination de "
                          "Gauss, est également égale à 0. Ainsi, la dernière équation du système linéaire résultant a "
                          "tous les coefficients égaux à 0, ce qui donne lieu à une infinité de solutions "
                          "(réponse 4)."))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return


def Ex2Chapitre2_10(L, U, b, x, y):
    """Provides the correction to exercise 2 of notebook 2_10

    :param L: lower triangular matrix from LU decomposition
    :type L: list[list]
    :param U: upper triangular matrix from LU decomposition
    :type U: list[list]
    :param b: right-hand side vector
    :type b: list[list]
    :param x: system solution
    :type x: list[list]
    :param y: temporary variable
    :type y: list[list]
    """

    if type(L) is list:
        L = np.array(L)

    if type(U) is list:
        U = np.array(U)

    if type(x) is list:
        x = np.array(x)

    if type(y) is list:
        y = np.array(y)

    y_true = np.linalg.solve(L, b)
    x_true = np.linalg.solve(U, y)

    res_x = np.linalg.norm(x - x_true) / np.linalg.norm(x_true) <= 1e-4
    res_y = np.linalg.norm(y - y_true) / np.linalg.norm(y_true) <= 1e-4

    if res_x and res_y:
        display(Latex("C'est correct"))
    else:
        display(Latex("C'est faux"))

    return


def Ex3Chapitre2_10():
    """Provides the correction to exercise 3 of notebook 2_10

    :return:
    :rtype:
    """

    a = widgets.Checkbox(
        value=False,
        description=r"If L is such that all its diagonal elements equal 1, then the temporary variable y is a vector "
                    r"of ones as well",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    b = widgets.Checkbox(
        value=False,
        description=r"La matrice L est diagonale.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    c = widgets.Checkbox(
        value=False,
        description=r"La matrice U est diagonale",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    d = widgets.Checkbox(
        value=False,
        description=r"La deuxième entrée de la solution est toujours 2.5 fois la 4ème.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    e = widgets.Checkbox(
        value=False,
        description=r"La deuxième entrée de la solution est toujours 5 fois la 4ème.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    f = widgets.Checkbox(
        value=False,
        description=r"La somme de toutes les entrées est toujours égale à 2.5",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    g = widgets.Checkbox(
        value=False,
        description=r"La somme de toutes les entrées sauf la deuxième, est toujours égale à 0.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    h = widgets.Checkbox(
        value=False,
        description=r"Le vecteur $\hat{x} = (1, 0 -1, 0)$ est une des solutions du systèmes d'équations linéaires.",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if a and not b and not c and d and not e and not f and g and h:
            display(Latex("C'est correct. En effet, l'ensemble des solutions peut être écrit comme "
                          "$x = [1-4a, 2.5a, 3a-1, a]$. On en déduit que la deuxième entrée est 2.5 fois la 4ème (réponse 4),"
                          " que la somme de toutes les entrée sauf la 2ème vaut 0 (réponse 7) "
                          "et que $\hat{x} = [1,0,-1,0]$ est une solution, c'est le cas $a=0$ (réponse 8). "
                          "Alors, si L est calculée de manière à obtenir des 1 sur sa diagonale, on peut déduire que "
                          "le vecteur $\vec{y}$ qui résout $L\vec{y}=\vec{b}$ n'est fait que de 1."))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return

def Ex1Chapitre2_11():
    """Provides the correction to exercise 1 of notebook 2_11
    """

    print("Cliquer sur CTRL pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    ans1 = widgets.SelectMultiple(
        options=['Addition', 'Multiplication', 'Addition par blocs', 'Multiplication par blocs'],
        description='Cas 1:',
        style=style,
        layout=Layout(width='30%', height='90px'),
        disabled=False,
    )
    ans2 = widgets.SelectMultiple(
        options=['Addition', 'Multiplication', 'Addition par blocs', 'Multiplication par blocs'],
        description='Cas 2:',
        style=style,
        layout=Layout(width='30%', height='90px'),
        disabled=False,
    )
    ans3 = widgets.SelectMultiple(
        options=['Addition', 'Multiplication', 'Addition par blocs', 'Multiplication par blocs'],
        description='Cas 3:',
        style=style,
        layout=Layout(width='30%', height='90px'),
        disabled=False,
    )
    ans4 = widgets.SelectMultiple(
        options=['Addition', 'Multiplication', 'Addition par blocs', 'Multiplication par blocs'],
        description='Cas 4:',
        style=style,
        layout=Layout(width='30%', height='90px'),
        disabled=False,
    )

    def correction(ans1, ans2, ans3, ans4):
        res_ans = np.zeros(4).astype(bool)
        res_ans[0] = 'Addition' in ans1 and 'Multiplication' in ans1 \
                   and 'Addition par blocs' in ans1 and not 'Multiplication par blocs' in ans1
        res_ans[1] = 'Addition' in ans2 and 'Multiplication' in ans2 \
                    and not 'Addition par blocs' in ans2 and 'Multiplication par blocs' in ans2
        res_ans[2] = not 'Addition' in ans3 and 'Multiplication' in ans3 \
                    and not 'Addition par blocs' in ans3 and 'Multiplication par blocs' in ans3
        res_ans[3] = not 'Addition' in ans4 and 'Multiplication' in ans4 \
                    and not 'Addition par blocs' in ans4 and not 'Multiplication par blocs' in ans4
        if res_ans.all():
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))
            correct = set(np.where(res_ans)[0] + 1)
            wrong = set(np.arange(1, 5)) - correct
            if correct:
                display(Latex(f"Corrects: {correct}"))
            else:
                display((Latex("Corrects: {}")))
            display(Latex(f"Manqué: {wrong}"))

    interact_manual(correction, ans1=ans1, ans2=ans2, ans3=ans3, ans4=ans4)

    return


def Ex2Chapitre2_11():
    """Provides the correction of exercise 2 of notebook 2_11
    """
    display(Latex("Insérez votre réponse ici"))
    a = widgets.IntText(
        value=0,
        step=1,
        description='Answer:',
        disabled=False
    )

    display(a)

    def f():
        A_tex = "$$ \\qquad \\qquad \\qquad  \\qquad \\qquad \\qquad \\quad" \
                " A = \\left(\\begin{array}{@{}cc|cc|c@{}} " \
                "a_{11} & a_{12} & a_{13} & a_{14} & a_{15} \\\\" \
                " a_{21} & a_{22} & a_{23} & a_{24} & a_{25} \\\\ " \
                "a_{31} & a_{32} & a_{33} & a_{34} & a_{35} \\\\ " \
                "a_{41} & a_{42} & a_{43} & a_{44} & a_{45} " \
                "\\end{array}\\right)$$"

        if a.value == 8:
            display(Latex(r"Correct! En effet les colonnes de la matrice A doivent être décomposées comme suit, "
                          r"afin de satisfaire les contraintes de dimensionnalité" + A_tex + r"Ensuite, la décomposition "
                          "en blocs par lignes peut être effectuée dans l'une des $2^{n-1}=8$ possibilités disponibles."))
        else:
            display(Latex(r"Incorrect! Aide: Faites attention à la façon dont A doit être décomposé en blocs par "
                          r"colonnes."))

    interact_manual(f)

    return


def Ex3Chapitre2_11(C1, C2, C3):
    """Provides the correction of exercise 3 of notebook 2_11

    :param C1:
    :type C1:
    :param C2:
    :type C2:
    :param C3:
    :type C3:
    :return:
    :rtype:
    """

    C1_true = [[]]
    C2_true = [[-1,-4,1], [4,6,2], [1,-1,2]]
    C3_true = [[6,7,2], [-10,-11,-2], [4,12,8], [-4,-8,-4]]

    C_bool = np.zeros(3).astype(bool)
    C_bool[0] = C1 == C1_true
    C_bool[1] = C2 == C2_true
    C_bool[2] = C3 == C3_true

    correct = set(np.where(C_bool)[0]+1)
    wrong = set(np.arange(1,4)) - correct

    if wrong:
        display(Latex("C'est faux."))
        if correct:
            display(Latex(f"Corrects: {correct}"))
        else:
            display((Latex("Corrects: {}")))
        display(Latex(f"Manqué: {wrong}"))
    else:
        display(Latex("C'est correct."))

    return


def Ex4Chapitre2_11(A1_inv, A2_inv):
    """Provides the correction to exercise 4 of notebook 2_11

    :param A1_inv: inverse of matrix A1
    :type A1_inv: list[list]
    :param A2_inv: inverse of matrix A2
    :type A2_inv: list[list]
    """

    A1 = np.array([[1, 2, 0, 1, 0], [0, -2, 1, -1, 0], [-2, -1, 1, 0, 1], [0, 0, 0, 2, 0], [0, 0, 0, 0, 1]])
    A1_inv_true = np.linalg.inv(A1)
    A1_inv = np.array(A1_inv)

    A2_inv_true = [[]]

    A_inv_bool = np.zeros(3).astype(bool)
    A_inv_bool[0] = A1_inv.shape == A1_inv_true.shape and np.linalg.norm(A1_inv - A1_inv_true) < 1e-6
    A_inv_bool[1] = A2_inv == A2_inv_true

    correct = set(np.where(A_inv_bool)[0] + 1)
    wrong = set(np.arange(1, 3)) - correct

    if wrong:
        display(Latex("C'est faux."))
        if correct:
            display(Latex(f"Corrects: {correct}"))
        else:
            display((Latex("Corrects: {}")))
        display(Latex(f"Manqué: {wrong}"))
    else:
        display(Latex("C'est correct."))

    return
