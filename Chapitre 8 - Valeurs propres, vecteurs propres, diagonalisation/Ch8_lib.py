import sys, os
sys.path.append('../Librairie')
import AL_Fct as al
import numpy as np
import sympy as sp
from IPython.utils import io
from IPython.display import display, Latex, Markdown
import plotly
import plotly.graph_objects as go


def vector_plot_3D(v, b):
    """
    Show 3D plot of a vector (v) and of b = A * v
    @param v: numpy array of shape (3,)
    @param b: numpy array of shape (3,)
    @return:
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
                        line=dict(color='red', width=4),
                        mode='lines+markers',
                        name='$v$'))

    fig.add_trace(go.Scatter3d(x=[0, b[0]], y=[0, b[1]], z=[0, b[2]],
                        line=dict(color='royalblue', width=4, dash='dash'),
                        mode='lines+markers',
                        name='$A \ v$'))

    fig.show()


def CheckEigenVector(A, v):
    """
    Check if v is an eigenvector of A, display step by step solution
    @param A: square sympy Matrix of shape (n,n)
    @param v: 1D sympy Matrix of shape (n,1)
    @return:
    """
    # Check Dimensions
    if A.shape[0] != A.shape[1] or v.shape[0] != A.shape[1]:
        raise ValueError('Dimension problem, A should be square (n x n) and v (n x 1)')

    if v == sp.zeros(v.shape[0], 1):
        display(Latex("$v$ est le vecteur nul, il ne peut pas être un vecteur propre par définition."))

    else:
        # Matrix Multiplication
        b = A * v

        # Print some explanation about the method
        display(Latex("On voit que $ b = A v = " + latexp(b) + "$"))
        display(Latex("On cherche alors un nombre $\lambda \in \mathbb{R}$ tel que $b = \lambda v" \
                      + "\Leftrightarrow" + latexp(b) + " = \lambda" + latexp(v) + '$'))

        # Symbol for lambda
        l = sp.symbols('\lambda', real=True)

        # Check if there is a solution lambda of eq: A*v = lambda * v
        eq = sp.Eq(b, l * v)
        sol = sp.solve(eq, l)

        # If there is l st b = l*v
        if sol:
            display(Latex("Il existe bien une solution pour $\lambda$. Le vecteur $v$ est donc un vecteur \
                          propre de la matrice $A$."))
            display(Latex("La valeur propre associée est $\lambda = " + sp.latex(sol[l]) + "$."))
        # Otherwise
        else:
            display(Latex("L'equation $b = \lambda v$ n'a pas de solution."))
            display(Latex("Le vecteur $v$ n'est donc pas un vecteur propre de la matrice $A$."))


def ch8_1_exo_2(A, l, vp, v):
    """
    Display step by step
    @param A: Square sympy matrix
    @param l: eigenvalue (float or int)
    @param vp: Boolean, given answer to question is l an eigenvalue of A
    @param v: proposed eigenvector
    @return:
    """

    # Check Dimensions
    if A.shape[0] != A.shape[1] or v.shape[0] != A.shape[1]:
        raise ValueError('Dimension problem, A should be square (n x n) and v (n x 1)')

    n = A.shape[0]

    eig = list(A.eigenvals().keys())

    for i, w in enumerate(eig):
        eig[i] = float(w)

    eig = np.array(eig)

    if np.any(abs(l-eig) < 10**-10):
        if vp:
            display(Latex("$\lambda = " + str(l) + "$ est bien une valeur propre de la matrice $A$."))
        else:
            display(Latex("Non, $\lambda = " + str(l) + "$ est bien une valeur propre de la matrice $A$."))

        if v != sp.zeros(n, 1):
            # Check the eigen vector v
            z = sp.simplify(A * v - l * v)
            if z == sp.zeros(n, 1):
                display(Latex("$v$ est bien un vecteur propre de $A$ associé à $\lambda = " + str(l) + "$ car on a:"))
                display(Latex("$$" + latexp(A) + latexp(v) + "= " + str(l) + "\cdot " + latexp(v) + "$$"))
            else:
                display(Latex("$v$ n'est pas un vecteur propre de $A$ associé à $\lambda = " + str(l) + "$ car on a:"))
                display(Latex("$$" + latexp(A) + latexp(v) + "\\neq \lambda" + latexp(v) + "$$"))
        else:
            display(Latex("$v$ est le vecteur nul et ne peut pas être par définition un vecteur propre."))

    else:
        if vp:
            display(Latex("En effet, $\lambda$ n'est pas une valeur propre de $A$."))
        else:
            display(Latex("Non, $\lambda = " + str(l) + "$ n'est pas une valeur propre de $A$."))


def red_matrix(A, i, j):
    """ Return reduced matrix (without row i and col j)"""
    row = [0, 1, 2]
    col = [0, 1, 2]
    row.remove(i - 1)
    col.remove(j - 1)
    return A[row, col]


def pl_mi(i, j, first=False):
    """ Return '+', '-' depending on row and col index"""
    if (-1) ** (i + j) > 0:
        if first:
            return ""
        else:
            return "+"
    else:
        return "-"


def brackets(expr):
    """Takes a sympy expression, determine if it needs parenthesis and returns a string containing latex of expr
    with or without the parenthesis."""
    expr_latex = sp.latex(expr)
    if '+' in expr_latex or '-' in expr_latex:
        return "(" + expr_latex + ")"
    else:
        return expr_latex


def Determinant_3x3(A, step_by_step=True, row=True, n=1):
    """
    Step by step computation of the determinant of a 3x3 sympy matrix strating with given row/col number
    @param A: 3 by 3 sympy matrix
    @param step_by_step: Boolean, True: print step by step derivation of det, False: print only determinant
    @param row: True to compute determinant from row n, False to compute determinant from col n
    @param n: row or col number to compute the determinant from (int between 1 and 3)
    @return: display step by step solution for
    """

    if A.shape != (3, 3):
        raise ValueError('Dimension of matrix A should be 3x3. The input A must be a sp.Matrix of shape (3,3).')
    if n < 1 or n > 3 or not isinstance(n, int):
        raise ValueError('n should be an integer between 1 and 3.')

    # Construc string for determinant of matrix A
    detA_s = sp.latex(A).replace('[', '|').replace(']', '|')

    # To print all the steps
    if step_by_step:

        # If we compute the determinant with row n
        if row:
            # Matrix with row i and col j removed (red_matrix(A, i, j))
            A1 = red_matrix(A, n, 1)
            A2 = red_matrix(A, n, 2)
            A3 = red_matrix(A, n, 3)
            detA1_s = sp.latex(A1).replace('[', '|').replace(']', '|')

            detA2_s = sp.latex(A2).replace('[', '|').replace(']', '|')
            detA3_s = sp.latex(A3).replace('[', '|').replace(']', '|')

            line1 = "$" + detA_s + ' = ' + pl_mi(n, 1, True) + brackets(A[n - 1, 0]) + detA1_s + pl_mi(n, 2) + \
                    brackets(A[n - 1, 1]) + detA2_s + pl_mi(n, 3) + brackets(A[n - 1, 2]) + detA3_s + '$'

            line2 = '$' + detA_s + ' = ' + pl_mi(n, 1, True) + brackets(A[n - 1, 0]) + "\cdot (" + sp.latex(sp.det(A1)) \
                    + ")" + pl_mi(n, 2) + brackets(A[n - 1, 1]) + "\cdot (" + sp.latex(sp.det(A2)) + ")" + \
                    pl_mi(n, 3) + brackets(A[n - 1, 2]) + "\cdot (" + sp.latex(sp.det(A3)) + ')$'
            line3 = '$' + detA_s + ' = ' + sp.latex(sp.simplify(sp.det(A))) + '$'

        # If we compute the determinant with col n
        else:
            # Matrix with row i and col j removed (red_matrix(A, i, j))
            A1 = red_matrix(A, 1, n)
            A2 = red_matrix(A, 2, n)
            A3 = red_matrix(A, 3, n)
            detA1_s = sp.latex(A1).replace('[', '|').replace(']', '|')
            detA2_s = sp.latex(A2).replace('[', '|').replace(']', '|')
            detA3_s = sp.latex(A3).replace('[', '|').replace(']', '|')

            line1 = "$" + detA_s + ' = ' + pl_mi(n, 1, True) + brackets(A[0, n - 1]) + detA1_s + pl_mi(n, 2) + \
                    brackets(A[1, n - 1]) + detA2_s + pl_mi(n, 3) + brackets(A[2, n - 1]) + detA3_s + '$'

            line2 = '$' + detA_s + ' = ' + pl_mi(n, 1, True) + brackets(A[0, n - 1]) + "\cdot (" + sp.latex(sp.det(A1))\
                    + ")" + pl_mi(n, 2) + brackets(A[1, n - 1]) + "\cdot (" + sp.latex(sp.det(A2)) + ")" + \
                    pl_mi(n, 3) + brackets(A[2, n - 1]) + "\cdot (" + sp.latex(sp.det(A3)) + ')$'

            line3 = '$' + detA_s + ' = ' + sp.latex(sp.simplify(sp.det(A))) + '$'

        # Display step by step computation of determinant
        display(Latex(line1))
        display(Latex(line2))
        display(Latex(line3))
    # Only print the determinant without any step
    else:
        display(Latex("$" + detA_s + "=" + sp.latex(sp.det(A)) + "$"))


def valeurs_propres(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError("A should be a square matrix")

    l = sp.symbols('\lambda')
    n = A.shape[0]
    poly = sp.det(A - l * sp.eye(n))
    poly_exp = sp.expand(poly)
    poly_factor = sp.factor(poly)

    det_str = sp.latex(poly_exp) + "=" + sp.latex(poly_factor)

    display(Latex("On cherche les valeurs propres de la matrice $ A=" + latexp(A) + "$."))
    display(Latex("Le polynome caractéristique de $A$ est: $$\det(A- \lambda I)= " + det_str + "$$"))

    eq = sp.Eq(poly, 0)
    sol = sp.solve(eq, l)
    if len(sol) > 1:
        display(Latex("Les racines du polynôme caractéristique sont $" + sp.latex(sol) + "$."))
        display(Latex("Ces racines sont les valeurs propres de la matrice $A$."))
    else:
        display(Latex("L'unique racine du polynôme caractéristique est" + str(sol[0])))


def texVector(v):
    """
    Return latex string for vertical vector
    Input: v, 1D np.array()
    """
    n = v.shape[0]
    return al.texMatrix(v.reshape(n, 1))


def check_basis(sol, prop):
    """
    Checks if prop basis is equivalent to sol basis
    @param sol: verified basis, 2D numpy array, first dim: vector indexes, second dim: idx of element in a basis vect
    @param prop: proposed basis
    @return: boolean
    """

    prop = np.array(prop, dtype=np.float64)

    # number of vector in basis
    n = len(sol)

    # Check dimension of proposed eigenspace
    if n != len(prop):
        display(Latex("Le nomber de vecteur(s) propre(s) donné(s) est incorrecte. " +
                      "La dimension de l'espace propre est égale au nombre de variable(s) libre(s)."))
        return False
    else:
        # Check if the sol vector can be written as linear combination of prop vector
        # Do least squares to solve overdetermined system and check if sol is exact
        A = np.transpose(prop)
        lin_comb_ok = np.zeros(n, dtype=bool)

        for i in range(n):
            x, _, _, _ = np.linalg.lstsq(A, sol[i], rcond=None)
            res = np.sum((A @ x - sol[i]) ** 2)
            lin_comb_ok[i] = res < 10 ** -13

        return np.all(lin_comb_ok)


def eigen_basis(A, l, prop_basis=None, disp=True, return_=False):
    """
    Display step by step method for finding a basis of the eigenspace of A associated to eigenvalue l
    Eventually check if the proposed basis is correct. Display or not
    @param A: Square sympy Matrix with real coefficients
    @param l: real eigen value of A (float or int)
    @param prop_basis: Proposed basis: list of base vector (type list of list of floats)
    @param disp: boolean if display the solution. If false it displays nothing
    @param return_: boolean if return something or nothing
    @return: basis: a correct basis for the eigen space (2D numpy array)
            basic_idx: list with indices of basic variables of A - l*I
            free_idx: list with indices of free variables of A - l*I
    """
    if not A.is_Matrix:
        raise ValueError("A should be a sympy Matrix.")

    # Check if A is square
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError('A should be a square matrix.')

    # Compute eigenvals in symbolic
    eig = A.eigenvals()
    eig = list(eig.keys())

    # Deal with complex number (removal)
    complex_to_rm = []

    for idx, el in enumerate(eig):
        if not el.is_real:
            complex_to_rm.append(idx)

    for index in sorted(complex_to_rm, reverse=True):
        del eig[index]

    eig = np.array(eig)

    # evaluate symbolic expression
    eig_eval = np.array([float(el) for el in eig])

    # Check that entered eigenvalue is indeed an eig of A
    if np.all(abs(l - eig_eval) > 1e-10) and len(eig) > 0:
        display(Latex("$\lambda$ n'est pas une valeur propre de $A$."))
        return None, None, None

    # Change value of entered eig to symbolic expression (for nice print)
    l = eig[np.argmin(np.abs(l - eig))]

    I = sp.eye(n)
    Mat = A - l * I
    b = np.zeros(n)

    if disp:
        display(Latex("On a $ A = " + latexp(A) + "$."))
        display(Latex("On cherche une base de l'espace propre associé à $\lambda = " + str(l) + "$."))

    # ER matrix
    e_Mat, basic_idx = Mat.rref()

    # Idx of basic and free varialbe
    basic_idx = list(basic_idx)
    basic_idx.sort()
    free_idx = [idx for idx in range(n) if idx not in basic_idx]
    free_idx.sort()

    n_free = len(free_idx)

    # String to print free vars
    free_str = ""
    for i in range(n):
        if i in free_idx:
            free_str += "x_" + str(i + 1) + " \ "

    # Display echelon matrix
    if disp:
        display(Latex("On échelonne la matrice du système $A -\lambda I = 0 \Rightarrow "
                      + al.texMatrix(np.array(Mat), np.reshape(b, (n, 1))) + "$"))
        display(Latex("On obtient: $" + al.texMatrix(np.array(e_Mat[:, :n]), np.reshape(b, (n, 1))) + "$"))
        display(Latex("Variable(s) libre(s): $" + free_str + "$"))

    # Build a list of n_free basis vector:
    # first dim: which eigenvector (size of n_free)
    # second dim: which element of the eigenvector (size of n)
    basis = np.zeros((n_free, n))
    for i in range(n_free):
        basis[i, free_idx[i]] = 1.0

    for idx, j in enumerate(free_idx):
        for i in basic_idx:
            basis[idx, i] = - float(e_Mat[i, j])

    # Show calculated basis
    basis_str = ""
    for idx, i in enumerate(free_idx):
        basis_str += "x_" + str(i + 1) + " \cdot" + texVector(basis[idx])
        if idx < n_free - 1:
            basis_str += " + "

    if disp:
        display(Latex("On peut donc exprimer la base de l'espace propre comme: $" + basis_str + "$"))

    if prop_basis is not None and disp:
        correct_answer = check_basis(basis, prop_basis)

        if correct_answer:
            display(Latex("La base donnée est correcte car on peut retrouver la base calculée ci-dessus" \
                          " avec une combinaison linéaire de la base donnée. "
                          "Aussi les deux bases ont bien le même nombre de vecteurs."))

        else:
            display(Latex("La base donnée est incorrecte."))

    if return_:
        return basis, basic_idx, free_idx


def generate_eigen_vector(basis, l, limit):
    """
    Function to generate a random eigenvector associated to a eigenvalue given a basis of the eigenspace
    The returned eigenvector is such that itself and its multiplication with the matrix will stay in range of limit
    in order to have a nice plot
    @param basis: basis of eigenspace associated to eigenvalue lambda
    @param l: eigenvalue
    @param limit: limit of the plot: norm that the engenvector or its multiplication with the matrix will not exceed
    @return: eigen vector (numpy array)
    """
    n = len(basis)
    basis_mat = np.array(basis).T
    basis_mat = basis_mat.astype(np.float64)
    coeff = 2 * np.random.rand(n) - 1
    vect = basis_mat @ coeff
    if abs(l) <= 1:
        vect = vect / np.linalg.norm(vect) * (limit - 1)
    else:
        vect = vect / np.linalg.norm(vect) * (limit - 1) / l
    return vect


def plot3x3_eigspace(A, xL=-10, xR=10, p=None, plot_vector=False):
    # To have integer numbers
    if p is None:
        p = xR - xL + 1

    n = A.shape[0]
    # Check 3 by 3
    if n != 3 or n != A.shape[1]:
        raise ValueError("A should be 3 by 3")

    w = A.eigenvals()
    w = list(w.keys())

    # Deal with complex number (removal)
    complex_to_rm = []

    for idx, el in enumerate(w):
        if not el.is_real:
            complex_to_rm.append(idx)

    for index in sorted(complex_to_rm, reverse=True):
        del w[index]
        display("Des valeurs propres sont complexes, on les ignore.")

    if len(w)==0:
        display("Toute les valeurs propres sont complexes.")
        return


    gr = 'rgb(102,255,102)'
    org = 'rgb(255,117,26)'
    # red = 'rgb(255,0,0)'
    blue = 'rgb(51, 214, 255)'
    colors = [blue, gr, org]
    s = np.linspace(xL, xR, p)
    t = np.linspace(xL, xR, p)
    tGrid, sGrid = np.meshgrid(s, t)
    data = []

    A_np = np.array(A).astype(np.float64)

    for i, l in enumerate(w):
        l_eval = float(l)
        basis, basic_idx, free_idx = eigen_basis(A, l_eval, disp=False, return_=True)
        n_free = len(basis)
        if n_free != len(free_idx):
            raise ValueError("len(basis) and len(free_idx) should be equal.")

        gr = 'rgb(102,255,102)'

        colorscale = [[0.0, colors[i]],
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

        X = [None] * 3

        if n_free == 2:
            X[free_idx[0]] = tGrid
            X[free_idx[1]] = sGrid
            X[basic_idx[0]] = tGrid * basis[0][basic_idx[0]] + sGrid * basis[1][basic_idx[0]]

            plot_obj = go.Surface(x=X[0], y=X[1], z=X[2],
                                  showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                  name="$ \lambda= " + sp.latex(l) + "$")


        elif n_free == 1:

            plot_obj = go.Scatter3d(x=t * basis[0][0], y=t * basis[0][1], z=t * basis[0][2],
                                    line=dict(colorscale=colorscale, width=4),
                                    mode='lines',
                                    name="$\lambda = " + sp.latex(l) + "$")
        elif n_free == 3:
            display(Latex("La dimension de l'espace propre de l'unique valeur propre est 3: tous les vecteurs" \
                          "$v \in \mathbb{R}^3 $ appartiennent à l'espace propre de la matrice $A$." \
                          "On ne peut donc pas reprensenter sous la forme d'un plan ou d'une droite."))
            return

        else:
            print("error")
            return

        data.append(plot_obj)

        if (plot_vector):
            v1 = generate_eigen_vector(basis, l_eval, xR)
            v2 = A_np @ v1

            data.append(go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]],
                                     line=dict(width=6),
                                     marker=dict(size=4),
                                     mode='lines+markers',
                                     name='$v_{' + sp.latex(l) + '}$'))

            data.append(go.Scatter3d(x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]],
                                     line=dict(width=6, dash='dash'),
                                     marker=dict(size=4),
                                     mode='lines+markers',
                                     name="$A \ v_{" + sp.latex(l) + "}$"))

        layout = go.Layout(
            showlegend=True,  # not there WHY???? --> LEGEND NOT YET IMPLEMENTED FOR SURFACE OBJECTS!!
            legend=dict(orientation="h"),
            autosize=True,
            width=800,
            height=800,
            scene=go.layout.Scene(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    range=[xL, xR]
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    range=[xL, xR]
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    range=[xL, xR]
                ),
                aspectmode="cube",
            )
        )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    return


def plot2x2_eigspace(A, xL = -10, xR = 10, p=None):
    if p is None:
        p = xR - xL + 1

    w = A.eigenvals()
    w = list(w.keys())

    # Deal with complex number (removal)
    complex_to_rm = []

    for idx, el in enumerate(w):
        if not el.is_real:
            complex_to_rm.append(idx)

    for index in sorted(complex_to_rm, reverse=True):
        del w[index]
        display("Une valeur propre est complexe, on l'ignore.")

    if len(w) == 0:
        display("Toute les valeurs propres sont complexes.")
        return

    data = []

    for i, l in enumerate(w):
        l_eval = float(l)
        basis, basic_idx, free_idx = eigen_basis(A, l_eval, disp=False, return_=True)

        n_free = len(basis)
        if n_free != len(free_idx):
            raise ValueError("len(basis) and len(free_idx) should be equal.")

        if n_free == 2:
            display(Latex("Tous les vecteurs du plan appartiennent à l'espace propre de A associé à $\lambda = " \
                          + sp.latex(l) + "$. On ne peut donc pas le représenter."))
            return
        else:
            t = np.linspace(xL, xR, p)

            trace = go.Scatter(x=t*basis[0][0], y=t*basis[0][1], marker=dict(size=6),
                                     mode='lines+markers', name="$\lambda = " + sp.latex(l) + "$")
            data.append(trace)

    layout = go.Layout(showlegend=True, autosize=True)

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    return


def plot_eigspace(A, xL=-10, xR=10, p=None):
    """
    Plot the eigenspaces associated to all eigenvalues of A
    @param A: Sympy matrix of shape (2,2) or (3,3)
    @param xL: Left limit of plot
    @param xR: Right limit of plot
    @param p: Number of points to use
    """
    n = A.shape[0]
    # Check 3 by 3 or 2 by 2
    if (n != 2 and n!=3) or n != A.shape[1]:
        raise ValueError("A should be 2 by 2 or 3 by 3.")

    if not A.is_Matrix:
        raise ValueError("A should be a sympy Matrix.")

    if n==2:
        plot2x2_eigspace(A, xL, xR, p)

    else:
        plot3x3_eigspace(A, xL, xR, p)


def latexp(A):
    """
    Function to output latex expression of a sympy matrix but with round parenthesis
    @param A: sympy matrix
    @return: latex string
    """
    return sp.latex(A, mat_delim='(', mat_str='matrix')


def ch8_8_ex_1(A, prop_answer):
    """
    Check if a matrix is diagonalisable.
    @param A: sympy square matrix
    @param prop_answer: boolean, answer given by the student
    @return:
    """
    if not A.is_Matrix:
        raise ValueError("A should be a sympy Matrix.")
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError('A should be a square matrix.')

    eig = A.eigenvects()
    dim_geom = 0

    for x in eig:
        dim_geom += len(x[2])
    answer = dim_geom == n

    if answer:
        display(Latex("Oui la matrice $A = " + latexp(A) + "$ est diagonalisable."))
    else:
        display(Latex("Non la matrice $A = " + latexp(A) + "$ n'est pas diagonalisable."))

    if answer == prop_answer:
        display(Latex("Votre réponse est correcte !"))
    else:
        display(Latex("Votre réponse est incorrecte."))


def isDiagonalizable(A):
    """
    Step by step method to determine if a given matrix is diagonalizable. This methods uses always (I think)
    the easiest way to determine it.
    @param A: sympy matrix
    @return: nothing
    """
    if not A.is_Matrix:
        raise ValueError("A should be a sympy Matrix.")

    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError('A should be a square matrix.')

    display(Latex("On cherche à déterminer si la matrice $A=" + latexp(A) + "$ de taille $n \\times n$ avec $n = " +
                  str(n) + "$ est diagonalisable."))

    if A.is_lower or A.is_upper:
        display(Latex("Les valeurs propres sont simple à trouver, ce sont les éléments diagonaux."))
    else:
        valeurs_propres(A)

    # Check if eigenvalue are all distincts
    eig = A.eigenvects()

    if len(eig) == n:
        display(Latex("On a $n$ valeurs propres distinctes. La matrice est donc diagonalisable."))
        return
    else:
        display(Latex("Les valeurs propres ne sont pas toutes distinctes. On va donc vérifier la multiplicité " +
                      "géométrique des valeurs propres ayant une multiplicité algébrique supérieur à 1."))

        # Some list to have info about eigenvalues with algebraic mult > 1
        idx = []
        eigenvalues = []
        mult_al = []
        mult_geo = []

        for i in range(len(eig)):
            if eig[i][1] > 1:
                idx.append(i)
                eigenvalues.append(eig[i][0])
                mult_al.append(eig[i][1])
                mult_geo.append(len(eig[i][2]))

        display(Latex("L'ensemble des valeurs propres ayant une multiplicité algébrique supérieur à 1 est " + str(
            eigenvalues) + "."))

        for i, l in enumerate(eigenvalues):
            display(Markdown("**On calcule la multiplicité géométrique pour $\lambda= " + sp.latex(l) +
                             "$ ayant une multiplicité algébrique de " + str(mult_al[i]) + ".**"))

            basis, basic, free = eigen_basis(A, l, prop_basis=None, disp=True, return_=True)
            display(Markdown("**La multiplicité géométrique pour $\lambda= " + sp.latex(l) + "$ est de " +
                             str(len(free)) + ".**"))
            if (len(free) < mult_al[i]):
                display(Markdown("**La multiplicité géométrique est strictement inférieur à la multiplicité"
                                 "algébrique + pour cette valeur propre. La matrice n'est donc pas diagonalisable.**"))
                return
            else:
                display(Latex("On a bien multiplicité algébrique = multiplicité géométrique pour cette valeur propre."))

        display(Markdown("**Toutes les valeurs propres ont une multiplicité algébrique et géométrique égales." +
                         " La matrice $A$ est donc bien diagonalisable !**"))