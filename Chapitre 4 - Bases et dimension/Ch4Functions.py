import sys
sys.path.insert(0,'..')
import Librairie.AL_Fct as al
sys.path.pop(0)

import numpy as np
from IPython.display import display, Markdown, Latex
import plotly.graph_objs as go
import plotly
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import plotly.express as px
import sympy as sp

def ch4_1_2ex1(solution):
    v = np.array([[1, 0, 2], [0, 1, 0], [1, 1, 1]]).transpose()
    e = np.array([3, 5, 4])
    s = np.array(solution)
    r = v @ s
    if np.allclose(e, r):
        display(Markdown("**Correction:** C'est correct!"))
    else:
        display(Markdown("**Correction:** C'est incorrect car: $\lambda_1 v_1 + \lambda_2 v_2 + \lambda_3 v_3 = \\begin{pmatrix} %s \\\\ %s \\\\ %s \end{pmatrix} \\neq \\begin{pmatrix} %s \\\\ %s \\\\ %s \end{pmatrix}$" % (r[0], r[1], r[2], e[0], e[1], e[2])))

    w = s * v

    cumsum = np.cumsum(np.insert(w, 0, 0, axis=1), axis=1).transpose()

    colors = px.colors.qualitative.Plotly
    global color_index
    color_index = 0

    data = []
    def addVector(start, v):
        global color_index

        color = colors[color_index]
        color_index = (color_index + 1) % len(colors)

        end = start + v
        trace = go.Scatter3d(
            x=[start[0], end[0], None],
            y=[start[1], end[1], None],
            z=[start[2], end[2], None],
            mode='lines',
            name=str(v),
            line=dict(color=color, width=4)
        )
        norm = np.sqrt(np.sum(v * v))
        n = v if norm == 0 else v / norm
        n = 1.5 * n
        c_end = end - 0.37 * n
        cone = go.Cone(x=[c_end[0]], y=[c_end[1]], z=[c_end[2]], u=[n[0]], v=[n[1]], w=[n[2]], name=str(v), colorscale=[[0, color], [1, color]], hoverinfo="none", showscale=False)

        data.append(trace)
        data.append(cone)

    addVector(np.zeros(3), e)

    for i in range(len(cumsum) - 1):
        start = cumsum[i]
        v = cumsum[i + 1] - start
        addVector(start, v)

    fig = go.Figure(data=data)

    fig.show()

def ch4_1_2ex2():
    radio = widgets.RadioButtons(
        options=['Oui, les vecteurs sont dépendants', 'Non, les vecteurs sont indépendants'],
        layout={'width': 'max-content'},
        value=None,
        description='Réponse:',
    )

    button = widgets.Button(description='Vérifier')

    out = widgets.Output()

    display(radio)
    display(button)
    display(out)

    def verification_2(e):
        if radio.value is not None:
            out.clear_output()
            with out:
                if radio.value.startswith('Non'):
                    display(Markdown("C'est incorrect, il existe $\lambda_1$, $\lambda_2$ et $\lambda_3$ tels que $\lambda_1 v_1 + \lambda_2 v_2 + \lambda_3 v_3 - \\begin{pmatrix} 3 \\\\ 0 \\\\ 4 \end{pmatrix} = 0$."))
                else:
                    display(Markdown("C'est correct!"))

    button.on_click(verification_2)

def ch4_1_2ex3(answer):
    radio = widgets.RadioButtons(
        options=['Oui, les vecteurs sont dépendants', 'Non, les vecteurs sont indépendants'],
        layout={'width': 'max-content'},
        value=None,
        description='Réponse:',
    )

    button = widgets.Button(description='Vérifier')

    out = widgets.Output()

    display(radio)
    display(button)
    display(out)

    def verification_3(e):
        if radio.value is not None:
            out.clear_output()
            with out:
                if radio.value.startswith('Oui') == answer:
                    display(Markdown("C'est correct!"))
                else:
                    display(Markdown("C'est incorrect!"))

    button.on_click(verification_3)

def ch4_1_2ex3a():
    ch4_1_2ex3(True)

def ch4_1_2ex3b():
    ch4_1_2ex3(True)

def ch4_1_2ex3c():
    ch4_1_2ex3(False)
    
def ch4_3ex1(answer, reason, callback, options=['Oui, les vecteurs forment une base', 'Non, les vecteurs ne forment pas une base']):
    radio = widgets.RadioButtons(
        options=options,
        layout={'width': 'max-content'},
        value=None,
        description='Réponse:',
    )

    button = widgets.Button(description='Vérifier')

    out = widgets.Output()

    display(radio)
    display(button)
    display(out)

    def verification(e):
        if radio.value is not None:
            out.clear_output()
            with out:
                if options.index(radio.value) == answer:
                    display(Markdown("C'est correct!<br />%s" % reason))
                else:
                    display(Markdown("C'est incorrect!<br />%s" % reason))
                callback()

    button.on_click(verification)

def plot_vectors(vectors, selected, solution):
    v = np.array(vectors).transpose()
    e = np.array(selected)
    s = np.array(solution)
    r = v @ s
    w = s * v
    
    cumsum = np.cumsum(np.insert(w, 0, 0, axis=1), axis=1).transpose()
    
    colors = px.colors.qualitative.Plotly
    global color_index
    color_index = 0
    
    data = []
    def addVector(start, v):
        global color_index
        
        color = colors[color_index]
        color_index = (color_index + 1) % len(colors)
        
        end = start + v
        trace = go.Scatter3d(
            x=[start[0], end[0], None],
            y=[start[1], end[1], None],
            z=[start[2], end[2], None],
            mode='lines',
            name=str(v),
            line=dict(color=color, width=4)
        )
        norm = np.sqrt(np.sum(v * v))
        n = v if norm == 0 else v / norm
        n = 0.5 * n
        c_end = end - 0.37 * n
        cone = go.Cone(x=[c_end[0]], y=[c_end[1]], z=[c_end[2]], u=[n[0]], v=[n[1]], w=[n[2]], name=str(v), colorscale=[[0, color], [1, color]], hoverinfo="none", showscale=False)
        
        data.append(trace)
        data.append(cone)
        
    addVector(np.zeros(3), e)

    for i in range(len(cumsum) - 1):
        start = cumsum[i]
        v = cumsum[i + 1] - start
        addVector(start, v)

    fig = go.Figure(data=data)

    fig.show()
    
def ch4_3ex1a():
    ch4_3ex1(1, """

En effet, les quatres vecteurs ne sont pas linéairement indépendants (Déf. 1-2) car il est possible d'exprimer l'un en fonction d'une combinaison des autres. Par exemple :
$$\\begin{pmatrix} 1 \\\\ 1 \\\\ 1 \end{pmatrix} = \\begin{pmatrix} 1 \\\\ 0 \\\\ 1 \end{pmatrix} + \\frac{1}{2} \\begin{pmatrix} 0 \\\\ 2 \\\\ 0 \end{pmatrix} + 0 \\begin{pmatrix} 0 \\\\ 0 \\\\ 3 \end{pmatrix}$$
Comme tous les vecteurs sont issus de $\mathbb{R}^3$ on peut facilement représenter la combinaison dans l'espace :
    """, lambda: plot_vectors([[1, 0, 1], [0, 2, 0], [0, 0, 3]], [1, 1, 1], [1, 0.5, 0]))
    
def ch4_3ex1b():
    ch4_3ex1(1, """On ne peut pas générer $\mathbb{R}^3$ à partir de cet ensemble. En effet, on ne peut par exemple pas représenter le vecteur $\\begin{pmatrix} 0 \\\\ 0 \\\\ 1 \end{pmatrix} \\in \mathbb{R}^3$ comme combinaison linéaire de $\\begin{pmatrix} 1 \\\\ 0 \\\\ 0 \end{pmatrix}$ et $\\begin{pmatrix} 0 \\\\ 1 \\\\ 0 \end{pmatrix}$.

Graphiquement les deux vecteurs ne peuvent engendrer qu'un plan - et non un espace :
    """, lambda: plot_vectors([[1, 0, 0]], [0, 1, 0], [1]))
    
def ch4_3ex2():
    vecs = np.array([[1, 1, 1], [0, 1, 2], [2, 1, 4], [2, 1, 0], [1, 0, -1]])
    
    select = widgets.SelectMultiple(
        options=['v1', 'v2', 'v3', 'v4', 'v5'],
        description='Sélection :',
        disabled=False
    )

    button = widgets.Button(description='Vérifier')

    out = widgets.Output()

    def callback(e):
        answer = [int(v[1:])-1 for v in select.value]
        out.clear_output()
        with out:
            if len(answer) == 0: # Empty
                pass
            elif len(answer) < 3:
                display(Markdown("C'est incorrect!<br />La solution entrée ne permet pas d'engendrer R^3."))
            elif len(answer) > 3:
                display(Markdown("C'est incorrect!<br />La solution entrée contient des vecteurs qui sont dépendants."))
            else:
                mat = np.array([vecs[answer[0]], vecs[answer[1]], vecs[answer[2]]]).transpose()
                det = np.linalg.det(mat)
                if det == 0:
                    display(Markdown("C'est incorrect!<br />La solution entrée contient des vecteurs qui sont dépendants."))
                else: # Correct
                    display(Markdown("C'est correct!<br />Il s'agit d'_une_ base."))

    button.on_click(callback)

    display(select)
    display(button)
    display(out)
    

def ch4_3ex3():
    ch4_3ex1(1, """

$\mathbb{P}(\mathbb{R})$ est un espace infini, il ne peut être généré que par des bases infinies (par exemple, $\{1, x, x^2, x^3, ...\}$).

Cet ensemble ne forme donc pas une base pour $\mathbb{P}(\mathbb{R})$.
    """, lambda: None)
    
def ch4_3ex4():
    ch4_3ex1(2, """

L'espace engendré par cet ensemble est $\mathbb{R}^{2 \\times 2}$ et donc sa dimension est $4$.

Attention cet ensemble n'est pas une base de $\mathbb{R}^{2 \\times 2}$ car ses éléments sont linéairement dépendants !

    """, lambda: None, ['2', '3', '4', '5'])

def ch4_4_5ex1():
    vs = np.array([[1, 1, 1], [2, 0, 3], [4, 0, 0], [1, 0, 0]])
    
    select = widgets.SelectMultiple(
        options=['v1', 'v2', 'v3', 'v4'],
        description='Sélection :',
        disabled=False
    )
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        answer = [int(v[1:])-1 for v in select.value]
        out.clear_output()
        with out:
            if len(answer) == 0: # Empty
                pass
            elif len(answer) < 3:
                display(Markdown("C'est incorrect!<br />La solution entrée ne permet pas d'engendrer $\\mathbb{R}^3$, et n'est donc pas une base."))
            elif len(answer) > 3:
                display(Markdown("C'est incorrect!<br />La solution entrée engendre $\\mathbb{R}^3$ mais n'est pas une base."))
            else:
                mat = np.array([vs[answer[0]], vs[answer[1]], vs[answer[2]]]).transpose()
                if np.linalg.matrix_rank(mat) != len(mat):
                    display(Markdown("C'est incorrect!<br />La solution entrée ne permet pas d'engendrer $\\mathbb{R}^3$, et n'est donc pas une base."))
                else: # Correct
                    display(Markdown("C'est correct!<br />Il s'agit d'_une_ base de $\\mathbb{R}^3$."))

    button.on_click(callback)

    display(select)
    display(button)
    display(out)

def ch4_4_5ex2(v):
    v = np.array(v)
    vs = np.array([[1, 4, 3, 0], [1, 0, 0, 1], [0, 1, 1, 0], v.flatten()])
    is_base = np.linalg.matrix_rank(vs) == len(vs)
    
    out = widgets.Output()
    display(out)
    
    with out:
        if is_base:
            display(Markdown("C'est correct!"))
        else:
            display(Markdown("C'est incorrect, ce vecteur ne permet pas de former une base."))
            
def ch4_6ex1():
    text = widgets.IntText(
        description='Réponse :',
        disabled=False
    )
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r = text.value
        feedback = ""
        is_correct = False
        if r == 6:
            feeback = "Comme la matrice n'est pas nulle, au moins une variable n'est pas libre."
        elif r >= 7:
            feedback = "La dimension de l'espace des solutions ne peut excéder la dimension de l'espace des variables."
        elif r == 5:
            is_correct = True
            feedback = "Le nombre maximal de variables libres dans ce système est $5$. Par la proposition 1 on en déduit la dimension maximale de l'espace des solutions du système homogène $AX = 0$."
        elif r >= 2 and r <= 4:
            feedback = "Ce n'est pas le nombre maximal de variables libres dans ce système."
        elif r <= 1:
            feedback = "Le nombre maximal de variables libres dans ce système ne peut être inférieur à $2$ ($\\text{nb. colonnes} - \\text{nb. lignes}$)."
            
        correct_text = "C'est correct!<br />"
        incorrect_text = "C'est incorrect.<br />"
        
        with out:
            display(Markdown((correct_text if is_correct else incorrect_text) + feedback))
        
    button.on_click(callback)
    
    display(text)
    display(button)
    display(out)

def ch4_6ex2():
    text = widgets.IntText(
        description='Réponse :',
        disabled=False
    )
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r = text.value
        
        with out:
            if r == 2:
                display(Markdown("C'est correct!<br />Le nombre de variables libres est $3$, par la proposition 1 on trouve la dimension de l'espace des solutions."))
            else:
                display(Markdown("C'est incorrect."))
        
    button.on_click(callback)
    
    display(text)
    display(button)
    display(out)

def ch4_6ex3(base):
    base = np.array(base)
    
    out = widgets.Output()
    display(out)

    with out:
        out.clear_output()

        feedback = ""
        is_correct = False
        
        s = base.shape

        if len(base) == 0:
            feedback = "L'ensemble ne peut pas être vide."
        elif len(s) != 2 or s[1] != 5:
            feedback = "Le format des vecteurs n'est pas bon."
        elif s[0] < 2:
            feedback = "L'ensemble entré ne contient pas assez de vecteurs pour engendrer toutes les solutions du système."
        elif s[0] > 2:
            feedback = "L'ensemble entré n'est pas une base."
        else:
            expected = np.array(sp.Matrix([[6, 1, -2, 0, 1], [8, 2, -3, 1, 0]]).rref()[0])
            actual = np.array(sp.Matrix(base).rref()[0])
            
            if not np.array_equal(actual, expected):
                feedback = "L'ensemble entré n'engendre pas l'espace solution du système."
            else:
                is_correct = True
            
        correct_text = "C'est correct!<br />"
        incorrect_text = "C'est incorrect.<br />"
            
        display(Markdown((correct_text if is_correct else incorrect_text) + feedback))

def ch4_7_8ex1():
    select = widgets.SelectMultiple(
        options=['0', '1', '2', '3', '4', '5'],
        description='Réponse :',
        disabled=False
    )
    
    button = widgets.Button(description='Vérifier')

    out = widgets.Output()

    display(select)
    display(button)
    display(out)

    def verification(e):
        if len(select.value) > 0:
            out.clear_output()
            with out:
                other = "<br>La dimension de l'espace des solutions dépend du nombre de variables libres : dans ce système celle-ci peut être au plus $5-3=2$."
                if select.value != ('0', '1', '2'):
                    display(Markdown("C'est incorrect!" + other))
                else:
                    display(Markdown("C'est correct." + other))

    button.on_click(verification)
    
def ch4_7_8ex2():
    text = widgets.IntText(description='Réponse :')
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r = text.value
        
        with out:
            other = "<br />En effet, comme la solution de $S_1$ est un plan les deux équations sont dépendantes. La solution de $S_2$ est donc aussi un plan, donc sa dimension est $2$."
            if r == 2:
                display(Markdown("C'est correct!" + other))
            else:
                display(Markdown("C'est incorrect." + other))
        
    button.on_click(callback)
    
    display(text)
    display(button)
    display(out)

def ch4_7_8ex3(answer, explanation):
    text = widgets.IntText(description='Réponse :')
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r = text.value
        
        with out:
            other = "<br />" + explanation
            if r == answer:
                display(Markdown("C'est correct!" + other))
            else:
                display(Markdown("C'est incorrect." + other))
        
    button.on_click(callback)
    
    display(text)
    display(button)
    display(out)

def ch4_7_8ex3a():
    ch4_7_8ex3(2, "$V$ est un sous-ensemble de $W$, donc $\dim (W + V) = \dim W = 2$.")

def ch4_7_8ex3b():
    ch4_7_8ex3(3, "$W$ et $V$ sont deux espaces indépendants, donc $\dim (W + V) = \dim W + \dim V = 2 + 1 = 3$.")

def ch4_7_8ex3c():
    ch4_7_8ex3(1, "La somme d'un même espace n'a pas d'effet : $\dim (V + V) = \dim V = 1$.")
    
def ch4_7_8ex3d():
    ch4_7_8ex3(2, "$V_1$ et $V_2$ sont deux espaces indépendants, donc $\dim (V_1 + V_2) = \dim V_1 + \dim V_2 = 1 + 1 = 2$.")

def ch4_7_8ex3e():
    ch4_7_8ex3(3, "$\dim (W_1 + W_2) = 2 + 2 - 1 = 3$.")
    
def ch4_7_8ex3f():
    ch4_7_8ex3(3, "$\dim (U + X) = 0 + 3 - 0 = 3$.")
    
    
def ch4_9ex1():
    r1 = 'Le rang ligne de 𝐴 est plus petit ou égal à 2, car c\'est un sous-espace vectoriel de ℝ2.'
    r2 = 'Le rang ligne de 𝐴 est plus petit ou égal à 3, car c\'est un sous-espace vectoriel de ℝ3.'
    r3 = 'Le rang ligne de 𝐴 est plus petit ou égal à 3, car engendré par 3 vecteurs.'
    r4 = 'Le rang ligne de 𝐴 est plus petit ou égal à 2, car engendré par 2 vecteurs.'
    r5 = 'Le rang colonne de A est plus petit ou égal à 2.'

    select = widgets.SelectMultiple(
        options=[(r1, 1), (r2, 2), (r3, 3), (r4, 4), (r5, 5)],
        description='Sélection :',
        disabled=False,
        layout=Layout(width='auto', height='100px')
    )

    button = widgets.Button(description='Vérifier')
    out = widgets.Output()

    def callback(e):
        out.clear_output()
        with out:
            if (1 in select.value or 3 in select.value):
                    print('Mauvaise réponse. \nAttention à ne pas confondre les espaces des lignes et colonnes')
            elif (2 not in select.value or 4 not in select.value or 5 not in select.value):
                print('Il manque au moins une réponse.')
            elif (2 in select.value and 4 in select.value and 5 in select.value):
                print('Correct !')
                     
    button.on_click(callback)

    display(select)
    display(button)
    display(out)
    
def ch4_9ex2_1():
    text = widgets.IntText(
        description='Réponse :',
        disabled=False
    )
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r = text.value
        
        with out:
            if r == 2:
                display(Markdown("Par la proposition 4 c'est  correct!"))
            elif r > 2:
                display(Markdown("Faux, c'est trop grand"))
            else:
                display(Markdown("Faux, c'est trop petit"))
        
    button.on_click(callback)
    
    display(text)
    display(button)
    display(out)

def ch4_9ex2_2():
    select = widgets.SelectMultiple(
        options=['E1', 'E2', 'E3', 'E4'],
        description='Sélection :',
        disabled=False,
        layout=Layout(width='auto', height='auto')
    )

    button = widgets.Button(description='Vérifier')

    out = widgets.Output()

    def callback(e):
        
        out.clear_output()
        with out:
            if len(select.value) <= 0: # Empty
                pass
            elif len(select.value) > 1:
                display(Markdown("Un seul de ces ensembles génère exactement l'espace ligne de A"))
            elif 'E1' not in select.value:
                display(Markdown("Faux"))
            else:
                display(Markdown("Correct !"))
    button.on_click(callback)
    display(select)
    display(button)
    display(out)
    


def ch4_9ex2_3():
    text = widgets.IntText(description='Réponse :', disabled=False)

    button = widgets.Button(description='Vérifier')
    button2 = widgets.Button(description='Solution', disabled=True)
    box = widgets.HBox(children=[button,button2])
    out = widgets.Output()

    def callback(e):
        out.clear_output()
        button2.disabled = False
        with out:
            if(text.value == 2):
                print('Correct !')
            else: 
                print('False, try again or check the solution')
    
    def solution(e):
        out.clear_output()
        with out:
            A = np.array([[1, 2, 3], [0, 1, 2]])
            A_t = A.transpose()
            display(Markdown('Pour trouver le rang colonne de $A$, nous utilisons la remarque 2 et trouvous le rang ligne de la transposée de $A$.'))
            display(Markdown('Par les propositions 3 et 4, nous échelonnons la matrice transposée et observont le nombre de lignes qui contiennent des pivots'))
            M = al.echelonMat('E', A_t)
            display(Markdown('Ainsi le rang colonne de $A$ est 2'))
                      
    button.on_click(callback)
    button2.on_click(solution)
    
    display(text)
    display(box)
    display(out)
    
def ch4_10ex1_1(A, b):
    A_sol = [[1, 4, 3, 4],[2, 6, 5, 8],[1, 0, 1, 4]]
    b_sol = [1, 1, 1]
    if A == [] or b == []:
        print("Au moins une des deux entrée est vide")
    elif not (len(A) == len(b)):
        print("Les tailles de la matrice et du vecteur ne correspondent pas")
    else:
        if A == A_sol:
            if b == b_sol:
                print("Correct !")
            else:
                print("Le vecteur b est faux, votre reponse correspond au système suivant:")
                al.printSyst(A, b)
        elif b == b_sol:
            print("La Matrice A est fausse, votre reponse correspond au système suivant:")
            al.printSyst(A, b)
        else:
            print("Faux, votre reponse correspond au système suivant:")
            al.printSyst(A, b)
            
def ch4_10ex1_2_display():
    A_sol = np.array([[1, 4, 3, 4],[2, 6, 5, 8],[1, 0, 1, 4]])
    A_sol_t = A_sol.transpose()
    M = al.echelonMat('E', A_sol_t)
    
def ch4_10ex1_2_1():
    
    text_rang = widgets.IntText()
    box = VBox([Label('Rang colonne de A:'), text_rang])
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        with out:
            if(text_rang.value == 2):
                print('Correct !')
            else: 
                print('False, try again or check the solution')
                     
    button.on_click(callback)
    display(box)
    display(button)
    display(out)

def ch4_10ex1_2_2(base):
  
    base_sol = [[1,2,1],[0,1,2]]
    
    if base_sol == []:
        print("L'entrée est vide")
    else:
        if base == base_sol:
            print("Correct !")
        else:
            print("Faux")

def ch4_10ex1_3():
 
    radio = widgets.RadioButtons(
        options=['Yes', 'No'],
        description='Answer:',
        disabled=False
    )

    button = widgets.Button(description='Vérifier')
    out = widgets.Output()

    def callback(e):
        out.clear_output()
        with out:
            if (radio.value == "Yes"):
                    print('Mauvaise réponse.')
            else: 
                print('Correct !')
                     
    button.on_click(callback)
    display(radio)
    display(button)
    display(out)

    
def ch4_11ex1(base):
    base_solution = [[1,2],[3,2]]
    is_correct = all(item in base_solution for item in base)
    if is_correct:
        is_correct = all(item in base for item in base_solution)
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def ch4_11ex2aB1(vB1):
    v = [5,6]
     
    if vB1==v:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def ch4_11ex2aB2(vB2):
    v = [2,1]
     
    if vB2==v:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def ch4_11ex2bB1(uB1):
    u = [8,4]
     
    if uB1==u:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def ch4_11ex2bB2(uB2):
    u = [-1,3]
     
    if uB2==u:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def ch4_11ex2cB1(wB1):
    w = [5,4]
     
    if wB1==w:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def ch4_11ex2cB2(wB2):
    w = [0.5,1.5]
     
    if wB2==w:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def ch4_11ex3B1(sB1):
    s = [13,10]
     
    if sB1==s:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def ch4_11ex3B2(sB2):
    s = [1,4]
     
    if sB2==s:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def ch4_12ex1():
    nbr_ligne = widgets.IntText(
        description='Nombre de lignes : \n',
        disabled=False
    )
    nbr_colonne = widgets.IntText(
        description='Nombre de colonne : \n',
        disabled=False
    )
    
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r_l = nbr_ligne.value
        r_c = nbr_colonne.value
        with out:
            if r_l == 5 and r_c == 6:
                display(Markdown("C'est correct!"))
            else:
                display(Markdown("C'est incorrect."))
        
    button.on_click(callback)
    
    display(nbr_ligne)
    display(nbr_colonne)
    display(button)
    display(out)

def ch4_12ex2():
    text = widgets.IntText(
        description='Réponse :',
        disabled=False
    )
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r = text.value
        
        with out:
            if r == 2:
                display(Markdown("C'est correct!"))
            else:
                display(Markdown("C'est incorrect."))
        
    button.on_click(callback)
    
    display(text)
    display(button)
    display(out)

def ch4_12ex3a():
    text = widgets.IntText(
        description='Réponse :',
        disabled=False
    )
    button = widgets.Button(description='Vérifier')
    out = widgets.Output()
    
    def callback(e):
        out.clear_output()
        r = text.value
        
        with out:
            if r == 3:
                display(Markdown("C'est correct!"))
            else:
                display(Markdown("C'est incorrect."))
        
    button.on_click(callback)
    
    display(text)
    display(button)
    display(out)
    
def ch4_12ex3b(base):
    base = base + [[1,4,0,2,0,1],[0,1,1,2,1,0],[0,0,1,3,3,2]]
    base = np.array(base)
    out = widgets.Output()
    display(out)

    with out:
        out.clear_output()

        feedback = ""
        is_correct = False
        
        s = base.shape
        

        if len(base) == 0:
            feedback = "Les vecteurs nuls ne peuvent pas ."
        elif len(s) != 2 or s[1] != 6:
            feedback = "Le format des vecteurs n'est pas bon."
        elif s[0] < 6:
            feedback = "L'ensemble entré ne contient pas assez de vecteurs pour engendrer toutes les solutions du système."
        elif s[0] > 6:
            feedback = "L'ensemble entré contient trop de vecteurs pour être une famille libre."
        else:
            expected = np.array(sp.Matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]]).rref()[0])
            actual = np.array(sp.Matrix(base).rref()[0])
            
            if not np.array_equal(actual, expected):
                feedback = "L'ensemble entré n'engendre pas l'espace solution du système."
            else:
                is_correct = True
            
        correct_text = "C'est correct!<br />"
        incorrect_text = "C'est incorrect.<br />"
            
        display(Markdown((correct_text if is_correct else incorrect_text) + feedback))
