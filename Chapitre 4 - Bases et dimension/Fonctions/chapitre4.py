import sys
sys.path.insert(0, './../')
import Librairie.AL_Fct as al

import numpy as np
from IPython.display import display, Markdown, Latex
import plotly.graph_objs as go
import plotly
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, Layout, HBox, VBox, Label
import matplotlib
import plotly.express as px
import sympy as sp

# Chapitres 4.1 - 4.2
def Ex1Chapitre4_1(solution):
    v = np.array([[1, 0, 2], [0, 1, 0], [1, 1, 1]]).transpose()
    e = np.array([3, 5, 4])
    s = np.array(solution)
    r = v @ s
    if np.allclose(e, r):
        display(Markdown("**Correction:** C'est correct!"))
    else:
        display(Markdown("**Correction:** C'est incorrect car: $\lambda_1 v_1 + \lambda_2 v_2 + \lambda_3 v_3 = \\begin{pmatrix} %s \\\\ %s \\\\ %s \end{pmatrix} \\neq \\begin{pmatrix} %s \\\\ %s \\\\ %s \end{pmatrix}$" % (r[0], r[1], r[2], e[0], e[1], e[2])))
    
    #plotly.offline.init_notebook_mode(connected=True)
    
    w = s * v
    
    x = np.cumsum(np.insert(w[0], 0, 0))
    y = np.cumsum(np.insert(w[1], 0, 0))
    z = np.cumsum(np.insert(w[2], 0, 0))

    pairs = [(0,1), (1,2), (2, 3)]

    x_lines = list()
    y_lines = list()
    z_lines = list()

    for p in pairs:
        for i in range(2):
            x_lines.append(x[p[i]])
            y_lines.append(y[p[i]])
            z_lines.append(z[p[i]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    trace1 = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines+markers',
        name='Combinaison linéaire entrée',
        marker_symbol='cross'
    )

    trace2 = go.Scatter3d(
        x=[0, e[0]],
        y=[0, e[1]],
        z=[0, e[2]],
        mode='lines+markers',
        name='Résultat attendu',
    )

    fig = go.Figure(data=[trace1, trace2])

    fig.show()
    

def Ex2Chapitre4_1():
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
                if radio.value.startswith('Oui'):
                    display(Markdown("C'est incorrect, il existe $\lambda_1$, $\lambda_2$ et $\lambda_3$ tels que $\lambda_1 v_1 + \lambda_2 v_2 + \lambda_3 v_3 - \\begin{pmatrix} 3 \\\\ 0 \\\\ 4 \end{pmatrix} = 0$."))
                else:
                    display(Markdown("C'est correct!"))

    button.on_click(verification_2)

    
def Ex3Chapitre4_1(answer):
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

    
# Chapitre 4.3
def Ex1Chapitre4_3(answer, reason, callback, options=['Oui, les vecteurs forment une base', 'Non, les vecteurs ne forment pas une base']):
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

    
def plot_1(vectors, selected, solution):
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
        cone = go.Cone(x=[end[0]], y=[end[1]], z=[end[2]], u=[n[0]], v=[n[1]], w=[n[2]], name=str(v), colorscale=[[0, color], [1, color]], hoverinfo="none", showscale=False)
        
        data.append(trace)
        data.append(cone)
        
    addVector(np.zeros(3), e)

    for i in range(len(cumsum) - 1):
        start = cumsum[i]
        v = cumsum[i + 1] - start
        addVector(start, v)

    fig = go.Figure(data=data)

    fig.show()
    
    
def Ex2Chapitre4_3():
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
    
    
# Chapitres 4.4 - 4.5
def Ex1Chapitre4_4():
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

    
def Ex2Chapitre4_4(v):
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


# Chapitre 4.6
def Ex1Chapitre4_6():
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

    
def Ex2Chapitre4_6():
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
    
def Ex3Chapitre4_6(base):
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
        

# Chapitres 4.7 - 4.8
def Ex1Chapitre4_7():
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
    
def Ex2Chapitre4_7():
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
    
def Ex3Chapitre4_7(answer, explanation):
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
    

# Chapitre 4.9
def Ex1Chapitre4_9():
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
    
    
def Ex2_1Chapitre4_9():
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
    
    
def Ex2_2Chapitre4_9():
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
    
    
def Ex2_3Chapitre4_9():
    text = widgets.IntText(description='Réponse :', disabled=False)

    button = widgets.Button(description='Vérifier')
    button2 = widgets.Button(description='Solution', disabled=True)
    box = HBox(children=[button,button2])
    out = widgets.Output()

    def callback(e):
        out.clear_output()
        button2.disabled = False
        with out:
            if(text.value == 2):
                print('Correct !')
            else: 
                print('Faux, essayez encore ou regardez la solution')
    
    def solution(e):
        out.clear_output()
        with out:
            A = np.array([[1, 2, 3], [0, 1, 2]])
            A_t = A.transpose()
            display(Markdown('Pour trouver le rang colonne de $A$, nous utilisons la remarque 1 et trouvous le rang ligne de la transposée de $A$.'))
            display(Markdown('Par les propositions 1 et 2, nous échelonnons la matrice transposée et observont le nombre de lignes qui contiennent des pivots'))
            M = al.echelonMat('E', A_t)
            display(Markdown('Ainsi le rang colonne de $A$ est 2'))
                      
    button.on_click(callback)
    button2.on_click(solution)
    
    display(text)
    display(box)
    display(out)
    


# Chapitre 4.10

def Ex1_1Chapitre4_10(A,b):
    A_sol = [[1, 4, 3, 4],[2, 6, 5, 8],[1, 0, 1, 4]]
    b_sol = [[1], [1], [1]]
    if A == [] or b == []:
        print("Attention, vous avez laissé au moins une des deux entrée vide")
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
            print("Faux, votre réponse correspond au système suivant:")
            al.printSyst(A, b)

def Ex1_2_ech_Chapitre_4_10():
    global m
    print('Échelonnez la matrice transposée de A')
    A_sol = np.array([[1, 4, 3, 4],[2, 6, 5, 8],[1, 0, 1, 4]])
    A_sol_t = A_sol.transpose()
    al.printA(A_sol_t)
    [i,j,r,alpha]= al.manualEch(A_sol_t)
    MatriceList=[np.array(A_sol_t)]
    m = A_sol_t


    button = widgets.Button(description='Appliquer')
    out = widgets.Output()

    def applique(e):
        global m
        out.clear_output()
        with out:
            m=al.echelonnage(i, j, r, alpha, A_sol_t, m, MatriceList)

    button.on_click(applique)
    display(button)
    display(out)

def Ex1_2_1Chapitre_4_10():
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
                print('Faux, essayez encore ou regardez la solution')
                     
    button.on_click(callback)
    display(box)
    display(button)
    display(out)

    
def Ex1_2_2Chapitre_4_10(base):
    base_sol = [[1,2,1],[0,1,2]]
    
    if base_sol == []:
        print("L'entrée est vide")
    else:
        if base == base_sol:
            print("Correct !")
        else:
            print("Faux")
    
    
def Ex1_3Chapitre_4_10():               
    radio = widgets.RadioButtons(
        options=['Oui', 'Non'],
        description='Réponse:',
        disabled=False
    )

    button = widgets.Button(description='Vérifier')
    button2 = widgets.Button(description='Solution', disabled=True)
    box = HBox(children=[button,button2])
    out = widgets.Output()

    def callback(e):
        out.clear_output()
        button2.disabled = False
        with out:
            if (radio.value == "Oui"):
                    print('Mauvaise réponse.')
            else: 
                print('Correct !')
                
    def solution(e):
        out.clear_output()
        with out:
            A = np.array([[1, 2, 1],[0, 1, 2],[0, 0, 1]])
            display(Markdown('Après échelonnage, la matrice devient'))
            al.printA(A)
            display(Markdown("Ainsi le rang colonne de la matrice augmentée est 3, et le système n'admet pas de solution"))
    
    button2.on_click(solution)
    button.on_click(callback)
    
    display(radio)
    display(box)
    display(out)

    
# Chapitre 4.11
def Ex1Chapitre4_11(base):
    base_solution = [[1,2],[3,2]]
    is_correct = all(item in base_solution for item in base)
    if is_correct:
        is_correct = all(item in base for item in base_solution)
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)


def Ex2a_B1_Chapitre4_11(vB1):
    v = [5,6]
     
    if vB1==v:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def Ex2a_B2_Chapitre4_11(vB2):
    v = [2,1]
     
    if vB2==v:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def Ex2b_B1_Chapitre4_11(uB1):
    u = [8,4]
     
    if uB1==u:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def Ex2b_B2_Chapitre4_11(uB2):
    u = [-1,3]
     
    if uB2==u:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def Ex2c_B1_Chapitre4_11(wB1):
    w = [5,4]
     
    if wB1==w:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def Ex2c_B2_Chapitre4_11(wB2):
    w = [0.5,1.5]
     
    if wB2==w:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    
def Ex3_B1_Chapitre4_11(sB1):
    s = [13,10]
     
    if sB1==s:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)

def Ex3_B2_Chapitre4_11(sB2):
    s = [1,4]
     
    if sB2==s:
        is_correct = 1
    else:
        is_correct = 0
    
    correct_text = "C'est correct!"
    incorrect_text = "C'est incorrect."
    
    display(correct_text if is_correct else incorrect_text)
    

# Chapitre 4.12
def Ex1Chapitre4_12():
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
    
def Ex2Chapitre4_12():
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
    
def Ex3aChapitre4_12():
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
    

def Ex3bChapitre4_12(base):
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