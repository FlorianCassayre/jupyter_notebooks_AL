import sys
sys.path.insert(0, './../')

import plotly
plotly.offline.init_notebook_mode(connected=True)
import ipywidgets as widgets
from ipywidgets import interact_manual, Layout
from Librairie.AL_Fct import drawLine
from IPython.display import display, Latex, display_latex


###############   CHAPITRE 1_3_4    ###############


def Ex3Chapitre1_3_4():
    """Provides the correction to exercise 3 of notebook 1_3-4
    """

    print("Cliquer sur CTRL (ou CMD) pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    res = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)'],
        description='Systèmes avec le même ensemble de solutions:',
        style=style,
        layout=Layout(width='35%', height='170px'),
        disabled=False,
    )

    def correction(res):
        if 'a)' in res and 'c)' in res :
            print("C'est correct!")
            print('Pour le système a), on peut par exemple faire\n')
            sola= '$\\left(\\begin{array}{cc|c} 1 & 1 & 3\\\\ -1& 4 & -1 \\end{array}\\right) \\stackrel{E_{12}}{\sim}\\left(\\begin{array}{cc|c} -1& 4 & -1\\\\ 1 & 1 & 3 \\end{array}\\right)\\stackrel{E_{1}(-2)}{\sim}\\left(\\begin{array}{cc|c} 2& -8 & 2\\\\ 1 & 1 & 3 \\end{array}\\right)$'
            display(Latex(sola))

            print("Pour le système b), les systèmes ne sont pas équivalents. Comme solution on peut exprimer x1 en fonction de x2 et on obtient deux droites (parallèles) de pente 1 mais de hauteurs -2 et 2.$")
            print('Pour le système c), on peut par exemple faire\n')
            sola= '$\\left(\\begin{array}{ccc|c} \dfrac{1}{4} & -2 & 1& 5\\\\ 0& 1 & -1 & 0\\\\ 1 & 2 & -1 & 0 \\end{array}\\right) \\stackrel{E_{1}(4)}{\sim}\\left(\\begin{array}{ccc|c} 1 & -8 & 4& 20\\\\ 0& 1 & -1 & 0\\\\ 1 & 2 & -1 & 0\\end{array}\\right)\\stackrel{E_{31}(-1)}{\sim}\\left(\\begin{array}{ccc|c} 1& -8 & 4&20\\\\ 0 & 1 & -1&0\\\\ 0&10 &-5 & -20\\end{array}\\right)\\stackrel{E_{3}\\big({\\tiny\dfrac{1}{5}}\\big)}{\sim}\\left(\\begin{array}{ccc|c}1& -8 & 4&20\\\\ 0 & 1 & -1&0\\\\ 0&2&-1 & -4\\end{array}\\right)$'
            display(Latex(sola))
            
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, res=res)

    return



def Ex4Chapitre1_3_4():
    """Provides the correction to exercise 4 of notebook 1_3-4
    """

    print("Cliquer sur CTRL (ou CMD) pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    res = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)'],
        description='Systèmes avec le même ensemble de solutions:',
        style=style,
        layout=Layout(width='15%', height='170px'),
        disabled=False,
    )

    def correction(res):
        if 'a)' in res and 'c)' in res and 'd)' in res and 'h)' in res:
            print("C'est correct!")
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, res=res)

    return



###############   CHAPITRE 1_5_6    ###############

def Ex1Chapitre1_5_6(data):
    """Provides the correction to exercise 1 of notebook 1_5-6
    e=matrice qui sont échelonnée, er=échelonnée réduite et r=rien
    """
    e=data[0].value
    er=data[1].value
    r=data[2].value
    r=list(r.split(','))
    r=[elem.strip() for elem in r if elem.strip()]
    er=list(er.split(','))
    er=[elem.strip() for elem in er if elem.strip()]
    e=list(e.split(','))
    e=[elem.strip() for elem in e if elem.strip()]

    corr_e=['C','D','E','G','H','I','J']
    corr_er=['D','H','I','J']
    corr_r=['A','B','F']

    if set(corr_r)==set(r) and set(corr_er)==set(er) and set(corr_e)==set(e):
        print('Correct')
    else:
        if not set(corr_r)==set(r):
            print("Les matrices n'étant ni échelonnées, ni échelonnées-réduites sont fausses. ")
        if not set(corr_e)==set(e):
            print("Les matrices étant échelonnées sont fausses. ")
        if not set(corr_er)==set(er):
            print("Les matrices étant échelonnées-réduite sont fausses. ")
    return





###############   CHAPITRE 1_7   ###############



def Ex2Chapitre1_7():
    """Provides the correction to exercise 2 of notebook 1_7
    """

    print("Cliquer sur CTRL pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    inc = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)', 'd)'],
        description='Incompatibles:',
        style=style,
        layout=Layout(width='15%', height='90px'),
        disabled=False,
    )
    comp = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)', 'd)'],
        description='Compatibles:',
        layout=Layout(width='15%', height='90px'),
        disabled=False
    )

    def correction(inc, c):
        if 'a)' in c and 'c)' in c and 'd)' in c and 'b)' in inc:
            print("C'est correct!")
            print("En particulier, les systèmes a) et d) admettent une infinité de solutions, tandis que le système c) "
                  "admet une solution unique.")
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, inc=inc, c=comp)

    return




def Ex3Chapitre1_7():
    """Provides the correction of exercise 3 of notebook 1_7
    """

    systa = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Incompatible'],
        description='Système a):',
        disabled=False,
    )
    systb = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Incompatible'],
        description='Système b):',
        disabled=False
    )
    systc = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Espace', 'Incompatible'],
        description='Système c):',
        disabled=False
    )
    systd = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Espace', 'Incompatible'],
        description='Système d):',
        disabled=False
    )
    choice = widgets.Dropdown(
        options=['a)', 'b)', 'c)', 'd)'],
        value='a)',
        description='Système:',
        disabled=False,
    )

    def plot(c):
        if c == 'a)':
            drawLine([[0], [0]], [[4], [1]])
        if c == 'b)':
            print("Le système est incompatible, donc il n'y a pas de solutions")
        if c == 'c)':
            drawLine([[-17], [5], [-10]], [[0], [0], [0]])
        if c == 'd)':
            drawLine([[1], [0], [0]], [[0], [-1], [1]])

    def correction(a, b, c, d):
        if 'Droite' in a and 'Incompatible' in b and 'Point' in c and 'Droite' in d:
            print("C'est correct!")
            print("Sélectionnez le système souhaité et appuyez sur 'Run Interact'"
                  " pour visualiser son ensemble de solution(s), le cas échéant")
            interact_manual(plot, c=choice)
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, a=systa, b=systb, c=systc, d=systd)

    return
