import sys
sys.path.insert(0, './../')

import plotly
plotly.offline.init_notebook_mode(connected=True)
import ipywidgets as widgets
from ipywidgets import interact_manual, Layout
from Librairie.AL_Fct import drawLine


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
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, res=res)

    return


def Ex3Chapitre1_3_4():
    """Provides the correction to exercise 4 of notebook 1_3-4
    """

    print("Cliquer sur CTRL (ou CMD) pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    res = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)'],
        description='Systèmes avec le même ensemble de solutions:',
        style=style,
        layout=Layout(width='35%', height='170px'),
        disabled=False,
    )

    def correction(res):
        if 'a)' in res and 'c)' in res and 'd)' in res and 'h)' in res:
            print("C'est correct!")
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, res=res)

    return


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
