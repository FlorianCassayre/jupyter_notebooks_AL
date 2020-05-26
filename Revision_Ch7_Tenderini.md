# Revision of Chapter 7

## General Issues
- It is better if you stick to the naming convention adopted in chapters
1,2,3,4 and 8 for the naming of the notebooks (i.e notebook index + title
as it can be found on the MOOC)
- It would be nice if you can add a link to the next notebook (in all but
the last notebook), as it has been done in Chapters 1 and 2
- Theoretical concepts have been explained very well, while I think you 
you should add more exercices in the notebooks, to give students
more material to work on. The implemented exercices are fine, thus I
would just replicate them on different cases (i.e. different matrices), 
eventually apporting some minor modifications.
- Often it is not clear whether students should do the computations
on paper and then check the answer by executing the next cell or not.
I would thus make it more clear, just by saying something like:
"Compute [this and that] and run the next cell to verify your answer"
- In the functions to compute the determinant step-by-step (which are nice!) 
sometimes the signs are repeated (like '+-3' or '--1'); it would be
nice if you can avoid it
- It would be better to differentiate better the exercises/examples cells;
you could use the syntax adopted in Chapter 1,2,3 and 8, where each 
exercise is named as "EXERCICE n \n" to make it more visible

## Notebook-specific issues

### Notebook 7.1
- **Exercise 1**: make it clear that students have to compute the 
determinant on paper and run the cell to verify, otherwise they are
given the answer for free
- **Exercise 2**: Why the determinant is computed in the "classical"
way in the first case and not using Sarrus' rule? It would be better
to use Sarrus, since it is the topic of the exercise

### Notebook 7.2
- The first sentence is not so clear to me.. you should maybe rephrase
it (but consider that my french is bad, so it could be my fault)
- In the theoretical section regarding invertibility, I would write
the formula better, as: "A^{-1} exists --> det(A) is not 0"
- In the first **example**, it would be better to rename the matrices
with capital letters, as this is the standard naming convention

### Notebook 7.3
- In the **proposition** at the beginning, I would erase the last sentence
("Generalment ..."), since it seems quite reductive to me and not that
useful
- In the **exercise**, report the expression of the matrix to be
"processed" in the Markdown cell and clearly state that students
have to perform computations on their own and execute the cell only to 
verify their answer
- I would add more exercises on this topic

### Notebook 7.4
- In the theoretical section, I would explain what do you mean by
"matrices semblables", eventually with a "RAPPEL" before the statement
- I would add more exercices on this topic, considering 3/4 different
couples of matrices and asking students to compute the determinant
of several other matrices arising from suitable combinations of
the given ones
- Here I would differentiate the function to check whether the answer
is correct or not and the one giving the solution. In this way the
student may perform new computations in case his/her answer is wring,
without being given the solution for free. Another good solution
is to add a button named "Solution", which students can select in case 
they want the solution to be displayed (there is an example of how to
do this in the last exercise of Notebook 4.9)

### Notebook 7.5
- Very nice plots! But also here, despite being an explanatory notebook, 
I would add a small exercise asking students to draw the parallelogram 
(or parallelepiped) corresponding to a given matrix (simple!), to
compute its volume and to compare it with the matrix determinant

### Notebook 7.7
- Also here I would just add more exercises on different matrices
(maybe a 2x2, a 3x3 and a 4x4); eventually you can ask students to compute
the inverse both using cofactors and using Gauss elimination method
(implemented in interactive way, see Chapter 2) and compare the results


Despite these modifications, I think notebooks are well made and well
designed; the theoretical part is definitely complete, while exercises
appear to me a little bit poor (not in their content, but in their 
number; I would make at least 2 exercices per notebook). If you 
manage to add some exercices and to fix the other minor stuff I
noticed, then I think you work is definitely fine!

Riccardo
