# Revision of Chapter 4 (until 4.10)

## General issues

- All "rubbish" code, containing the implementation of the widgets, 
the verification of the answers or the generation of the plots should
be moved to a different `.py` file (as done in Chapters 1, 2, 3 and 7). 
Notebooks should only contain a simple call to the verification (or plot 
generator) function, with input arguments that must not refer in any way to the
correct solution
- It would be nice to add a link to the subsequent notebook at the end of each
one (but the last one obviously), as done in Chapters 1 and 2

## Notebook-specific issues

### Notebook 4.1-4.2
None

### Notebook 4.3
- In **exercise 1**, add example with a set that actually defines a basis
- I think it is better to move all theoretical concepts at the beginning,
rather than distributing them through the file

### Notebook 4.4-4.5
- In **exercise 1**, I had problems in seleting the alternatives (it could be my
laptop, but maybe make a brief check)

### Notebook 4.6
- **Proposition 3** is not so clear; I understood it but it took me a while, 
thus I think students may find it a little bit obsure. The perfect way
would be to add a brief guided example, to practically show what you meant.
- **Exercise 1**: the explanation of the correct answer is a bit short, you can 
maybe expand it and make it more clear
- **Exercise 2**: the explanation of the correct answer is wrong (the number of free
variables is 2, not 3; 3 is the number of independent variables)

### Notebook 4.7-4.8
- **Exercise 1**: since the space of solutions depends on the system, I would
rephrase the question asking "Which could be the dimensions of...?"; additionally
it seems to me that the answer is wrong. If a system has 3 equations and 5 unknowns,
then the dimension of the space of the solutions ranges from 5 (if all equations
are null, since in such case any vector of R^5 would be a solution) to 2 (if
all te equations are independent, since in such case we impose 3 constraints of 
5 variables).

### Notebook 4.9
- Be careful with numbering in the theoretical section. If a remark is the first
one, then it should be numbered as Remark 1 (even if a theorem is defined before); 
same for the tho propositions and even for the Lemma in 4.10.
- **Exercise 1**: I'd rather use more separate widgets than a unique one with
the select-multiple option; I think it is way better looking.
- I think you should add other exrecises here, on the same style of the proposed one

### Notebook 4.10
- **Exercise 1.1**: why are you asking to insert vector b as a row vector... we 
have been using a different notation in the rest of the notebooks!
- **Exercise 1.2**: you can let the students perform the echelonment, instead of 
giving it done to them. You can just look at how the interactive Gauss Elimination
is used in notebooks of chapter 2... it is pure copy-pasting
- **Exercise 1.3**: you can add an interactive echeloment of the augmented matrix,
to let the students practically unsterstand why the systems cannot admit any 
solution


Despite these small observations, I liked your work guys! I did not have any problem 
executing the notebooks and I liked the vast majority of the exercises you have
implemented. I think that moving rubbish code away and fixing some minor errors
would make them look definitely fine.

Riccardo