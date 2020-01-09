---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
David Barnett


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


Can this code be executed in the command line?
Can this code be directly converted to raw python code without JSON formatting?
How well does Jupyter's formatting convert to other notebook/markdown formats? (Like R markdown)


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.

```python
import numpy as np
import pandas as pd
```

## Exercise 1

```python
# YOUR SOLUTION HERE
a = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
print(a) 
```

## Exercise 2

```python
# YOUR SOLUTION HERE
b = np.array([[3,1,1,1],[1,3,1,1],[1,1,3,1],[1,1,1,3],[1,1,1,1],[1,1,1,1]])
print(b)
```

## Exercise 3

```python
# YOUR SOLUTION HERE
print(a*b)
print("a*b works because each number in a is multiplied by the number its corresponding position in b")
print("The dot product will not work, however, because a 6X4 matrix cannot be multiplied by another 6X4 matrix")
```

## Exercise 4

```python
# YOUR SOLUTION HERE
print(np.dot(a.transpose(),b))
print()
print(np.dot(a,b.transpose()))

print("The matrices are different sizes because both have a different number of rows than columns, meaning the first matrix must be 4X4 and the second matrix must be 6X6")
```

## Exercise 5

```python
# YOUR SOLUTION HERE
def greeting(x):
    if x == True:
        print("Hello")
    else:
        print("Goodbye")
        
greeting(True)
greeting(False)
```

## Exercise 6

```python
# YOUR SOLUTION HERE
def array_math():
    arr1 = [1,2,3]
    arr2 = [4,5,6]
    arr3 = [7,8,9]
    print("Mean 1:", np.mean(arr1))
    print("Mean 2:", np.mean(arr2))
    print("Mean 3:", np.mean(arr3))
    print("Sum:", np.sum(arr1) + sum(arr2) + sum(arr3))
    
array_math()
```

## Exercise 7

```python
# YOUR SOLUTION HERE
def ones(array):
    sum = 0
    for arr1 in array:
        for arr2 in arr1:
            if arr2 == 1:
                sum += 1
    print(sum)
    
arr = np.array([[1, 5, 8, 4], [0, 1, 2, 1], [9, 9, 5, 3]])
ones(arr)
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
# YOUR SOLUTION HERE
a = pd.DataFrame([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
print(a)
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
# YOUR SOLUTION HERE
b = pd.DataFrame([[3,1,1,1],[1,3,1,1],[1,1,3,1],[1,1,1,3],[1,1,1,1],[1,1,1,1]])
print(b)
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
# YOUR SOLUTION HERE
print(a.mul(b))
print("Still no dot product because dot product multiplies by the outer dimensions, which are not the same number")
```

## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
# YOUR SOLUTION HERE
def ones(dataframe):
    sum = 0
    for i, row in dataframe.iterrows():
        for arr2 in row:
            if arr2 == 1:
                sum += 1
    print(sum)
    
df = pd.DataFrame([[1, 5, 8, 4], [0, 1, 2, 1], [9, 9, 5, 3]])
ones(df)
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.head()
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
## YOUR SOLUTION HERE
titanic_df["name"]
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
## YOUR SOLUTION HERE
titanic_df.set_index('sex',inplace=True)

female_df = titanic_df.loc[["female"]]

print(female_df.head())
female_df.shape[0]
```

## Exercise 14
How do you reset the index?

```python
## YOUR SOLUTION HERE
titanic_df.reset_index()
```

```python

```
