
# Numpy 

## Numpy - Intro
```xml
It stands for Numberical Python


It is used for large count of numerical calculations.
It is fase (since optimizations are written in C)
Numpy operations are much faster when it comes to Lists than traditional Python

To use numpy import it first in your code: 
import numpy as np


Numpy documentation:
https://numpy.org/doc/
```
## Datatypes and attributes
```xml

Everything in numpy is represented as ndarray (n-dimensional array)



3-dimensional array in Numpy: 
a3 = np.array([[[1, 2, 3.1],
                [4, 5, 6]],
               [[7, 8, 9],
                [10, 11, 12.7]],
               [[13, 14, 15],
                [16, 17, 18]]])
Shape of this array is (2,3,3)
Axis 0 = rows
Axis 1 = columns
Axis n = no. of repeatations of rows and columns

a3.shape -> this will return 3,2,3
a3.ndim -> this will return no. of dimensions = 3
a3.dtype -> This will return the type of array = dtype('float64')
a3.size -> No of elements in the array = 18 
type(a3) -> This is return the type as numpy.ndarray


import pandas as pd
df = pd.DataFrame(a3) - > This will throw error as 
only 2 dimensional arrays can be passed to DataFrame

```

## Creating Numpy arrays
```xml
Important: Press Shift+Tab inside () of a function ->
This will give the full details/help of any function 


np.ones(shape, dtype=None, order='C', *, like=None)
-> Return a new array of given shape and type, 
filled with ones.

np.ones((2,3))- > Return a new array of shape 2,3 filled 
with 1. 

np.zeros((2,3)) -> Same as above but filled with zeros


np.arange([start,] stop[, step,], dtype=None, *, like=None)
-> Return evenly spaced values within a given interval.

np.arange(0,10,2) -> will return array([0, 2, 4, 6, 8])

np.randon.randint(low, high=None, size=None, dtype=int)
Return random integers from `low` (inclusive) to `high` (exclusive).

np.random.randint(1,8, size=(3,5)) -> Will return random
array of integer between 1 to 8 filling an array of shape 3,5

np.random.random() -> 
Return random floats in the half-open interval [0.0, 1.0]

np.random.random((5,2)) -> This will return a random 
array of shape (5,2) with values between 0 to 1 in float

random_array3=np.random.rand(5,2) -> This does the same 
thing as above

Note: When Numpy sets random numbers they are 
Psudo random numbers 



```


### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/
https://numpy.org/doc/

```