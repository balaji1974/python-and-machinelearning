
# Numpy - Basics

## Numpy - Intro
```xml
It stands for Numeric Python


It is used for large count of numerical calculations.
It is fase (since optimizations are written in C)
Numpy operations are much faster when it comes to Lists than traditional Python

To use numpy import it first in your code: 
import numpy as np


Numpy documentation:
https://numpy.org/doc/
```
## Basics of numpy
```xml
Datatypes and attributes
------------------------

Everything in numpy is represented as ndarray (n-dimensional array)

1-dimensional array: 
a1 = np.array([1, 2, 3])

2-dimensional array: 
a2 = np.array([[1, 2.0, 3.3],
              [4,5,6.5]])

3-dimensional array: 
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

4 dimensional array 
a4 = np.random.randint(10, size=(2,3,4,5))
a4
This will be a 4 rows, 5 columns array repeated 3 times which will be repeated 2 times
Result will be something like this: 
array([[[[2, 7, 1, 5, 8],
         [6, 9, 0, 7, 1],
         [0, 2, 9, 1, 8],
         [5, 9, 1, 4, 0]],

        [[2, 6, 0, 3, 2],
         [5, 6, 1, 1, 5],
         [4, 6, 5, 6, 0],
         [1, 5, 3, 3, 2]],

        [[0, 6, 9, 6, 9],
         [1, 2, 1, 2, 8],
         [4, 7, 8, 9, 3],
         [8, 2, 9, 8, 0]]],


       [[[0, 3, 9, 4, 6],
         [9, 8, 5, 0, 1],
         [9, 4, 2, 0, 3],
         [5, 3, 2, 3, 4]],

        [[5, 4, 1, 6, 1],
         [8, 0, 9, 0, 6],
         [4, 3, 4, 5, 0],
         [1, 2, 0, 9, 8]],

        [[9, 6, 3, 8, 4],
         [7, 5, 6, 6, 6],
         [5, 3, 0, 6, 7],
         [6, 3, 9, 7, 8]]]])

import pandas as pd
df = pd.DataFrame(a3) - > This will throw error as 
only 2 dimensional arrays can be passed to DataFrame

Creating Numpy arrays
---------------------
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
Psudo random numbers (deterministic random numbers)

Random Seed
-----------
np.random.seed(seed=<any number>) -> This allows to create random numbers, 
but these numbers are reproducable when rerun 

# Eg. This will create the same set of random numbers 
# every time you run it, if set within the same cell 
np.random.seed(seed=5)
np.random.randint(10, size=(5,3))


Viewing Arrays and Matrices
---------------------------
np.unique(<numpy array>) -> This will find the unique elements 
of a numpy array

Viewing elements of the array:
a1[0] -> will return 1
a2[0] -> will return array([1. , 2. , 3.3])
a3[0] -> will return array([[1. , 2. , 3.1],
                            [4. , 5. , 6. ]])

a3[:2, :2, :2] -> Display by slicing 
array([[[ 1.,  2.],
        [ 4.,  5.]],

       [[ 7.,  8.],
        [10., 11.]]])

a4[:, :, :, :4] -> This will give the first 4 elements (columns) of the array 
a4[:, :, :2, :4] -> This will give the 4 column and 2 row elements of the array 

Manipulate array
----------------

# Arithmetic: 

ones = np.ones(3) 

a1 + ones -> will return the sum of elements of both arrays
np.add (a1, ones) -> Same as above 

a1 - ones -> will return the difference of elements of both arrays 

a1 * ones -> Will return the product of elements of both arrays

a1 / ones -> will return the division of elements of both arrays 

# Different shapes -> Broadcasting 
# This feature is know as numpy broadcasting: 
# where smaller array is broadcast across larger array so 
# that they have compatible shapes. 
a1 * a2 -> will multiple every row elements of a1 by a2 

https://numpy.org/doc/stable/user/basics.broadcasting.html

a2 / a1 -> division 

a2 // a1 -> floor division

a2 ** 2 -> each elements by Power of 2 
np.square(a2) -> same as above

a1 % 2 -> will return reminder of each element after divison by zero 

np.exp(a1) -> will return exponential of each element 

np.log(a1) -> will return log of each elements 

# Aggregation: 
# Create a python list 
listy_list = [1,2,3]

np.sum(listy_list) -> sum of all elements in the array 

# Create a large array
massive_array = np.random.random(100000)
# Display the 1st 100 items of this array
massive_array[:100]

### Important -> Always use numpy aggregation functions 
### over python aggregation functions as it is optimized 
### for numerical calculations and much faster than python functions 

# Comparing python opertors with numpy operators
# %timeit -> Python's magic function, 
# measure's execution time of small code snippets
%timeit sum(massive_array) # python's sum
%timeit np.sum(massive_array) # numpy's sum

np.mean(a2) -> Mean 
np.max(a2) -> Maximum 
np.min(a2) -> Minimum

Standard Deviation and variance 
-------------------------------

# Std Dev -> How spread out a group of number is from the mean 
np.std(a2) -> Standard deviation 

# Variance -> The measure of average degree to which each number is different to the mean 
# Higher variance -> Wider range of numbers
# Lower variance -> Lower range of numbers
np.var(a2)

# std dev = sqrt(var)
np.std(a2) == np.sqrt(np.var(a2))

high_var_array=np.array([1, 100, 200, 300, 4000, 5000])
low_var_array=np.array([2,4,6,8,10])
np.var(high_var_array) # 4296133.472222221
np.var(low_var_array) # 8.0

np.std(high_var_array) # 2072.711623024829
np.std(low_var_array) # 2.8284271247461903

np.mean(high_var_array) # 1600.1666666666667
np.mean(low_var_array) # 6.0

import matplotlib.pyplot as plt
plt.hist(high_var_array)
plt.show()

plt.hist(low_var_array)
plt.show()


Reshape and Transpose
---------------------
a2.reshape (2,3,1) -> Will reshape the array without losing information

a2.T -> Transpose will switch X-axis to Y and Y-axis to X


Dot Product
-----------
np.random.seed(0)
mat1 = np.random.randint(10, size = (5,3))
mat2 = np.random.randint(10, size = (5,3))


mat1 *  mat2 # element-wise multiplication - hadamard product

mat1.dot(mat2) -> This will return error as the shapes are not aligned for dot product multiplication 
# The column of the first matrix must match with the rows of the second matrix - Rule

mat1.T.dot(mat2) -> This will work with a result of 3*3 matrix or 
mat2.T.dot(mat1) -> This will also work with a result of 3*3 matrix - same result as above

mat1.dot(mat2.T) -> This will result in a 5*5 matrix 


Sample
------

# Create a random array of 5*3 matrix with numbers less than 20
np.random.seed(0) 
sales_amount = np.random.randint(20, size=(5,3))
sales_amount 

# create weekly sales dataframe
weekly_sales = pd.DataFrame(sales_amount, 
                            index=["Mon", "Tues", "Wed", "Thurs", "Fri"],
                            columns=["Almond butter", "Peanut butter", "Cashew butter"])
weekly_sales  

# Create prices array 
prices = np.array([10, 8, 12])
prices

# Create butter price dataframe 
butter_prices = pd.DataFrame(prices.reshape(1,3), 
                        index=["Price"],
                        columns=["Almond butter", "Peanut butter", "Cashew butter"])
butter_prices

# Total sales 
total_sales = prices.dot(weekly_sales.T)
total_sales

# Daily sales 
daily_sales = butter_prices.dot(weekly_sales.T)
daily_sales

# Create a daily sales column and update its values
weekly_sales["Daily Sales $"]=daily_sales.T
weekly_sales

Comparision Operators
---------------------
a1 > a2 -> every element of a1 greater than corresponding element in a2 
# result of this would be 
array([[False, False,  True],
       [False, False, False]])

a1 >= a2 -> every element of a1 greater than or equal to corresponding element in a2 
# result of this would be 
array([[ True,  True,  True],
       [False, False, False]])


a1 == a2 -> To check if 2 array elements are equal to or not 

Sorting Arrays
--------------
random_array.sort() -> sort the array in column axis 
np.argsort(random_array) -> return the index of the columns in sorting order 
np.argmin(a1) -> return the min index of the array
np.argmax(a1) -> return the max index of the array


Numpy - Example
---------------
# Image as a markdown 
<img src="images/panda.png"> # make the cell markdown (esc + M and then shift enter)

# Turn the image into a numpy array
from matplotlib.image import imread
panda = imread("images/panda.png")
print(type(panda))

# Check the array and its attributes
panda
panda.size, panda.shape, panda.ndim

# The image is stored pixel by pixel using its red/yellow/blue color value 
panda[:5]

# read other images 
car=imread("images/car-photo.png")
print(type(car))
car[:5]

dog=imread("images/dog-photo.png")
dog[:5]

```

## Numpy - Key Concepts
```xml

Array vs List
-------------
list_a = [[1,2,3],[4,5,6]] # List 
array_a = np.array(list_a) # ndarray 


Broadcasting
------------
If we want to perform elementwise operations, 
but have elements of different size and/or dimension, 
we can broadcast the smaller variable and 
create a broadcased version with the size of the larger one. 
It is like "stretching" on variable over the other 
to produce output with the same shape 


Type casting
------------
Taking every element of an array and changing it 
to a specific datatypes 


DataTypes
---------
Complete list of numpy datatypes 
https://numpy.org/doc/2.1/user/basics.types.html

Running a function on a given axis
----------------------------------
Break the ND array into smaller arrays of (n-1) dimensions 
and apply the function to each one of these breakdowns either 
row wise or column wise. 
axis = 0 -> apply the function on the column of the array  
axis = 1 -> apply the function on the row of the array  

Slicing
-------
Creating a new array by taking chunks of values 
from existing array
Eg. 
a[1:1:1] -> means one dimensional with 1st row, upto 1st row and step 1
a[1:1:1, 1:1:1] -> means two dimensional with 1st row, upto 1st row and step 1 
and 1st column,upto 1st column and step 1

Defining random numbers
-----------------------
from numpy.random import Generator as gen
from numpy.random import PCG64 as pcg

array_RG = gen(pcg())
array_RG.normal(size = (5,5))

array_RG = gen(pcg(seed = 365)) 
array_RG.normal(size = (5,5))

```

## Numpy - Random numbers and Statistical Distributions
```xml
# Generating random numbers between 0 to 1
array_RG = gen(pcg(seed = 365)) 
array_RG.random(size = (5,5))

# Generating integers between 10 to 100 
array_RG = gen(pcg(seed = 365)) 
array_RG.integers(low = 10, high = 100, size = (5,5)) 

# Chooses among a given set (with possible weighted probabilities).
array_RG = gen(pcg(seed = 365)) 
array_RG.choice((1,2,3,4,5), p = [0.1,0.1,0.1,0.1,0.6],size = (5,5))

# The default Poisson distribution.
array_RG = gen(pcg(seed = 365)) 
array_RG.poisson(size = (5,5))

# Poisson distribution while Specifying lambda. 
array_RG = gen(pcg(seed = 365)) 
array_RG.poisson(lam = 10,size = (5,5))

# A binomial distribution with p = 0.4 and 100 trials. 
array_RG = gen(pcg(seed = 365)) 
array_RG.binomial(n = 100, p = 0.4, size = (5,5))

# A logistic distribution with a location = 9 and scale = 1.2.
array_RG = gen(pcg(seed = 365)) 
array_RG.logistic(loc = 9, scale = 1.2, size = (5,5))

https://numpy.org/doc/stable/reference/random/generator.html

```

## Numpy - Applications of Random Generators
```xml

# Create the individual columns of the dataset we're creating. 
array_RG = gen(pcg(seed = 365)) 
array_column_1 = array_RG.normal(loc = 2, scale = 3, size = (1000))
array_column_2 = array_RG.normal(loc = 7, scale = 2, size = (1000))
array_column_3 = array_RG.logistic(loc = 11, scale = 3, size = (1000))
array_column_4  = array_RG.exponential(scale = 4, size = (1000))
array_column_5  = array_RG.geometric(p = 0.7, size = (1000))


# Use np.array to generate a new array with the 5 arrays we created earlier. 
# Use the transpose method to make sure our dataset isn't flipped. 
random_test_data = np.array([array_column_1, array_column_2, array_column_3, array_column_4, array_column_5]).transpose()
random_test_data
random_test_data.shape


# Saving the arrays to an extrenal file we're creating. 
# file name -> "Random-Test-from-NumPy.csv"
# random_test_data -> data we're exporting (saving to an external file)
# format -> strings
# delimiter ","
# We'll talk more about these in just a bit. 
np.savetxt("Random-Test-from-NumPy.csv", random_test_data, fmt = '%s', delimiter = ',')


# Importing the data from the file we just created. 
rand_test_data = np.genfromtxt("Random-Test-from-NumPy.csv", delimiter = ',')
print(rand_test_data)

```

## Important Resources
```xml
The Basics of NumPy Arrays by Jake VanderPlas
https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html

A Visual Introduction to NumPy by Jay Alammar
https://jalammar.github.io/visual-numpy/

NumPy Quickstart tutorial (part of the NumPy documentation)
https://numpy.org/doc/1.17/user/quickstart.html

```


### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/
https://numpy.org/doc/

https://www.udemy.com/course/preprocessing-data-with-numpy/learn/

```