
# Python - Basics 

## Install Python
```xml


For python installation download the installer from the below location and install. 
https://www.python.org/downloads/


After installation open the command like and check the installation with the following command:
python3 --version

```

## Basics

### Basic Math Operations
```xml
Printing in Python 
print("Hello World")
This can be run on the IDE or saved as a file and run 


Operations
print(5*5) -> This is for multiplication 
print(7-5) -> This is for subtraction
print(7+5) -> This is for addition 
print(6/2) -> This is for divison
print(10**2) -> This is for power
print(10%2) -> This is for reminder

print(10+5*2) -> Complex activities, but presedence is for multiplcation 
Operation precedence can be found in the link below:
https://www.programiz.com/python-programming/precedence-associativity


# will be used for comment in python

```
 
### Basic BuiltIn Functions
```xml

print(abs(-10.05))  # prints 10.5 -> Absolute 
print(pow(2, 5))  # returns: 32 -> Power 
print(max(34, 45, 67))  # returns: 67 -> Maximum
print(min(34, 45, 67))  # returns: 34 -> Minimum 


```

### Variables
```xml
Variables are case sensitive, must start with letter or _ and second character onwards could be numbers 

a=5
print(a)
a=7
print(a)

name="Balaji"
age=35
sal=45.5

print(name)
print(age)
print(sal)

sal="Reassgined"
print(sal)

x=5
x=x+1
print(x)


x //=5 # This is compound operations which is short form for x = x//5 (integer division)
print(x)

age **=2 # This is compound operations which is short form for age = age**2 (power)
print(age)

print(x, age, sal) # print all values in a single print statement


x=6
y=3
result=(x*y)+(y/x) # (6*3) + (3/6) = 18+0.5 which will result in 18.5
print(result)

print(f"This is f string {x} example") 

```


### For loops
```xml
For loop is used to iterate a block of statements (indention is used for starting and ending of block)

for i in range(1,10) : 
	print(i) 
	print("hello")

print("We are outside the block now")


# Multiplication tabbles in python
for i in range(1,11) :
	print(f"7 * {i} = {i*7}")

# Range function with increment step of 2
sum=0
for i in range(1,11,2) :
	sum=sum+i # will sum all odd numbers
print(sum)

# Print the sum of squares of the first 10 numbers
sum_of_squares=0
for i in range(1,11) :
    sum_of_squares=sum_of_squares+(i**2)
print(sum_of_squares)


# Print the sum of squares of the first 10 numbers
sum_of_cubes=0
for i in range(1,11) :
    sum_of_cubes=sum_of_cubes+(i**3)
print(sum_of_cubes)

# Print the factorial of 6
factorial=1
for i in range(1,7) :
    factorial=factorial*i
print(factorial)


# Different for loops for practise
for i in range(5):
    print(i)
 
for i in range(2, 11):
    print(i)
 
for i in range(2, 11, 2):
    print(i)
 
for i in range(3, 13, 3):
    print(i)
 
for i in range(3, -4, -1):
    print(i)
 
total = 0
for i in range(1, 101):
    total += i
print(total)

```


### Nested For Loops
```xml
Loop within another loop 

# In nested for loop the outer loop runs for the as many iterations of the inner loop
for i in range(1,3):
    for j in range(1,3):
        print(f"i = {i}, j = {j}")


# Printing a pattern in Python
for i in range(5): # Range will run from 0 to 5 
	for j in range(5):
		print("*", end="") # will not print newline at the end, but will print an empty character  
	print() # will print empty line


# Printing another pattern in Python
for i in range(5): # Range will run from 0 to 5 
	for j in range(i+1):
		print("*", end="") # will not print newline at the end, but will print an empty character  
	print() # will print empty line


```

### Functions in Python
```xml
Funtions are block of code that can be reused 
It has input and output parameters along with an invoke call. 

Tips: 
Always use descriptive names for your functions to make your code more readable.
Keep your functions small and focused on a single task.
Test your functions with different inputs to ensure they work correctly in all situations.


Check all function samples in the example 06-function.py 


```

### Data Types
```xml
int - Eg. 1, 9, -5, 1056
float - Eg. 3.5, 4.0, -9.2 

type(value) -> This will return the data type of the value for eg. type(5) will return integer - <class 'int'>

In python a divison operator always returns a float, even if they are divisible 

The result of operations between int and float is always a float



BASIC ARITHMETIC
----------------
Python supports basic arithmetic operations. Let's create a simple variable i and increment it by 1.

i = 1
i = i + 1
print(i)  # Output: 2


+= OPERATOR

The += operator is a shorthand to increment a variable.

i += 1
print(i)  # Output: 3


OTHER COMPOUND ASSIGNMENTS
--------------------------
Apart from +=, Python also supports other compound assignment operators such as -=, /=, *=.

i += 1
print(i)  # Output: 4
 
i -= 1
print(i)  # Output: 3
 
i /= 1
print(i)  # Output: 3.0
 
i *= 2
print(i)  # Output: 6.0


DYNAMIC TYPING IN PYTHON
------------------------
In Python, the type of a variable can change during the execution of a program. This is called dynamic typing.

i = 2
print(type(i))  # Output: <class 'int'>
 
i = i / 2.0
print(type(i))  # Output: <class 'float'>


PERFORMING INTEGER DIVISION
---------------------------

The double slash operator (//) performs integer (or floor) division.

number1 = 5
number2 = 2
print(number1 // number2)  # Output: 2


Compound assignment works with // too:

number1 //= 2
print(number1)  # Output: 2


EXPONENTIATION
--------------

The double asterisk operator (**) or the pow() function performs exponentiation.

print(5 ** 3)  # Output: 125
print(pow(5,3))  # Output: 125


TYPE CONVERSION
---------------
Python provides several functions to convert between different data types. These include int(), float(), and round().

int() Function

print(int(5.6))  # Output: 5


Rounding Numbers

You can round a number to the nearest integer using the round() function. This function can also round a number to a specified number of decimals.

print(round(5.6))  # Output: 6
print(round(5.4))  # Output: 5
print(round(5.5))  # Output: 6
print(round(5.67, 1))  # Output: 5.7
print(round(5.678, 2))  # Output: 5.68

Converting int to float
You can also convert an int to float using the float() function.

print(float(5))  # Output: 5.0


i=10
print(type(i==0)) # output: type Boolean

Always remember to capitalize the first letter when using True and False.

Use == for comparison and = for assignment.

Be mindful of the difference between equality (==) and assignment (=) to avoid logical errors in your code.

Int value of True and False:
print(int(True)) # will return 1 
print(int(False)) # will return 0

SUMMARY
-------
Integer: An integer is a whole number, without a fraction. Examples include 1, 2, 6, -1, and -2. int
Floating Point Numbers: Floating-point numbers, or floats, represent real numbers and are written with a decimal point dividing the integer and fractional parts. Examples include 2.5 and 2.55.
Boolean Values: In Python, True and False are the Boolean values.
+= OPERATOR: Shorthand to increment a variable. i += 1. Similar operators -=, /=, *=
Integer division The double slash operator (//): performs integer (or floor) division. 5//2 results in 2.
Dynamic Typing In Python: The type of a variable can change during the execution of a program
Remember Operations can also be performed between int and float. The result of an operation between an int and a float is always a float.
Use == for comparison and = for assignment.

```


### Conditions
```xml


```



### Reference
```xml
https://www.udemy.com/course/python-programming-for-beginners-with-exercises/
https://www.python.org/downloads/

```