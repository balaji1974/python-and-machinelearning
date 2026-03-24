# Python - Advance 

## Start with Basics 
```xml
# System commands 
import sys
print(sys.version)
copyright

```

## Another IDE for Python - PyCharm community edition 
```xml
Download PyCharm community edition and install 

https://www.jetbrains.com/pycharm/download

Scroll all the way down for community edition 
Download and Install 

Create a project and create a python file called hello.py under it
Open the file and type:
print("Hello World !!")
Save and Run 

```

## Different ways of running python 
```xml
Python interpreter -> By directly opening python from command line using the command python 
Python shell -> Idle interpreter 
PyCharm IDE 
Jupyter Notebook 

```
## Strings 
```xml

# Escape character
print("Balaji's") # prints Balaji's
print("Balaji\"s") # prints Balaji"s 
print("Balaji\ns") # prints Balaji in line 1 & s in the next line
print("delhi\newyork") # prints delhi and ewyork in the next line
print(r"delhi\newyork") # prints delhi\newyork

"new" * 5 # Will display 'newnewnewnewnew'

# Multiline text 
print(''' Hi how are you ?
This is a multiline text 
I am continuing the text further''')
# This will return 
 Hi how are you ?
This is a multiline text 
I am continuing the text further


```

## Input & Read / Write
```xml
# Simple input in Python 
name = input("What is your name ? ")
print ("Hello ",name)


# Files
file = open ("bala.txt", "w") # 'w' means write to a file
file.write("Hi !!! This is balaji here.\n") # This is the content of the file
file.close() # close the file


file = open ("bala.txt", "a") # 'a' means write to a file
file.write("This is line no.2 that has been appended to the file") # This is appended content
file.close() # close the file

file = open ("bala.txt", "r") # 'r' means read from a file
content = file.read()
print(content)
file.close()

```

## Global and local variables
```xml
def my_function(global_variable):
    print(global_variable)
    local_variable="Local Hello"
    print(local_variable)

global_variable="Global Hello"
my_function(global_variable)

```

## Python Graphics 
```xml
From the Pycharm IDE import PythonTurtle module 
PyCharm -> Settings -> Python Interpreter -> Add (+) -> PythonTurtle -> Install Package -> Close

Run the following program: (save it first as turtledemo.py)
from turtle import *
forward(90)
right(120)
forward(90)
done()


Drawing a square: (square.py)
from turtle import *
forward(90)
right(120)
forward(90)
done()


```


## Python `self` attribute
```xml
# Self represents the instance of the class. 
# By using the “self”  we can access the attributes 
# and methods of the class in Python

class Mynumber:
    def __init__(self, value):
        self.value = value
    
    def print_value(self):
        print(self.value)

obj1 = Mynumber(17)
obj1.print_value()

```

## Python import - different ways
```xml

1. Standard Import 
Imports the entire module using its original name 
Example: import random

2. Import Specific Items
Imports specific attributes (functions, classes, variables) from a module.
Example: from random import choice, randint

3. Import with an Alias
Imports a module and renames it, usually with a shorter name.
Example: import pandas as pd

4. Absolute Imports
Specifies the full path to the module, starting from the project's root.
Example: import mypackage.mymodule

5. Import Specific Items with Alias
Imports specific attributes from a module and renames them.
Example: from os.path import join as join_path

6. Import All Items
Imports all attributes from a module.
Example: from math import *
Note: Using * is generally discouraged due to potential namespace conflicts.

7. Relative Imports 
Specifies the location of the module relative to the current script.
Uses dots (.) to indicate the level of the directory.
. (single dot): Refers to the current directory.
Example: from . import mymodule 
.. (two dots): Refers to the parent directory.
Example: from .. import mypackage

8. Dynamic Imports Imports modules at runtime using importlib.
Example: 
import importlib
my_module = importlib.import_module('my_module')

9. Implicit and Explicit Relative Imports
Implicit relative imports have been deprecated in Python 3.
Explicit relative imports use dots to specify the relative path.

10. Namespace Packages 
Allows subpackages to be distributed independently while still importable under a shared namespace.
Example: google.cloud.logging

11. Importing Inside Functions Modules can be imported within a function's scope 
Example:
def my_function():
    import math
    print(math.sqrt(16))
    
```


## Python module
```xml
# A module is nothing but a python file 
# that could be imported into another python
# program and called locally

# calmodule.py
def summation(a,b):
    return a+b

def subtraction(a,b):
    return a-b

# runmodule.py
from calmodule import summation, subtraction

print(summation(1,7))
print(subtraction(7,3))

# Another way of importing modules
import calmodule
print(calmodule.summation(1,7))
print(calmodule.subtraction(7,3))

```

## Python packages
```xml

# Python packages are a way to organize and structure code 
# by grouping related modules into directories. 
# A package is essentially a folder that contains an __init__.py file 
# and one or more Python files (modules). 
# This organization helps manage and reuse code effectively, 
# especially in larger projects. 
# It also allows functionality to be easily shared and 
# distributed across different applications.

# 1. Create a package 
# A package is a folder in the current working directory of python 
# create a package called vertibrates

# 2. Create modules inside the package
# Create 2 files called bird and fish inside the folder 
# bird.py
class bird:
    def __init__(self):
        self.members = ["pigeon", "crow", "sparrow"]

    def printBird(self):
        for member in self.members:
            print(member)

# fish.py
class fish:
    def __init__(self):
        self.members = ["shark", "squid", "salmon"]

    def printFish(self):
        for member in self.members:
            print(member)

# 3. create an init file to initize the 2 python files 
# as modules inside the package
# __init__.py
from .fish import fish
from .bird import bird

# 4. From the parent folder outside of vertibrates 
# create a calling file that calls this package and its 
# assocaiated modules
from vertibrates import fish
from vertibrates import bird

fishprint = fish()
fishprint.printFish()
birdprint = bird()
birdprint.printBird()


# This will print the following result:
shark
squid
salmon
pigeon
crow
sparrow

```

## Python - GUI
```xml
# Default bundled GUI framework of Python
Tkinter - check feet2meter.py

# Top GUI frameworks of Python
Kivy
PyQt
PyGUI




```

## Python - Variable arguments
```xml
# *args example
def fun(*args):
    return sum(args)

print(fun(1, 2, 3, 4)) 
print(fun(5, 10, 15))
Output =>
10
30

# **kwargs example
def fun(**kwargs):
    for k, val in kwargs.items():
        print(k, val)

fun(a=1, b=2, c=3)
Output => 
a 1
b 2
c 3
```

## Install python package from juypter notebook 
```xml
import sys
!conda install --yes --prefix {sys.prefix} seaborn # This will install the seaborn package 

```


## File system commands (file.py)
```xml
import os 

All Operating system commands are in this library. 
Please refer to file.py which has sufficient comments to explain each command

```

## TempFile (temp.py)
```xml
import tempfile 

check temp.py which contains code with the necessary comments

```

## Connect to internet (http1.py)
```xml
check http1.py which contains code with the necessary comments

```

## Networking
```xml
Check myserver.py 

```

## DocString
```
A docstring (documentation string) is a string literal that occurs as the first statement in a 
Python module, function, class, or method definition. It serves as a built-in description of the object's 
purpose and behavior, making code easier to understand, maintain, and automate documentation generation. 
```
#### Key Characteristics
```
Placement: Immediately after the definition line of a module, function, class, or method.
Delimiters: Enclosed in triple double quotes """ (or triple single quotes ''', though double quotes are standard).
Accessibility: Unlike comments, docstrings are retained at runtime and can be accessed using the 
object's __doc__ attribute (e.g., function_name.__doc__) or the built-in help() function.
Purpose: To document the public API for users and tools, explaining what the code does, its inputs, and outputs, 
rather than how it works internally (which is better suited for comments). 
```
#### Example
```
def determine_magic_level(magic_number):
    """
    Multiply a wizard's favorite number by 3 to reveal their magic level.
    """
    return magic_number * 3
```

## *args vs *kwargs
```
In Python, *args and **kwargs are used to allow a function to accept a variable number of arguments.
The primary distinction is how they handle the arguments: *args is for positional arguments
and **kwargs is for keyword arguments

*args collects any number of extra positional arguments into a tuple, which can then be iterated over
inside the function.

Example:
def print_args(arg1, *args):
    print(f"First explicit argument: {arg1}")
    for arg in args:
        print(f"Another arg from *args: {arg}")

print_args("hello", "world", "python", "is", "awesome")

**kwargs collects any number of keyword arguments (passed as key=value) into a dictionary.

Example:
def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_kwargs(name="Alice", age=30, city="New York")
```

## global vs nonlocal
```
The global and nonlocal keywords in Python are used to modify variables outside the current local scope,
but they target different scopes in Python's LEGB (Local, Enclosing, Global, Built-in) scope model. 

The global keyword is used to access and modify variables in the module-level (global) scope.
Example: 
x = 10 # Global variable

def my_function():
    global x
    x = 20 # Modifies the global x
    print("Inside function:", x)

my_function()
print("Outside function:", x)
# Output:
# Inside function: 20
# Outside function: 20

The nonlocal keyword is used within nested functions to access and modify variables in the nearest
enclosing function scope, but not the global one.
Example:
def outer_function():
    x = 10 # Enclosing scope variable

    def inner_function():
        nonlocal x
        x = 20 # Modifies the 'x' in outer_function's scope
        print("Inside inner function:", x)

    inner_function()
    print("Inside outer function:", x)

outer_function()
# Output:
# Inside inner function: 20
# Inside outer function: 20

```

## Pure Function
```
A pure function in Python is a function that consistently produces the same output for the same input
and has no side effects. It is a core concept in functional programming that leads to more predictable
and testable code. 

Key Characteristics: 
Deterministic: The function always returns the same result when given the same arguments, regardless of
when it is called or the program's overall state.
No Side Effects: It does not modify any external state or have any interactions with the outside world
beyond returning a value.
This means it avoids:
* Modifying global variables or variables in an outer scope.
* Performing input/output (I/O) operations, such as printing to the console, reading files, or
accessing databases or networks.
* Depending on external factors like the current time or random number generators.
* Mutating its input parameters (for mutable objects; instead, a new object should be returned with the changes).

Example
# A pure function for adding two numbers
def add(a, b):
    return a + b

# A pure function that returns a new list without modifying the original
def pure_sort(original_list):
    return sorted(original_list)
```

## Map, Filter, Reduce, Zip
```
map()
The map() function applies a given function to each item in an iterable (like a list or tuple)
and returns an iterator of the results. 
Purpose: To transform every element in an iterable.
Example: To square all numbers in a list.
def square(number):
    return number * number

numbers = [1, 2, 3, 4, 5]
squared_numbers_map = map(square, numbers)

# Convert the map object to a list to see the results
print(list(squared_numbers_map))
# Output: [1, 4, 9, 16, 25]
```
---
```
filter()
The filter() function tests each element in a sequence with a function that must return a
boolean value (True or False). It then "filters" out elements for which the function returns False,
returning an iterator of the remaining elements. 
Purpose: To select specific elements from an iterable based on a condition.
Example: To get only the even numbers from a list.
def is_even(number):
    if (number % 2) == 0:
        return True
    else:
        return False

numbers = [1, 2, 3, 4, 5]
even_numbers_filter = filter(is_even, numbers)

# Convert the filter object to a list to see the results
print(list(even_numbers_filter))
# Output: [2, 4]
```
---
```
reduce()
The reduce() function applies a rolling computation to sequential pairs of values in an iterable,
ultimately returning a single, accumulated result. It needs to be imported from the functools module. 
Purpose: To accumulate a single value from an iterable (e.g., sum, product, max).
Example: To calculate the sum of all elements in a list.
from functools import reduce

def custom_sum(first, second):
    return first + second

my_numbers = [1, 2, 3, 4]
sum_result = reduce(custom_sum, my_numbers)

print(sum_result)
# Output: 10
```
---
```
zip()
The zip() function combines elements of multiple iterables into a single iterator of tuples.
Each tuple contains elements from the same index position in the original iterables.
The iteration stops when the shortest iterable is exhausted. 
Purpose: To combine corresponding elements from two or more iterables.
Example: To combine a list of numbers with a list of letters

numbers_list = [1, 2, 3, 4]
letters_list = ['a', 'b', 'c']
combined_zip = zip(numbers_list, letters_list)

# Convert the zip object to a list to see the results
print(list(combined_zip))
# Output: [(1, 'a'), (2, 'b'), (3, 'c')]

```

## List and Set Comprehension
```
List comprehension is a concise and efficient way in Python to create new lists based on existing iterables
(like lists, tuples, or strings), often replacing traditional for loops and lambda functions combined with map() and filter().
It follows a syntax inspired by mathematical set-builder notation. 

Syntax
The basic syntax for a list comprehension is:
new_list = [expression for item in iterable if condition] 

expression: The operation or value to apply to each item to produce elements of the new list.
item: A variable representing the current element in the iterable.
iterable: The source sequence or collection to loop over.
if condition (optional): A filter to include only items that satisfy the condition. 

Examples:
Basic
squares = [x**2 for x in range(10)]
# Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

Filtering with condition
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = [n for n in numbers if n % 2 == 0]
# Output: [2, 4, 6]

If Else Conditional Expression
a = [1, 2, 3, 4, 5]
result = ['Even' if n % 2 == 0 else 'Odd' for n in a]
# Output: ['Odd', 'Even', 'Odd', 'Even', 'Odd']

Nested list comprehension
nested_list = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in nested_list for item in sublist]
# Output: [1, 2, 3, 4, 5, 6]

```
---
```
Set comprehension in Python is a concise and efficient way to create a new set from an existing iterable,
automatically eliminating duplicate elements in the process. It is an elegant, single-line alternative
to using a for loop with the set.add() method. 

Syntax
The basic syntax of a set comprehension is enclosed in curly braces {} and contains an expression, 
a for loop, and an optional if condition for filtering: 
{expression for item in iterable if condition}

expression: The value to be added to the new set in each iteration, which can be a function call
or any valid expression.
item: A variable that represents the current element in the iterable.
iterable: Any Python iterable object, such as a list, tuple, string, or range.
if condition (optional): A filter that determines whether the item should be included in the resulting set. 

Example:
Creating a set from a list
numbers = [1, 2, 2, 3, 4, 4, 5]
unique_numbers = {n for n in numbers}
print(unique_numbers)
# Output: {1, 2, 3, 4, 5}

Applying a transformation
tools = ["Python", "Django", "Flask", "pandas", "NumPy"]
lowercase_tools = {tool.lower() for tool in tools}
print(lowercase_tools)
# Output: {'django', 'numpy', 'flask', 'pandas', 'python'}

Filtering a condition
product_ids = [998, 1001, 1002, 999, 1500]
valid_ids = {pid for pid in product_ids if pid >= 1000}
print(valid_ids)
# Output: {1001, 1002, 1500}

```

## Modules and Packages
```
Modules
A module is the fundamental unit of code organization in Python. 
Definition: A module is a Python file containing functions, classes, and variables 
that can be used in other Python programs.
Creation: You create a module simply by saving a Python file with a .py extension 
(e.g., my_module.py).
Usage: You can import a module using the import statement, after which you can access 
its contents using dot notation (e.g., import my_module; my_module.greet("Alice")). 
You can also import specific components (e.g., from my_module import greet) or 
use aliases (e.g., import my_module as mm).
Examples: Built-in modules include math, random, and os. 
```
---
```
Packages
A package is a way to structure a larger application's module namespace using directories 
and dot notation. 
Definition: A package is a directory that contains multiple modules and, optionally, 
other sub-packages.
Creation: To create a package, you create a directory and add your module files inside it. 
Historically, you needed an __init__.py file to signify the directory as a package, 
but this is no longer required as of Python 3.3 for "namespace packages". 
The __init__.py file can contain initialization code or define which items are exported when 
using from package import *.
Usage: Modules within a package are accessed using dotted module names 
(e.g., import sound.effects.echo imports the echo module from the effects subpackage within 
the sound package).
Examples: Popular third-party packages (often referred to as libraries) include NumPy for 
numerical computing, Pandas for data analysis, and Requests for web requests. 
These are typically installed using the pip package manager from the Python Package Index (PyPI). 

```

## Miscellaneous 
```xml

# Python Keywords 
help() -> This will open the help module
# inside help if you type 'keywords' you will see the list of all the python keywords 
# or 
import keyword
keyword.kwlist


# Escape codes
print ("c\\test\data.txt")
# Other escape codes are as follows: 
\<newline> - Backslash and newline ignored
\\ - Backslash (\)
\' - Single quote (')
\" - Double quote (")
\a - ASCII Bell (BEL)
\b - ASCII Backspace (BS)
\f - ASCII Formfeed (FF)
\n - ASCII Linefeed (LF)
\r - ASCII Carriage Return (CR)
\t - ASCII Horizontal Tab (TAB)
\v - ASCII Vertical Tab (VT) 
https://docs.python.org/3/reference/lexical_analysis.html#grammar-token-python-grammar-stringescapeseq


# Ignore all warning in python
import warnings
warnings.filterwarnings("ignore") # Not advisable 

warnings.filterwarnings("default") # this will bring warnings back 

```
## Python Cheat Sheet
https://zerotomastery.io/cheatsheets/python-cheat-sheet/

## Statistics and Math Course with Python
https://academy.zerotomastery.io/courses/learn-topic8/lectures/50109508

### Reference
```xml
https://www.udemy.com/course/pythoncourse/learn/lecture/4953418#overview
https://docs.python.org/3/reference/
https://www.geeksforgeeks.org/
```
