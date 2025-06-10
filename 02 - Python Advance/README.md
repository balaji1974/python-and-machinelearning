
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


### Reference
```xml
https://www.udemy.com/course/pythoncourse/learn/lecture/4953418#overview
https://docs.python.org/3/reference/
https://www.geeksforgeeks.org/
```