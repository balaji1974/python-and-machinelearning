
# Python - Basics 

## Install Python
```xml


For python installation download the installer from the below location and install. 
https://www.python.org/downloads/


After installation open the command like and check the installation with the following command:
python3 --version

Another way to install python is part of Anaconda/MiniConda installation, 
which will be covered in the next section and python can be run here using
Jupyter notebook

```

## Basics

### Comments in Python 
```xml
'#' will be used for comment in python
Eg. 
# Name: Balaji 
# Description: Module to calculate average of employee salaries

```

### DataTypes & Operations
```xml
integer - Eg. 1, 9, -5, 1056
float - Eg. 3.5, 4.0, -9.2 
string - Eg. "Balaji", 'Balaji'
boolean - Eg. True or False
none - Eg. None

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
print(7//2) -> This is integer or floor divison and results in 3
print(10**2) -> This is for power
print(10%2) -> This is for reminder

print(10+5*2) -> Complex activities, but presedence is for multiplcation 
Operation precedence can be found in the link below:
https://www.programiz.com/python-programming/precedence-associativity

# Parenthesis, Exponents, Multiplication/Division, Addition/Subtraction - PENDAS
3 + 4 * 5 -> This will follow the PENDAS rule 


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


### Conditions
```xml
IF STATEMENT
------------
An if statement is used to check a condition. If the condition is True, the indented block of code following the if statement is executed.


i = 5
if i>3:
    print(f"{i} is greater than 3")
# Output: 5 is greater than 3


Logical operator 
----------------

and, or, not & xor 

The and operator returns True only when both operands are True.
# 1. Logical and Operator
 
print(True and False)        # output: False
print(True and True)         # output: True
print(True and False)        # output: False
print(False and True)        # output: False
print(False and False)       # output: False


The or operator returns True if at least one of the operands is True.
# 2. Logical or Operator
 
print(True or False)         # output: True
print(False or True)         # output: True
print(True or True)          # output: True
print(False or False)        # output: False

The not operator returns the negation of the bool value.
# 3. Logical not or (!) Operator
 
print(not True)              # output: False
print(not(True))             # output: False
print(not False)             # output: True
print(not(False))            # output: True

The ^ operator, also known as the exclusive or (xor) operator, returns True when the operands have different boolean values.
# 4. Logical ^ (XOR) Operator
 
print(True ^ True)           # output: False
print(True ^ False)          # output: True
print(False ^ True)          # output: True
print(False ^ False)         # output: False



Python provides various operators to compare values and perform logical operations. 
You can compare values using the equal to (==), not equal to (!=), less than (<), greater than (>), less than or equal to (<=), or greater than or equal to (>=) operators.


Any non-zero value in Python will return true 
print(bool(6))  # True
print(bool(-6))  # True
print(bool(0))  # False



The if statement checks a specific condition. 
If the condition is true, it executes the code block underneath it.

The else statement provides an alternative code block that will execute 
if the if statement's condition is not met (i.e., if it's false).

elif stands for "else if". 
This statement allows us to check multiple conditions. 
If the if condition is false, the program checks the elif condition. 
If the elif condition is true, it executes the code block underneath it.


```

### While Loop
```xml
While loop is a powerful control structure that allows repetitive execution 
of a block of code while a specified condition is true. 
It's essential to manage the loop variables properly to avoid 
infinite loops and other logical errors.


The break statement is used to exit a loop when a specific condition is met. 
Here's a simple example to illustrate how it works:
The continue statement is used to skip the current iteration of a loop and 
proceed to the next iteration. Here's how you can use it:

Using while loops:
Continues until a specified condition is met.
Condition evaluated before each iteration.

Using for loops:
Iterates over a known and finite sequence.

Using break statement:
Exits the loop based on a certain condition.

Using continue statement:
Skips the current iteration based on a certain condition.

Common Pitfalls:
Ensure that the condition in the loop eventually evaluates to False to avoid infinite loops.
Be cautious with indentation, as it determines the scope of the loop.

```

### Strings
```xml
Strings in Python are represented with str type. 
You can use either single quotes or double quotes to define a string.
There is no distinct data type for single characters. Both strings and single characters are represented by the str class.


print("Hello World")  # Output: Hello World
print('Hello World')  # Output: Hello World


The type() method allows you to find the type of a variable.
message = "Hello World"
print(type(message))  # Output: <class 'str'>


The str class provides various methods to manipulate and inquire about strings.
Converting to uppercase and lowercase
message = "hello"
print(message.upper())  # Output: HELLO
print(message.lower())  # Output: hello
print("hello".capitalize())  # Output: Hello
print('hello'.capitalize())  # Output: Hello


Checking lower case, title case, and upper case
print('hello'.islower())  # Output: True
print('Hello'.islower())  # Output: False
print('Hello'.istitle())  # Output: True
print('hello'.istitle())  # Output: False
print('hello'.isupper())  # Output: False
print('Hello'.isupper())  # Output: False
print('HELLO'.isupper())  # Output: True




Checking if a string is a numeric value
print('123'.isdigit())  # Output: True
print('A23'.isdigit())  # Output: False
print('2 3'.isdigit())  # Output: False
print('23'.isdigit())   # Output: True




Checking if a string only contains alphabets or alphabets and numerals
print('23'.isalpha())   # Output: False
print('2A'.isalpha())   # Output: False
print('ABC'.isalpha())  # Output: True
print('ABC123'.isalnum())  # Output: True
print('ABC 123'.isalnum())  # Output: False


Checking if a string ends or starts with a specific substring
print('Hello World'.endswith('World'))   # Output: True
print('Hello World'.endswith('ld'))      # Output: True
print('Hello World'.endswith('old'))     # Output: False
print('Hello World'.endswith('Wo'))      # Output: False
print('Hello World'.startswith('Wo'))    # Output: False
print('Hello World'.startswith('He'))    # Output: True
print('Hello World'.startswith('Hell0')) # Output: False
print('Hello World'.startswith('Hello')) # Output: True



Finding a substring within a string
print('Hello World'.find('Hello'))   # Output: 0
print('Hello World'.find('ello'))    # Output: 1
print('Hello World'.find('Ello'))    # Output: -1
print('Hello World'.find('bello'))   # Output: -1
print('Hello World'.find('Ello'))    # Output: -1



You can use the in keyword to check whether a character or sequence of characters exists within a specific set.
print('Hello' in 'Hello World')   # Output: True
print('ello' in 'Hello World')    # Output: True
print('Ello' in 'Hello World')   # Output: False
print('bello' in 'Hello World')    # Output: False


The String Module in Python: It provides a collection of utilities that can be used for common string operations. To use this module, you'll need to import it first.
import string
 
print(string.ascii_letters)        
# Output: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ


You can compare two strings to check if they are the same using the equality (==) operator.
str1 = "test"
str2 = "test1"
print(str1)  # Output: test
print(str2)  # Output: test1
print(str1 == str2)  # Output: False

When comparing strings, ensure that both the strings are of the same case and contain the same characters in the same order, as the comparison is case-sensitive and character-sensitive.
Beyond equality, other comparison operators like != for inequality, < for less than, and > for greater than, are compared lexicographically.
Tip: Remember that strings are compared character by character, based on the ASCII value of the characters, in lexicographical comparison.

print(len('Balaji')) # prints the string length 
print(reversed('Balaji')) # reverses the words

x=text.split() # splits the string into words

print(sorted('Balaji')) #sorts the word - ascii sort, better to apply upper or lower before sorting

s="Hi balaji"
# replace(<old value>, <new value>, <optional-occurance>)
print(s.replace("i","X", 1)) # Will return HX balaji - Only 1 time i will be replaced by X 

# counts no. of occurance
print(s.count('i')) # will return 2 


String samples
--------------
data = "Hello World"
print(type(data))
 
char = 'A'
count = 0
for c in "ABRACADABRA":
    if c == char:
        count += 1
print(count)
 
str1 = "apple"
str2 = "banana"
print(str1 < str2)
 
text = "Python "
repeated_text = text * 3
print(repeated_text)
 
str1 = "Hello"
str2 = " World"
result = str1 + str2
print(result)
 
str1 = "Python"
str2 = "Java"
print(len(str1) == len(str2))

Multiplying a number by another number will return the product of those numbers.
print(1 * 20)                 # output: 20


You can repeat a string by multiplying it with a number. The string will be concatenated the number of times you multiply it.
print('1' * 20)              # output: '11111111111111111111'


You can use this method with any string or character to repeat it.
print('A' * 10)              # output: 'AAAAAAAAAA'

```

### Object Oriented Programming
```xml
***** 
Important Note: 
Python supports Procedural (functions), Object Oriented (Objects) and Functional Programming (programs are constructed by applying and composing functions)
*****

Classes, methods & attributes
In Object Oriented Programming, code is organized around classes and objects.
Example: 
Customer Class:
Attributes: name, address
Methods: login, logout

class Customer:
 
    Initialize name, address
    Method to login: Display "{name} logged in"
    Method to logout: Display "{name} logged out"

Instance of the class is called object 

Class: A blueprint for creating objects.
Object: An instance of a class.
Method: A function defined within a class.
Attribute: Variables belonging to an object.


class Planet:
    pass

--> Python does not use the new keyword like Java for creating new instances.
eg. earth = Planet()

--> Attempting to access an attribute that has not been defined will result in an AttributeError.
eg. print(earth.name)  # This will result in an AttributeError

--> Python allows you to dynamically add attributes to instances.
earth.name = 'The Earth'
print(earth.name)  # Output: 'The Earth'

-->Each object has its own set of attributes
venus = Planet()
print(venus.name)  # This will result in an AttributeError
 
# You need to explicitly set the name for Venus
venus.name = 'Venus'
print(venus.name)  # Output: 'Venus'

--> Both data and methods are considered as attributes. Trying to call a method that doesn’t exist will result in an AttributeError
venus.do_something()  # This will result in an AttributeError


Constructor
-----------
When you create instances of a class, sometimes you want to set an initial state for those instances. This is where constructors come into play.
A constructor is defined using the __init__ method. 
The self parameter is a reference to the instance of the class and is used to access variables that belong to the class.
Eg. 
class MotorBike:
    def __init__(self):
        print("MotorBike instance created")
 
honda = MotorBike()  # Output: "MotorBike instance created"
ducati = MotorBike()  # Output: "MotorBike instance created"

To set the initial attributes for objects, we can add parameters to the constructor.
Eg.
class MotorBike:
    def __init__(self, speed):
        print(speed)
 
honda = MotorBike(50)  # Output: 50
ducati = MotorBike(250)  # Output: 250

To initilize attributes in a constructor we can do the following: 
class MotorBike:
    def __init__(self, speed):
        self.speed = speed
 
honda = MotorBike(50)
ducati = MotorBike(250)
 
print(honda.speed)  # Output: 50
print(ducati.speed)  # Output: 250


If you try to define multiple constructors, Python will only keep the last one.

Naming convention: 
------------------
Variables and methods: Use lowercase and underscores to separate words (e.g., distance_from_sun, get_name).
Classes: Use CamelCase (e.g., MotorBike, Book).

Encapsulation refers to the bundling of data (attributes) and the methods (functions) that operate on the data into a single unit or class.


The 'self' parameter is crucial for instance methods. It allows the method to access the instance on which it was called.
Omitting self can lead to errors, especially when calling the method on an instance.
You can chain method calls on the same instance using self to call other methods within the same class.

Encapsulation
-------------
WHY ENCAPSULATION IS GOOD?
Validation: As demonstrated, encapsulation allows us to include validation logic within our methods, making our objects more robust.
Ease of Maintenance: In the future, if you want to include more complex validations or logic, you only have to change it in one place — inside the method itself.
Abstraction: Consumers of your class don’t need to know the internal details; they just need to call the appropriate methods. This encapsulates (hides) the internal state management logic from the outside world.


Everything is an Object in Python
Even basic data types like integers, booleans, strings, and floats as objects. 
In Python, functions are also treated as objects. This allows you to assign functions to variables and pass them around, giving you a lot of flexibility.
# Functions as objects
print(do_something)  # Output: <function do_something at some_memory_address>
 
# Assigning functions to variables
test = do_something
test()  # Output: something


```

### List Data Structure in Python 
```xml
In Python, you can use the list data structure to simplify the storage and calculation. Here's how you can create a list to store the marks:

marks = [23, 56, 67]
print(sum(marks))  # Outputs: 146 - sum of the list elements
print(max(marks))  # Outputs: 67 - max of all the elements in the list
print(min(marks))  # Outputs: 23 - min of all the elements in the list
print(len(marks))  # Outputs: 3 - size of the list 


Adding Elements :
marks.append(76)
print(marks)  # Outputs: [23, 56, 67, 76]

Inserting in a specific position of the list: 
marks.insert(2, 60)
print(marks)  # Outputs: [23, 56, 60, 67, 76]

Remove a value from the  list:
marks.remove(60)
print(marks)  # Outputs: [23, 56, 67, 76]


Searching and Checking Existence :
print(55 in marks)  # Outputs: False
print(56 in marks)  # Outputs: True
print(marks.index(67))  # Outputs: 2
print(marks)  # Outputs: [23, 56, 67, 76]


Note: If you try to find the index of a value that does not exist in the list, you will get an error:
print(marks.index(69)) # this will throw an error


Iterating Through a List :
for mark in marks:
    print(mark)
# Outputs:
# 23
# 56
# 67
# 76

Delete an element from the list: 
del(mark [3])
print(mark)

Add multiple values to an already existing list
marks.extend([3,7,8])
print(marks)

Another way to extend
marks += [11,12]
print(marks)


Append to an existing list
marks.append(99)
print(marks)


Python lists can store multiple data types within the same list, such as integers, strings, and even other lists!
Python lists are dynamic, meaning they can grow and shrink in size as needed, unlike arrays in some other languages.
Python supports negative indexing for its sequences. The index of -1 represents the last item in the list, -2 represents the second to last item, and so on.

# Sort and reverse
numbers = [4, 2, 9, 1]
numbers.sort() # sort 
print(numbers) 
numbers.reverse() # reverse 
print(numbers) 

The reverse() method directly modifies the original list. Conversely, reversed() yields an iterator that facilitates looping through elements in reverse, but it doesn't alter the list.
sort() modifies the original list. On the other hand, sorted() returns a sorted version, preserving the original list unchanged.
You can conveniently use both sorted() and reversed() directly within loops for iterating over elements in a specific sequence.
By passing the key argument to sorted(), you can set custom sorting logic. The direction of sorting is determined by the reverse argument.

```

### 2D List Data Structure in Python 
```xml
A 2D list is a list of list 
Eg.
numbers=[[4, 2, 9, 1],[4, 2, 9, 1],[4, 2, 9, 1]]

```

### List of strings
```xml
Ascii value
char='C'
ord(char) -> This will return the ascii value of  'C'
chr(ord(char)+1) -> This will return 'D' -> converts the ascii back to character 

You can define functions with a variable number of arguments. 
These are known as variable arguments, and they allow you to pass a varying number of values to a function.
The *args syntax allows you to pass a variable number of positional arguments to a function. 

```

### Advance OOPS concepts 
```xml
# Method to 'represent' the object as a string - Equivalent to toString method in Java
def __repr__(self):
	return repr((self.member_var1, self.member_var2, self.member_var3, self.member_var4)) # note double brackets 

-> Object composition is the process of adding one object within another object 

-> Inheritance is an OOP feature that allows one class to inherit properties and behaviors (methods) from another class. 
The class that is inherited from is known as the “superclass” or “parent class,” and the class that inherits is called the “subclass” or “child class.”

Eg.
class Pet(Animal): -> Here Pet class inherits the properties of Animal class 
    def groom(self):
        print("groom")


-> Starting Python 3, every class implicitly inherits from the built-in object class unless explicitly specified otherwise. 
The object class provides default implementations for a number of methods, including __repr__, which is used for the string representation of an instance.

-> Unlike Java, python supports multiple inheritance 
Multiple inheritance allows a single class to inherit from more than one class.

-> The below code calls the super class constructor in Python
super().__init__()


-> An abstract class serves as a blueprint for other classes. 
It allows you to define methods that must be created within any child classes built from the abstract class. 
In other words, an abstract class can contain methods that have no implementation in the base class itself.

Eg. abstract class

from abc import ABC, abstractmethod 
class AbstractAnimal(ABC):
    @abstractmethod
    def bark(self): pass


-> Template Method Pattern: Introduced a design pattern that uses abstract classes to define an algorithm structure, leaving the implementation of individual steps to subclasses.
Eg. 
class AbstractRecipe(ABC):
 
    def execute(self): # this is the template method ensuring right order of calling abstract methods
        self.prepare()
        self.recipe()
        self.cleanup()
 
    @abstractmethod
    def prepare(self): pass
 
    @abstractmethod
    def recipe(self): pass
 
    @abstractmethod
    def cleanup(self): pass

-> Python supports polymorphism, means "many forms."

```

### List Data Structure - Advance Concepts
```xml

List Slicing
------------

You can extract elements from a list using slicing by specifying the start and end indexes.
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
print(len(numbers))  # Output: 10
print(numbers[2])    # Output: 'Two'
print(numbers[2:6])  # Output: ['Two', 'Three', 'Four', 'Five']

By leaving out the start or end index, you can capture all elements from the start or right up to the list's end.
print(numbers[:6])  # Output: ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
print(numbers[3:])  # Output: ['Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

Introduce a step to fetch every nth element.
print(numbers[1:8:2])  # Output: ['One', 'Three', 'Five', 'Seven']
print(numbers[1:8:3])  # Output: ['One', 'Four', 'Seven']
print(numbers[::3])    # Output: ['Zero', 'Three', 'Six', 'Nine']

You can reverse a list by employing a step of -1.
print(numbers[::-1])   # Output: ['Nine', 'Eight', 'Seven', 'Six', 'Five', 'Four', 'Three', 'Two', 'One', 'Zero']
print(numbers[::-3])   # Output: ['Nine', 'Six', 'Three', 'Zero']

Leverage slicing to eliminate list elements.
del numbers[3:]
print(numbers)         # Output: ['Zero', 'One', 'Two']
 
Utilize slicing to modify values in a list.
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
numbers[3:7] = [3, 4, 5, 6]
print(numbers)        # Output: ['Zero', 'One', 'Two', 3, 4, 5, 6, 'Seven', 'Eight', 'Nine']

```

### Stack & Queue Data Structure 
```xml
Stack:
A stack is a LIFO (Last In, First Out) data structure. This means the last element you insert is the first one you take out.

Operations in Stack 
Push: To add an element to the top of the stack.
Pop: To remove the top element from the stack.
Top: To look at the top element without removing it.
IsEmpty: To check if the stack is empty.


Queue: 
A queue follows a FIFO (First In, First Out) principle.

Operations in Queue 
Enqueue: To add an element to the rear of the queue.
Dequeue: To remove the front element from the queue.
Front: To look at the front element without removing it.
IsEmpty: To check if the queue is empty.

```

### List Comprehension
```xml
List comprehension is a concise way to create lists in Python.
List comprehension offers a powerful, readable, and concise way to create lists by applying expressions and conditionals to elements in existing lists. 
By using list comprehension, you can reduce the amount of code needed and increase the readability of your code, especially when filtering or transforming elements.

Eg. 
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
numbers_length_four = []
numbers_length_four = [number for number in numbers if len(number) == 4]
print(numbers_length_four)  # ['Zero', 'Four', 'Five', 'Nine']


List comprehensions offer a concise way to create lists by applying expressions and conditionals to elements in existing lists.
They can be used to filter elements based on specific criteria, resulting in more efficient and readable code.
They can be applied to various types of data, including strings, numbers, and even complex data structures.
Using list comprehensions can lead to code that is easier to understand and maintain, especially when dealing with complex filtering or transformation operations.

```


### Time Complexity & Recursion
```xml
Time complexity measures how the runtime of an algorithm grows as the size of the input grows.
O(1) means the time to complete is constant and does not depend on the input size.
O(n) means the time to complete grows linearly as the input size grows.
O(n^2) means the time to complete grows quadratically as the input size grows.


Recursion is a technique where a function calls itself within its definition.
It simplifies complex problems by breaking them down into smaller sub-problems.
It provides a clean, elegant approach to solving problems.
Always include a base case to terminate recursion.
Recursive methods can consume more memory due to the call stack.


```

### Search Algorithm
```xml
Linear Search Algorithm
-----------------------
Time complexity: O(n)
No data sorting required.
It works with unsorted data.
It is easy to implement.


Binary Search Algorithm
-----------------------
Fast search algorithm for sorted data.
Time complexity: O(log n)
It works with sorted data. 


```

### Exception Handling
```xml

A built-in means an object directly accessible to Python code without an import statement
import builtins
help(builtins.ZeroDivisionError) # Shows object hirarchy of the exceptions, e.g., ZeroDivisionError


Key Elements of exception handling: 
All exceptions in Python inherit the BaseException class 

User-Friendly Messages: Always communicate effectively with the end-user by showing meaningful error messages.
Debugging Information: When an exception occurs due to a programmatic error, don’t just suppress it. Log sufficient information to aid debugging efforts.


Python exception handling:
try:
	<block of statements>
except: 
	<block of statements>
else: 
	<block of statements>
finally:
	<block of statements>

-> except is executed when an exception occurs and the exception type matches.
-> else is executed only when no exception occurs.
-> finally is always executed, regardless of whether an exception occurs or not.
Rules: 
With a try, you can have multiple except blocks to handle different types of exceptions.
You cannot have an else block without preceding except blocks. An else always follows one or multiple except blocks.
You can have a try with just a finally block. Even if an exception occurs, the code inside the finally block will be executed.


Catching specific exceptions:
try:
	<block of statements>
except TypeError:  # catching TypeError
	<block of statements>
except ZeroDivisionError: # catching Zero Division Error
	<block of statements>


Catching multiple exceptions:
try:
	<block of statements>
except (ZeroDivisionError, TypeError):  # catching Zero Division Error & TypeError
	<block of statements>


All Exceptions in Python follow exception Hirearchy: 
Eg. 
BaseException
    Exception
        ArithmeticError
            FloatingPointError
            OverflowError
            ZeroDivisionError


Accessing exception details with the exception block: 
try:
    sum([1, '1'])
except TypeError as error: # error has the details of the exception 
    print(error)


Exception best practises:
Never Hide Exceptions: Always log information. This aids in debugging and provides valuable context to whoever needs to solve the problem.
Think About the User: If an exception occurs, consider what the user should see and what actions they can take.
Support the Support Team: Provide all the necessary information in the logs or alerts so that your support team (or you) can diagnose and fix issues efficiently.
Consider the Calling Method: If you’re designing an API, think about what the caller can actually do with the exception information. Make your exceptions as informative as possible.
Global Exception Handling: Implement a top-level exception handler to catch any unhandled exceptions. Ensure the user sees a friendly, actionable message.

```

### Data Structure - Set
```xml
A set in Python does not contain duplicates.
Set is represented by {} 

List vs Set 
-----------
Duplication of Elements:
List: Allows duplicate elements.
Set: Does not allow duplicate elements. If you try to add a duplicate element, it won't be added.

Order of Elements:
List: Maintains the order of elements as they were added.
Set: Does not maintain any specific order of elements.

Accessing Elements:
List: Accessed by index, which is a positional value.
Set: Cannot be accessed by index, as it is an unordered collection.

Declaration:
List: Uses Square Brackets.  [1,2,3,4,5]
Set: Uses Curly Braces. {1,2,3,4,5}

Packing elements into lists and sets provides a flexible way to pass multiple arguments to functions and manage collections of data effectively.
You can use the * operator to unpack elements from a list and pass them as arguments to a function.

Union = | -> Combines elements of both set by exculding duplicate elements 
Intersection = & -> Returns the elements common to both sets 
Difference = - -> Removes the duplicate of first and second sets and returns the remaining elements in the first set 

```

### Data Structure - Dictionary
```xml
A dictionary in Python is a collection of key-value pairs.
It provides methods to access, modify, iterate through, and delete keys and values. 
It offers flexibility and efficiency in managing collections of data where the index can be anything, not just a number.

Eg. 
occurances = dict(a=5, b=6, c=8)
print(occurances)  # {'a': 5, 'b': 6, 'c': 8}
print(type(occurances))  # <class 'dict'>

See example code for usage 

# Different ways to create a dictionary
a = dict (one=1, two=2, three=3)
b = {"one":1, "two":2, "three":3}
c = dict(zip(["one", "two", "three"],[1, 2, 3]))
d = dict([("one",1),("two",2),("three",3)])
e = dict({"one":1, "two":2, "three":3})

```

### Data Structure - Tuples
```xml
A tuple is an immutable sequence of values.
Tuples can be used to return multiple values from a function.
You can destructure a tuple to assign its values to separate variables.
Tuples support standard sequence operations like len and indexing.


A tuple is immutable, meaning that once defined, its content cannot be altered.
Being immutable, tuples can be more memory-efficient and faster in certain situations.
Tuples are often used to represent a single set of related attributes or a record. For example, a tuple might represent details about a person, such as name, age, and country.
Eg. 
person = ('Alice', 30, 'USA')
print(person)  # Outputs: ('Alice', 30, 'USA')

person =("Hi", 37) # Can be reassigned new set of values 
print(person)  # Outputs: ('Hi', 37)

print(person[1]) # Outputs: 37

# person[1] = "bala" # Will not work as tuples are immutable, once assigned cannot change 

person2 =("Hi All", 50)

person3 = person + person2
print(person3)  # Outputs: ('Hi', 37, 'Hi All', 50)

```

## Check out codes in the sample code section
```xml
guessnumber.py -> Number guessing game (guess between 1 to 99)
rolldice.py -> Rolling dice game (roll dice until you dont want to roll again)

```

### Reference
```xml
https://www.udemy.com/course/python-programming-for-beginners-with-exercises/
https://www.python.org/downloads/

https://www.udemy.com/course/data-analysis-with-pandas/
https://www.udemy.com/course/pythoncourse/

```