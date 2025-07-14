
# Python - 100 days of code

## Install Python and Setup Course
```xml
Step 1
Download and install Python to your computer. 
Head over to the official Python website to download the latest version 
of Python for your computer system. Then complete the installation.

Make sure you tick the box to Add to PATH / Add to Environment Variables. 
You will also need tcl/tk later in the course, 
so a Custom install gives you access to all of these settings.

https://www.python.org/downloads/

Step 2
Download the Free Community Edition of PyCharm using the link below:
https://www.jetbrains.com/edu-products/download/#section=pycharm-edu

Step 3
Once PyCharm installs successfully, you should see the welcome screen. 
Click on the Learn tab and click "Enable Access".

Step 4
Once all the required plugins install successfully. 
Close and Restart PyCharm.


Step 5
Head over to the link below and click "Open in PyCharm".
https://plugins.jetbrains.com/plugin/25212-100-days-of-code--the-complete-python-pro-bootcamp?noRedirect=true

```

## Day 1 - Printing, String manuplation, Input, Variable  
```xml
print("Hello World") # print with paranthesis
print("Hello World\nHello World") # print in 2 different lines with line break
print("Hello "+"world") # print by concat 2 strings 


# Comment
print("Hello "+input("What is your name? ") + "!") # input function within print function

# Variable
username = input("What is your name? ") # storing input to variable
print(username) # print 
length =len(username) # find length 
print(length) # print

# All together
print("Welcome to the band name generator")
city = input("What is the name of the city you grew up in? ")
pet = input("What is your pet name? ")
print("Your band name could be "+city+" "+ pet)

```


## Day 2 - Data types and String Manupulations 
```xml
# Subscripting
print("Hello"[4]) # this will print o
print("Hello"[-3]) # this will print l, ie 3 chars from reverse 

# Integer
print(123+456) # will result in 579

# Large numbers
print(123456789)
print(123_456_789) # both represent same number and _ just for vizuvalization

# floating point
print(123.45)

#Boolean
print(True)
print(False)


#Printing data type - this will print the data type 
print(type("abc")) # <class 'str'>
print(type(124)) # <class 'int'>
print(type(124.67)) # <class 'float'>
print(type(True)) # <class 'bool'>

# Type conversion
# will convert string to integer data type
print(int("134")+int("123"))

int() -> to convert to integer 
str() -> to convert to string 
float() -> to convert to float
bool() -> to convert to boolean 

# Mathematical Operations
print(7+6) # 13
print(65-4) # 61
print(9*3) # 27
print(25/5) # 5.0, division will always result in floating point number
print(25//5) # 5, will just remove the decimal part after division 
print(5**3) # 125, which is 5 to the power of 3 


# Python follows PEDMAS rule 
print(3 * 3 + 3 / 3 - 3) # will result in 7.0
print(3 * (3 + 3 / 3 - 3)) # will result in 3.0


# Number manipulation
bmi = 84 / 1.65 ** 2
print(bmi)

print(int(bmi)) # floor or truncates the decimal value
print(round(bmi)) # rounds the integer value
print(round(bmi,2)) # rounds with 2 decimal places

# Assignment operators such as the addition assignment operator 
+= will add the number on the right to the original value of the variable 
on the left and assign the new value to the variable.

+=
-=
*=
/=

score = 5
score += 1 # increment by 1
print(score)
score -= 2 # decrement by 2
print(score)

# f-Strings 
# In Python, we can use f-strings to insert a variable 
# or an expression into a string.

age = 12
height = 5.3
# Below will output I am 12 years old and my height is 5.3.
print(f"I am {age} years old and my height is {height}") 

# Format the output after rounding to 2 decimal places
subtotal=round((bill+ (bill * (tip/100)))/people,2)
print(format(subtotal, '.2f')) # will print 55.00 for bill=150, tip=10% and people=3

```

## Day 3 - Control flow and logical operators
```xml


```

### Reference
```xml
Dr. Angela Yu, Developer and Lead Instructor
https://www.udemy.com/course/100-days-of-code/


```