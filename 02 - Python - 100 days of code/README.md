
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
pritn(5%3) # will return 2 which is the reminder after division

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
If else:
-------
if condition:
	do this
else:
	do this

# To check if the number is odd or even
num=int(input("Enter the number to check odd or even: "))
if num % 2 == 0:
    print("This is a even number")
else:
    print("This is a odd number")


Nested If Else:
--------------
if condition:
	if another condition:
		do this
	else:
		do this
else:
	do this 

# To pay different price based on age 
print("Welcome to the rollercoaster!")
height = int(input("What is your height in cm? "))
if height >= 120:
    age = int(input("What is your age? "))
    if age >= 21:
        print("Please pay $15 for ticket")
    else:
        print("Please pay $7 for ticket")
else:
    print("Sorry you have to grow taller before you can ride.")

If, Elif Else
-------------
if condition1:
	do A
elif condition2:
	do B
else:
	do C

# To pay based on age groups and photo needed or not
print("Welcome to the rollercoaster!")
height = int(input("What is your height in cm? "))
bill=0
if height >= 120:
    print("You can ride the rollercoaster")
    age = int(input("What is your age? "))
    if age <= 12:
        print("Child tickets are $5.")
        bill=5
    elif age <= 18:
        print("Please pay $7.")
        bill=7
    else:
        print("Please pay $12.")
        bill=12
    photo = input("Do you want a photo? ")
    if photo == "y":
        bill+=3
    print(f"Your final ticket price is ${bill}")
else:
    print("Sorry you have to grow taller before you can ride.")


Logical Operators:
------------------
and => both expressions to be true to evalutate to true
or => any expression to be true to evaluate to true 
not => inverse, true will evulate to false and false will evaluate to true

age >= 45 and age <= 65 can be written in a simpler way as:
45 <= age <= 65 


```

## Day 4 - Randomisation and List 
```xml
# import the random module
import random

#This will produce a random whole number between 1 and 10 (inclusive).
rand_num = random.randint(1, 10)
print (rand_num)

# This will produce a random number between 0.0 and 1.0
# 0.0 <= random.random() < 1.0
rand_num_0_to_1 = random.random()
print(rand_num_0_to_1)

# This will generate a random number between 0 and 5.
print(random.random() * 5)

# This will also generate a random floating point number between 1 and 10.
random_float = random.uniform(1, 10)
print(random_float)

# Head or Tail based on random int 
head_tail = random.randint(0, 1)
if head_tail==0:
    print("Head")
else:
    print("Tail")


# Python List 
# Inbetween square brackets
fruits = ["Cherry", "Apple", "Pear"]

# This will print Cherry
print(fruits[0])

# This will print Apple, from the end 2 places backwards
print(fruits[-2])

# Alter list
fruits[2] = "Orange"
print(fruits)

# Append will add to the end of the list
fruits.append("Mango")
print(fruits)

# Extend will add a list of elements to the end of the existing list
fruits.extend(["Pears", "Avacodo", "Grapes"])
print(fruits)

# Insert an element in the index position
fruits.insert(3,"Pineapple")
print(fruits)

# Removes and returns the element at the specified index
print(fruits.pop(1))
print(fruits)

# Removes the first occurance of the element
fruits.remove("Avacodo")
print(fruits)


# Pick an item randomly from the list - challenge
friends = ["Alice", "Bob", "Charlie", "David", "Emanuel"]

import random
# Print random item from the list
print(friends[random.randint(0,len(friends)-1)]) # Total length -1 as upper limit 

# Same can be achieved with the choice function in random module
print(random.choice(friends))

# Nested List
fruits = ["Cherry", "Apple", "Pear"]
veg = ["Cucumber", "Kale", "Spinnach"]

#The list would look like this: [["Cherry", "Apple", "Pear"], ["Cucumber", "Kale", "Spinnach"]]
fruits_and_veg = [fruits, veg]
print(fruits_and_veg)

```

## Day 5 - Loops
```xml
fruits = ["Apple", "Peach", "Pear"]

# For Loops
for fruit in fruits:
    print(fruit)

# sum
student_scores = [150, 142, 185, 120, 171, 184, 149, 24, 59, 68, 199, 78, 65, 89, 86, 55, 91, 64, 89]
total = 0
for score in student_scores:
    total += score
print(total)
# without loop
print(sum(student_scores))

# max
scores = [8, 65, 89, 86, 55, 91, 64, 89]
max_score= 0
for score in scores:
    if score > max_score:
        max_score=score
print(max_score)
# without loop
print(max(scores))


# Range function 
# Will print from 1 to 100
for number in range(1, 101):
    print(number)

# Will print from 1 to 100 with intervals of 3 
for number in range(1, 101, 3):
    print(number)


## Exercise
# Program to print random password
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

print("Welcome to the PyPassword Generator!")
nr_letters = int(input("How many letters would you like in your password?\n"))
nr_symbols = int(input(f"How many symbols would you like?\n"))
nr_numbers = int(input(f"How many numbers would you like?\n"))

# To print password in the order of letters first, numbers next and finally symbols
import random
pwd=''
for l in range (0,nr_letters):
    pwd += random.choice(letters)
for n in range (0,nr_numbers):
    pwd += random.choice(numbers)
for s in range (0,nr_symbols):
    pwd += random.choice(symbols)
print(pwd)

# To print password in completely random order of letters, numbers and symbols
pwd_list=[]
for l in range (0,nr_letters):
    pwd_list.append(random.choice(letters))
for n in range (0,nr_numbers):
    pwd_list.append(random.choice(numbers))
for s in range (0,nr_symbols):
    pwd_list.append(random.choice(symbols))
random.shuffle(pwd_list) # shuffle the list
res = ''.join(pwd_list)  # create a string by joining elements of the list
print(res)


```

## Day 6 - Functions and Karel 
```xml


```

### Reference
```xml
Dr. Angela Yu, Developer and Lead Instructor
https://www.udemy.com/course/100-days-of-code/


```