
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


```


### Reference
```xml
https://www.udemy.com/course/pythoncourse/learn/lecture/4953418#overview
https://docs.python.org/3/reference/
```