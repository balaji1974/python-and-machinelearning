#  Sample Error types in Python

# ZeroDivisionError: This occurs when you try to divide a number by zero.
>>> 1/0
ZeroDivisionError: division by zero

# TypeError: This arises when an operation is performed on an inappropriate data type.
>>> '2' + 2
TypeError: unsupported operand type(s) for +: 'int' and 'str'

# NameError: This happens when you try to access an undefined variable.
>>> value
NameError: name 'value' is not defined

# AttributeError: This occurs when you try to access a non-existent attribute of an object.
>>> values.non_existing
AttributeError: 'list' object has no attribute 'non_existing'


# IndentationError: This occurs due to improper indentation
>>>     values = [1,'1']
IndentationError: unexpected indent


# List all errors in python
import builtins # a built-in means an object directly accessible to Python code without an import statement
# List all names in builtins module
builtin_names = dir(builtins)
 
# Print first 100 names as an example
print("First 100 built-in names:", builtin_names[:100])
 
# Show help for one of the exceptions, e.g., ZeroDivisionError
help(builtins.ZeroDivisionError)


# Exception Handling
try:
    i = 0
    j = 10/i
except:
    print("Exception caught!")
    j = 0
print(j)


# Catching specific exceptions
try:
    i = 0  # Change this value to see different outcomes
    j = 10/i
    values = [1, '1']
    print(sum(values))
except TypeError: # catching TypeError
    print("TypeError")
    j = 0
except ZeroDivisionError: # catching Zero Division Error
    print("ZeroDivisionError")
    j = 0
print(j)
print("End")


# Sample Exceptions 

# Specific Error 
try:
    10/0
except TypeError:
    print("TypeError")
except ZeroDivisionError:
    print("ZeroDivisionError")
print("End")

# Catching classes that do not inherit from BaseException is not allowed
try:
    10/0
except object: # catching classes that do not inherit from BaseException is not allowed
    print("ZeroDivisionError")
print("End")

# Catching all Exceptions
try:
    10/0
except Exception:
    print("Exception")

# Catching multiple exceptions in a single block
try:
    sum([1, '1'])
except (ZeroDivisionError, TypeError):
    print("Exception")

# Accessing exception details: 
try:
    sum([1, '1'])
except TypeError as error:
    print(error)


# Else and Finally blocks
try:
    i = 1  # Simulating input from user
    j = 10/i
except Exception as error:
    print(error)
    j = 0
else:
    print("Else")
finally:
    print("Finally")
print(j)
print("End")
# Output
# Else
# Finally
# 10.0
# End


# As a bare minimum every try block will have a finally block or an except block 
try:
    i = 0  # Simulating input from user
    j = 10 / i
finally:
    print("Finally")  # This line will be executed
# This line will NOT be executed because an exception is raised and not caught
print("End")


# Throwing or raising our own exceptions 
class Currency:
    def __init__(self, currency, amount):
        self.currency = currency
        self.amount = amount
 
    def __repr__(self):
        return repr((self.currency, self.amount))
 
    def __add__(self, other):
        if self.currency != other.currency:
            raise Exception("Currencies Do Not Match")
        total_amount = self.amount + other.amount
        return Currency(self.currency, total_amount)
 
# Testing the Currency class
value1 = Currency("USD", 20)
value2 = Currency("USD", 30)
print(value1 + value2)  # Output should be ('USD', 50)
 
value3 = Currency("INR", 30)
# The following line should raise an exception: "Currencies Do Not Match"
print(value1 + value3)


# Creating custom exception class and using them
class CurrenciesDoNotMatchError(Exception):
    def __init__(self, message):
        super().__init__(message)
 
class Currency:
    def __init__(self, currency, amount):
        self.currency = currency
        self.amount = amount
 
    def __repr__(self):
        return repr((self.currency, self.amount))
 
    def __add__(self, other):
        if self.currency != other.currency:
            raise CurrenciesDoNotMatchError(self.currency + " " + other.currency)
        total_amount = self.amount + other.amount
        return Currency(self.currency, total_amount)
 
value1 = Currency("USD", 20)
value2 = Currency("INR", 30)
print(value1 + value2)  # Should raise CurrenciesDoNotMatchError: USD INR


