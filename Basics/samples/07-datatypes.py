print(type(5))
print(type(7.5))
print(type(4/2))


value1 = 4.5
value2 = 3.2
print(value1 + value2)  # Output: 7.7
print(value1 - value2)  # Output: 1.2999999999999998
print(value1 / value2)  # Output: 1.40625
print(value1 % value2)  # Output: 1.2999999999999998

i = 10
print(i + value1)  # Output: 14.5
print(i - value1)  # Output: 5.5
print(i / value1)  # Output: 2.2222222222222223


#function that calculates simple interest
def calculate_simple_interest(principal, interest, duration) :
    return principal + (principal * interest * 0.01 * duration)
    
print(calculate_simple_interest(10000, 5, 5)) 


# // -> Performs integer division
a=5
b=2
print(a//b) # returns integer value
print(a/b) # returns float value 

# power operator
a=5
b=2
print(a**b) 
print(pow(a,b)) 

# Rounding numbers
print(round(5.6))  # Output: 6
print(round(5.4))  # Output: 5
print(round(5.5))  # Output: 6
print(round(5.67, 1))  # Output: 5.7
print(round(5.678, 2))  # Output: 5.68

# Decimals 
a=4.5
b=3.2

print(a-b)

import decimal 
from decimal import Decimal 
a=Decimal('4.5') - Decimal('3.2')
print(a)

# Constants 
import math
print(math.pi)       # Outputs: 3.141592653589793
print(math.e)        # Outputs: 2.718281828459045

#Boolean
print(True) # 'true' is not used in python and 'True' is used to denote boolean
print(False) # 'false' is not used in python and 'False' is used to denote boolean
 
# Int value of True and False
print(int(True)) # will return 1 
print(int(False)) # will return 0


i=10
print(i >= 15)  # Output: False
print(i >= 10)  # Output: True
print(i > 10)  # Output: False
print(i <= 10)  # Output: True
print(i < 10)  # Output: False
print(i == 10)  # Output: True
print(i == 11)  # Output: False


# Sample Exercises
x = 5
y = 3.5
z = True
print(type(x))
print(type(y))
print(type(z))
 
a = 5
b = 2.5
result = a + b
print(result)
 
c = True
d = False
result = c + d
print(result)
 
num_int = 7
num_float = float(num_int)
print(num_float)
 
num_float = 3.14
num_int = int(num_float)
print(num_int)
 
is_raining = True
int_value = int(is_raining)
print(int_value)
 
str_num = "123"
num = int(str_num)
print(num)
 
str_num = "3.14"
num = float(str_num)
print(num)
 
num = 10
str_num = str(num)
print(str_num)
 
is_true = True
str_bool = str(is_true)
print(str_bool)
 
print(bool('True'))
print(bool('true'))
print(bool('false'))
print(bool(''))

# function to check age >=18 
def can_access_library(age) :
    return (age>=18)

print(can_access_library(17))

# function to check the max speed is equal to 200 km/hr
def is_eligible_for_race(max_speed):
    return (max_speed==200)
    
print(is_eligible_for_race(150))  # Output: False
print(is_eligible_for_race(200))  # Output: True


