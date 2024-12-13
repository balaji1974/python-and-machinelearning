# to check the angle of the triange is 180 or not 

def is_valid_triangle(angle1, angle2, angle3): 
    if angle1<=0:
        return False
    if angle2<=0:
        return False
    if angle3<=0:
        return False
        
    return angle1+angle2+angle3 ==180


print(is_valid_triangle(90,60,30)) # returns true
print(is_valid_triangle(90,30,30)) # returns false


# Function that calculates the sum of all divisors of a given integer number
def calculate_sum_of_divisors(number):
    sum=0
    if number<=0:
        return sum
    for i in range(1,number+1):
        if number%i==0:
            sum+=i
    return sum

print(calculate_sum_of_divisors(6))   # Output: 12
print(calculate_sum_of_divisors(15))  # Output: 24


# Code to test  if a Number is a Perfect Number 
# A perfect number is a positive integer that is equal to the sum of its positive divisors, excluding itself.
def is_perfect_number(num) :
    if(num<=0) :
        return False
    sum=0
    for i in range(1, num-1) :
      if(num%i==0): 
          sum+=i
    return num==sum

    
print(is_perfect_number(6))  # Output: True
print(is_perfect_number(28))  # Output: True
print(is_perfect_number(5))  # Output: False


# function to find the last digit of a given number
def get_last_digit(num) :
    return num%10
    
print(get_last_digit(123))  # Output: 3
print(get_last_digit(9087))  # Output: 7
print(get_last_digit(6))  # Output: 6


# Python function to check if both numbers are even 
def are_both_even(i,j) :
    return (i%2==0) and (j%2==0)
    
print(are_both_even(4, 2))  # Output: True
print(are_both_even(3, 4))  # Output: False


# Function to find if a given year is leap year or not 
# A year is a leap year in the Gregorian calendar if:
# It is divisible by 4 (AND)
# It is NOT divisible by 100 (except when it is divisible by 400)
def is_leap_year(year) :
    if (year<1) :
        return False
    if not (year % 4==0) :
        return False
    if not (year%100==0) :
        return True
    if (year%400==0) :
        return True
    return False
    
print(is_leap_year(2200))


# Function to check a right angle triangle using Pythagorean theorem
def is_right_angled_triangle(side1, side2, side3) :
    if side1 <= 0 or side2 <= 0 or side3 <=0:
        return False
    if side1**2 + side2**2 == side3**2 :
        return True
    if side2**2 + side3**2 == side1**2 :
        return True
    if side3**2 + side1**2 == side2**2 :
        return True
    return False 
    
print(is_right_angled_triangle(4, 5, 7))  # Output: True


# Sample if elif and else statements 
if(False):
   print("False")
 
if(True):
   print("True")
 
x = -6
if x:
   print("something")
 
 
k = 15
if (k > 20):
  print(1)
elif (k > 10):
  print(2)
elif (k < 20):
  print(3)
else:
  print(4)
 
 
l = 15
if (l < 20):
    print("l<20")
if (l > 20):
    print("l>20")
else:
    print("Who am I?")
 
a = 10
b = 20
if a > 5:
    if b < 30:
        print("Inner condition met")
    else:
        print("Inner condition not met")
else:
    print("Outer condition not met")
 
 
m = 15
if m>20:
    if m<20:
        print("m>20")
    else:
        print("Who am I?")
 
 
number = 5
if number < 0:
  number = number + 10
number = number + 5
print(number) 
 
 
number = 5
if number < 0:
  number = number + 10
  number = number + 5
print(number) 
 
number = 5
if(number%2==0):
   isEven = True
else:
   isEven = False
print(isEven)
 
x = 10
y = 5
if x > 5 and y < 10:
    print("Condition 1")
elif x == 10 or y == 5:
    print("Condition 2")
else:
    print("Condition 3")
 
x = 5
if not x == 3:
    print("x is not equal to 3")
else:
    print("x is equal to 3")
 
 
number = 4
isEven = True if number%2==0 else False
print(isEven)



# if else student grade assigment
def assign_grade(marks) :
    grade=''
    if marks>=90:
        grade='A'
    elif marks>=80: 
        grade='B'
    elif marks>=70: 
        grade='C'
    elif marks>=60: 
        grade='D'
    elif marks>=50: 
        grade='E'
    else : 
        grade='F'
    return grade    

print(assign_grade(85))

# Weather advisory code using if elif and else
def provide_weather_advisory(temp) :
    advisory=""
    if(temp<0) :
        advisory="It's freezing! Wear a heavy coat."
    elif(temp<=10):
        advisory="It's cold! Bundle up."
    elif(temp<=20): 
        advisory="It's cool! A light jacket will do."
    else:
        advisory="It's warm! Enjoy the day."
    return advisory
    
print(provide_weather_advisory(15))  # Output: "It's cool! A light jacket will do."

