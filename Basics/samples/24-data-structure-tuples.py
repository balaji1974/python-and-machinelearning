# Tuples can be defined by simply separating values with a comma. 
# You can return multiple values from a function as a tuple.
def create_my_tuple(): # returns a tuple containing three values.
    return 'Balaji', 1980, 'India'

# When you call the method, you can assign the resulting tuple to a variable.
my_tuple = create_my_tuple()
print(type(my_tuple)) # <class 'tuple'>


# You can assign the values from a tuple to individual variables. 
# This is called destructuring.
# Attempting to destructure a tuple into a wrong number of variables will result in an error. 
# Make sure that the number of variables matches the number of elements in the tuple.
# This will cause an error if 'my_tuple' doesn't have exactly 3 elements
name, year, country = my_tuple
print(name)       # Outputs: Ranga
print(year)       # Outputs: 1981
print(country)    # Outputs: India


# You can find the length of a tuple using the len function and access its elements using an index.
print(len(my_tuple))  # Outputs: 3
print(my_tuple[0])    # Outputs: Balaji
print(my_tuple[1])    # Outputs: 1980
print(my_tuple[2])    # Outputs: India


# Opertions on Tuples
# count of all elements in a tuple
my_tuple = (4, 5, 6, 7, 8)
print(len(my_tuple)) # 5


# value of one element of a tuple  
my_tuple = (100, 200, 300)
a, b, c = my_tuple
print(b) # 200

# combining two tuples with addition   
tuple_1 = (1, 2, 3)
tuple_2 = (4, 5, 6)
result = tuple_1 + tuple_2
print(result) # (1, 2, 3, 4, 5, 6)
 
# returning multiple values from function using tuple
def my_function(x, y):
    return x + y, x - y
result = my_function(10, 5)
print(result) # (15, 5)
 
# Tuple inside a tuple - nested tuple and accessing its element 
nested_tuple = ((1, 2), (3, 4), (5, 6)) #Nested Tuples
print(nested_tuple[1][0]) # 3
 

# Tuple slicing 
my_tuple = (10, 20, 30, 40, 50) # Tuple Slicing
print(my_tuple[-1]) # 50, last element 
print(my_tuple[1:4]) # (20, 30, 40) -> index 1 to 3 
print(my_tuple[1:-1]) # (20, 30, 40) -> index 1 to one element before last 
 

# count and index of element 
my_tuple = (10, 20, 30, 40, 10)
print(my_tuple.count(10)) # 2 - counts occurances of an element 
print(my_tuple.index(30)) # 2 - returns index of an element 
 

# Tuple packing and unpacking 
a, b, c = 10, 20, 30 # Tuple Packing and Unpacking
my_tuple = a, b, c
x, y, z = my_tuple
print(x + y + z) # 60 -> addition of values 


# function to swap the first and last element of a tuple 
def swap_elements(input_tuple): 
    return input_tuple[-1], *input_tuple[1:-1], input_tuple[0]

print(swap_elements((1, 2, 3, 4)))          # Output: (4, 2, 3, 1)
print(swap_elements((7, 14, 21, 28)))      # Output: (28, 14, 21, 7)
print(swap_elements(('apple', 'banana', 'cherry'))) # Output: ('cherry', 'banana', 'apple')
print(swap_elements((5, 10)))               # Output: (10, 5)


# Count the occurance of element within a tuple 
def count_occurrences(input_tuple, target):
    return input_tuple.count(target)
    
print(count_occurrences((1, 2, 2, 3, 2), 2))  # Expected Output: 3


# Function that processes a tuple containing student details and returns a formatted summary string describing the student.
def student_summary(student): 
    name, age, grade = student
    return (f"Student Name: {name}, Age: {age}, Grade: {grade}")
    
print(student_summary(('Alice', 20, 89.5)))  # Expected Output: "Student Name: Alice, Age: 20, Grade: 89.5"
print(student_summary(('Bob', 22, 75.8)))  # Expected Output: "Student Name: Bob, Age: 22, Grade: 75.8"
print(student_summary(('Charlie', 19, 92.0)))  # Expected Output: "Student Name: Charlie, Age: 19, Grade: 92.0"


# Function that computes the Euclidean distance between two given points, which are represented as tuples.
import math
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

print(calculate_distance((1.0, 2.0), (4.0, 6.0)))  # Expected Output: 5.0
print(calculate_distance((0.0, 0.0), (0.0, 5.0)))  # Expected Output: 5.0

