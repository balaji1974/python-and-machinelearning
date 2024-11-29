# creating an empty class
class Country: 
	pass


# creating an object of a class and assigning attributes to it.
class Country: # Class
    pass
india = Country() # creating object - instance of a class
usa = Country() # creating object - instance of a class
netherlands = Country() # creating object - instance of a class

india.name = 'India' # adding attributes to the object 
india.capital = 'New Delhi' # adding attributes to the object 
  
usa.name = 'USA' # adding attributes to the object 
usa.capital = 'Washington' # adding attributes to the object 
 
netherlands.name = 'Netherlands' # adding attributes to the object 
netherlands.capital = 'Amsterdam' # adding attributes to the object 

print(india.name)  # Output: 'India' - accessing the object attribute
print(usa.capital)  # Output: 'Washington' - accessing the object attribute


# Few more examples of creating class, objects and attributes of the object and printing them
class MotorBike:
    pass
honda = MotorBike()
ducati = MotorBike()
print(honda)  # Output will be a memory location
print(ducati)  # Output will be a memory location
honda.speed = 50
ducati.speed = 250
print(honda.speed)  # Output: 50
print(ducati.speed)  # Output: 250

# another class
class Book:
    pass
first_book = Book()
second_book = Book()
third_book = Book()
first_book.name = 'The Art of Computer Programming'
second_book.name = 'Learning Python'
third_book.name = 'Learning Restful Services In 50 Steps'
print(first_book.name)
print(second_book.name)
print(third_book.name)


# Creating constructor and setting attributes
class MotorBike:
    def __init__(self, speed): # constructor 
        self.speed = speed # setting value for attributes
 
honda = MotorBike(50) # sending value to be added to attributes of object 
ducati = MotorBike(250) # sending value to be added to attributes of object 
 
print(honda.speed)  # Output: 50
print(ducati.speed)  # Output: 250

# constructor with default values
class Planet:
    def __init__(self, name="Earth"):
        self.name = name
        self.speed = 10
        self.distance_from_sun = 10000
 
planet = Planet()
 
print(planet.speed)  # Output: 10
print(planet.distance_from_sun)  # Output: 10000

# code to convert inches to feet - 1 feet = 12 inches
class Dimension:
    #TODO: 
    def __init__(self, inches): #convert given inches into feet and inches
        if(inches<0) :
            self.feet=-1
            self.inches=-1
        else :
            self.feet=inches // 12
            self.inches=inches % 12 
    

# Examples
dim = Dimension(25)
print(dim.feet)    # Output: 2
print(dim.inches)  # Output: 1


# Example class
# 1. An object honda is created with a speed of 50.
# 2. An object ducati is created with a speed of 250.
# 3. The increase_speed method is called on honda, increasing its speed by 150. The new speed becomes 50 + 150 = 200.
# 4. The increase_speed method is called on ducati, increasing its speed by 25. The new speed becomes 250 + 25 = 275.
# 5. The decrease_speed method is called on honda, decreasing its speed by 50. The new speed becomes 200 - 50 = 150.
# 6. The decrease_speed method is called on ducati, decreasing its speed by 25. The new speed becomes 275 - 25 = 250.

class MotorBike:
    def __init__(self, speed):
        self.speed = speed  # State
 
    def increase_speed(self, how_much):
        self.speed += how_much  # Behavior
 
    def decrease_speed(self, how_much):
        self.speed -= how_much  # Behavior
 
honda = MotorBike(50)
ducati = MotorBike(250)
 
honda.increase_speed(150)
ducati.increase_speed(25)
 
honda.decrease_speed(50)
ducati.decrease_speed(25)
 
print(honda.speed)  # Output: 150
print(ducati.speed)  # Output: 250


# extend the Book class to add a copies attribute and methods to increase or decrease the number of copies. 
class Book:
    def __init__(self, name, copies=0):
        self.name = name
        self.copies = copies
 
    def increase_copies(self, how_much):
        self.copies += how_much
 
    def decrease_copies(self, how_much):
        self.copies -= how_much
 
# Instances
the_art_of_computer_programming = Book('The Art of Computer Programming')
learning_python = Book('Learning Python in 100 Steps', 100)
learning_restful_services = Book('Learning RestFul Service in 50 Steps')
 
# Using the methods
learning_python.increase_copies(25)
learning_python.decrease_copies(10)
 
# Directly changing the `copies` attribute
learning_python.copies = 50
 
# Output
print(the_art_of_computer_programming.name)
print(learning_python.name)
print(learning_restful_services.name)
print(learning_python.copies)


# Chaining methods within a class
class Planet:
    def rotate(self):
        print("rotate")
 
    def revolve(self):
        print("revolve")
        
    def rotate_and_revolve(self):
        self.rotate()
        self.revolve()
 
# Create an instance and call its methods
earth = Planet()
earth.rotate_and_revolve()  # Output will be "rotate" followed by "revolve"


# Calculate area and perimeter for a square
class Square:
    def __init__(self, side): 
        self.side=side

    def calculate_area(self):
        if(self.side <= 0):
            return -1
        else:
            return self.side*self.side 
        
    def calculate_perimeter(self):
        if(self.side <= 0):
            return -1
        else: 
            return self.side*4

        
# Example 1
square = Square(5)
print(square.calculate_area())       # Output: 25
print(square.calculate_perimeter())  # Output: 20
 
# Example 2
square_with_zero_side = Square(0)
print(square_with_zero_side.calculate_area())       # Output: -1
print(square_with_zero_side.calculate_perimeter())  # Output: -1
 
# Example 3
square_with_non_positive_side = Square(-5)
print(square_with_non_positive_side.calculate_area())       # Output: -1
print(square_with_non_positive_side.calculate_perimeter())  # Output: -1


# This method adjusts the current position of the point in the 2-dimensional space. The parameters dx and dy represent the changes in the x-coordinate and y-coordinate respectively.
# This method calculates and returns the Euclidean distance between the current point and another point other.
import math

class Point:

    def __init__(self, x, y): 
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def distance_to(self, other):
        return math.sqrt((other.x-self.x)**2 + (other.y-self.y)**2)

movement=Point(3,4)
movement.move(1,2)
print(movement.x)
print(movement.y)

initial=Point(3,4)
another=Point(6,8)
print(initial.distance_to(another))


# Function as objects in Python
def do_something():
    print("something")
 
# Functions as objects
print(do_something)  # Output: <function do_something at some_memory_address>
 
# Assigning functions to variables
test = do_something
test()  # Output: something


# Object samples

class Car:
    pass
my_car = Car()
print(type(my_car))
 
class MyClass:
    def empty_method(self):
        pass
my_instance = MyClass()
my_instance.empty_method()
 
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
 
    def area(self):
        return self.length * self.width
 
my_rect = Rectangle(5, 3)
print(my_rect.area())
 
 
 
class BankAccount:
    def __init__(self):
        self.balance = 0
 
    def deposit(self, amount):
        self.balance += amount
 
    def withdraw(self, amount):
        self.balance -= amount
 
    def get_balance(self):
        return self.balance
 
my_account = BankAccount()
my_account.deposit(100)
my_account.withdraw(30)
print(my_account.get_balance())

# Assign RGB color values and invert colors
class RGBColor:
    def __init__(self, red, green, blue):
        self.red=red
        self.green=green
        self.blue=blue

    def invert(self):
        self.red =255-self.red
        self.green = 255-self.green
        self.blue = 255-self.blue

color = RGBColor(255, 0, 0)
color.invert()
print(color.red)   # Prints: 0
print(color.green) # Prints: 255
print(color.blue)  # Prints: 255



