# Sample class, state (member variables) and behaviour (methods)

# Define the Fan class
class Fan:
    # Constructor to initialize the fan
    def __init__(self, make, radius, color):
        self.make = make
        self.radius = radius
        self.color = color
        self.speed = 0  # Initial speed is 0
        self.is_on = False  # Fan is initially off
 
    # Method to represent the Fan object as a string
    def __repr__(self):
        return repr((self.make, self.radius, self.color, self.speed, self.is_on))
 
    # Method to switch the fan on
    def switch_on(self):
        self.is_on = True
        self.speed = 3  # Set initial speed to 3
 
    # Method to switch the fan off
    def switch_off(self):
        self.is_on = False
        self.speed = 0  # Reset speed to 0
 
    # Method to increase the fan speed
    def increase_speed(self):
        if self.is_on and self.speed < 5:  # Max speed is 5
            self.speed += 1
 
    # Method to decrease the fan speed
    def decrease_speed(self):
        if self.is_on and self.speed > 0:  # Min speed is 0
            self.speed -= 1
 
# Create a Fan object and test the methods
fan = Fan('Manufacturer 1', 5, 'Green')
print(fan)  # Output: ('Manufacturer 1', 5, 'Green', 0, False)
 
fan.switch_on()
print(fan)  # Output: ('Manufacturer 1', 5, 'Green', 3, True)
 
fan.increase_speed()
print(fan)  # Output: ('Manufacturer 1', 5, 'Green', 4, True)
 
fan.switch_off()
print(fan)  # Output: ('Manufacturer 1', 5, 'Green', 0, False)

# *************************************** #

# Object composition - Adding object within another object 

# Define the Book class
class Book(object):
    # Constructor to initialize the Book
    def __init__(self, id, name, author):
        self.id = id
        self.name = name
        self.author = author
        self.reviews = []  # Empty list for storing reviews
 
    # Representation method for printing the Book object
    def __repr__(self):
        return repr((self.id, self.name, self.author, self.reviews))

    # Add the add_review method to the Book class
    def add_review(self, review):
        self.reviews.append(review)

# Create a Book instance
book = Book(123, 'Object Oriented Programming with Python', 'Ranga')
print(book)  # Output: (123, 'Object Oriented Programming with Python', 'Ranga', [])

# Define the Review class
class Review:
    # Constructor to initialize the Review
    def __init__(self, id, description, rating):
        self.id = id
        self.description = description
        self.rating = rating
 
    # Representation method for printing the Review object
    def __repr__(self):
        return repr((self.id, self.description, self.rating))

# Create a Review instance
review = Review(10, 'Great Book', 5)
print(review)  # Output: (10, 'Great Book', 5)


# Add reviews to the book
book.add_review(Review(10, 'Great Book', 5))
book.add_review(Review(101, 'Awesome', 5))
print(book)  # Output: (123, 'Object Oriented Programming with Python', 'Ranga', [(10, 'Great Book', 5), (101, 'Awesome', 5)])

# *************************************** #

# Inheritance 

# Define Animal class with a bark method
class Animal:
    def bark(self):
        print("bark")
 
# Create an instance of Animal and call its bark method
animal = Animal()
animal.bark()  # Output: bark

# Define Pet class with groom methods
# Pet class to inherit from Animal
class Pet(Animal):
    def groom(self):
        print("groom")
 
# Create an instance of Pet
dog = Pet()
 
# Check if inheritance works
dog.bark()  # Output: bark
dog.groom()  # Output: groom

# *************************************** #

# every class implicitly inherits from the built-in object class 

# Define a Book class with a custom __repr__ method
class Book():
    def __repr__(self):
        return repr('new book')
 
# Create an instance of Book and print it
book = Book()
print(book)  # Output: 'new book'


# *************************************** #

# Multiple inheritance in Python
class LandAnimal:
    def __init__(self):
        super().__init__()
        self.walking_speed = 5
        
    def increase_walking_speed(self, how_much):
        self.walking_speed += how_much
 
class WaterAnimal:
    def __init__(self):
        super().__init__()
        self.swimming_speed = 10
        
    def increase_swimming_speed(self, how_much):
        self.swimming_speed += how_much
 
class Amphibian(WaterAnimal, LandAnimal):
    def __init__(self):
        super().__init__()
        
amphibian = Amphibian()
amphibian.increase_swimming_speed(25)
amphibian.increase_walking_speed(50)
print(amphibian.swimming_speed)  # Output: 35
print(amphibian.walking_speed)  # Output: 55

# *************************************** #

# Another example for multiple inheritance

#TODO: Implement the `start_engine` method to return "Engine started".
class Engine:
    def start_engine(self):
        return "Engine started"

#TODO: Implement `number_of_wheels` method to return number of wheels - 4.
class Wheels:
    def number_of_wheels(self):
        return 4

#TODO: Make class inherit from Engine & Wheels
#TODO: Implement the `drive` method to return "Car is driving".
class Car(Engine, Wheels): 
    def drive(self): 
        return "Car is driving"

vehicle = Car()
 
# Test engine start
result_start = vehicle.start_engine()
print(result_start) # Output: "Engine started"
 
# Test drive
result_drive = vehicle.drive()
print(result_drive) # Output: "Car is driving"
 
# Test number of wheels
num_wheels = vehicle.number_of_wheels()  
print(num_wheels) # Output: 4

# *************************************** #

# Abstract Class
from abc import ABC, abstractmethod
 
class AbstractAnimal(ABC):
    @abstractmethod
    def bark(self): pass

# Implementing the abstract method in a subclass
class Dog(AbstractAnimal):
    def bark(self):
        print("Bow Bow")
 
Dog().bark()  # Output: "Bow Bow"

# *************************************** #

# Template method pattern using Abstract class

from abc import ABC, abstractmethod
 
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

class Recipe1(AbstractRecipe):
 
    def prepare(self):
        print('do the dishes')
        print('get raw materials')
 
    def recipe(self):
        print('execute the steps')
 
    def cleanup(self): pass
 
Recipe1().execute()

class MicrowaveRecipe(AbstractRecipe):
 
    def prepare(self):
        print('do the dishes')
        print('get raw materials')
        print('switch on microwave')
 
    def recipe(self):
        print('execute the steps')
 
    def cleanup(self):
        print('switch off microwave')
 
MicrowaveRecipe().execute()


# *************************************** #

# Polymorphism example 
class Shape:
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius * self.radius
 
class Square(Shape):
    def __init__(self, side):
        self.side = side
    
    def area(self):
        return self.side * self.side
 
class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return self.length * self.width

shapes = [Circle(5), Square(4), Rectangle(2, 5)]
 
for shape in shapes:
    print(f"The area of {shape} is {shape.area()}")


# *************************************** #

# Another polymorphism test
class Shape:
    def area(self):
        pass
    def perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius * self.radius
 
    def perimeter(self):
        return 2 * 3.14 * self.radius
        
class Square(Shape):
    def __init__(self, side):
        self.side = side
    
    def area(self):
        return self.side * self.side
 
class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return self.length * self.width
        
    def perimeter(self):
        return 2 * (self.length + self.width)

shapes = [Circle(5), Square(4), Rectangle(2, 5)]
 
for shape in shapes:
    print(f"The area of {shape} is {shape.area()}")

# *************************************** #

# 