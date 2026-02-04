
# def is used for creating a function
def hello_world() :
	print("Hello World")

#invoke the function 
hello_world()


# function with parameter // ALSO FUNCTION OVERLOADING 
def hello_world(cnt) :
	for i in range(cnt):
		print("Hello World!")


# function with parameter // ALSO FUNCTION OVERRIDING - REPLACE a previous implementation  
def hello_world(abc) :
	for i in range(abc):
		print("Hello World!!!!")


hello_world(5)

# function with multiple parameters
def hello_world(value, times) :
	for i in range(times):
		print(value)


hello_world("Hello World!!", 5)

#function with return values
def calculate_complex_calc(a,b) :
	return ((a**2)+(b**3))

print(calculate_complex_calc(7,5))


# function to calculate cube of a number 
def cube_of_number(v) :
    return v**3
    
print(cube_of_number(9))


# function to calcule the product of four numbers
def product_of_four_numbers(a,b,c,d) :
    return a*b*c*d

print(product_of_four_numbers(5,10,2,3))

#function to print the average of 5 numbers
def average_of_five_numbers(a,b,c,d,e) :
    return ((a+b+c+d+e)/5)
    
print(average_of_five_numbers(6,7,9,12,15))


# function to calcule the third angle  of a triangle
def calculate_third_angle(a,b) :
    return (180-a-b)
    
print(calculate_third_angle(50, 60))


# function to calculate the sum of squares of the first n even numbers
def sum_of_squares(n):
    sum=0
    for i in range(2, n*2+1, 2): 
        sum += (i**2)
    return sum    
print(sum_of_squares(5))


# function with default values
def hello_world(value, times=5) :
	for i in range(times):
		print(value)

hello_world("Hello World!!",10)


# function with pass will leave code execution without errors
def hello_world(value, times=5) : 
	for i in range(times): pass

hello_world("Hello WorldXXX",10)

