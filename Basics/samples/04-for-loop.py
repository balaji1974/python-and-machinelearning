


for i in range(1,10) : 
	print(i) 
	print("hello")

print("We are outside the block now")


# Multiplication tabbles in python
for i in range(1,11) :
	print(f"7 * {i} = {i*7}")

# Range function with increment step of 2
sum=0
for i in range(1,11,2) :
	sum=sum+i # will sum all odd numbers
print(sum)

# Print the sum of squares of the first 10 numbers
sum_of_squares=0
for i in range(1,11) :
    sum_of_squares=sum_of_squares+(i**2)
print(sum_of_squares)


# Print the sum of squares of the first 10 numbers
sum_of_cubes=0
for i in range(1,11) :
    sum_of_cubes=sum_of_cubes+(i**3)
print(sum_of_cubes)

# Print the factorial of 6
factorial=1
for i in range(1,7) :
    factorial=factorial*i
print(factorial)

