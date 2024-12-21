# Basic for loop
for i in range(1, 11):
    print(i)


# Prime number 
def is_prime(num) :
    if num<2:
        return False
    for i in range(2, num):
        if num%i==0 :
            return False
    return True

print(is_prime(5))  # Output: True

# sum of squares of number upto a limit
def sum_of_squares_upto_limit(limit) :
    i=1
    sum=0
    while(i<limit) :
        sum+= i*i
        i+= 1
    return sum
        
print(sum_of_squares_upto_limit(30))  # Expected Output: 8555

# Calculate the Sum of Cubes Up to a Limit
def sum_of_cubes_upto_limit(limit) :
    i=1 
    sum=0
    
    while (i**3<=limit) :
        sum+=i**3
        i+=1 
    return sum
print(sum_of_cubes_upto_limit(30))  # Output: 36


#Find the number of digits in a given number
def get_number_of_digits(number) :
    if(number<0):
        return -1
    elif(number==0):
        return 1 
    number_of_digits=0
    while(number>0):
        number=number//10
        number_of_digits+=1
    return number_of_digits

print(get_number_of_digits(123))  # Output: 3
print(get_number_of_digits(9087))  # Output: 4
print(get_number_of_digits(6))  # Output: 1


# calculate the first fibonacci number that exceeds a given threshold
def next_fibonacci(threshold):
    a, b=0, 1
    while(True) :
        sum=a+b
        if(sum>threshold) :
            break
        a, b=b, sum 
    return sum

print(next_fibonacci(20))  # Expected Output: 21


