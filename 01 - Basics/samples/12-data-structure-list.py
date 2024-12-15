# List data structure
marks = [45, 54, 80]
print(sum(marks)) # Outputs: 179
print(sum(marks)/len(marks)) # Outputs: 59.666666666666664

marks.append(43)
print(sum(marks)/len(marks)) # Outputs: 55.5
print(type(marks)) # Outputs: <class 'list'>


# Functions in a list 
marks = [23, 56, 67]
print(sum(marks))  # Outputs: 146 - sum of the list elements
print(max(marks))  # Outputs: 67 - max of all the elements in the list
print(min(marks))  # Outputs: 23 - min of all the elements in the list
print(len(marks))  # Outputs: 3 - size of the list 


# Adding Elements :
marks.append(76)
print(marks)  # Outputs: [23, 56, 67, 76]

# Inserting in a specific position of the list: 
marks.insert(2, 60)
print(marks)  # Outputs: [23, 56, 60, 67, 76]

#Remove a value from the  list:
marks.remove(60)
print(marks)  # Outputs: [23, 56, 67, 76]


#Searching and Checking Existence :
print(55 in marks)  # Outputs: False
print(56 in marks)  # Outputs: True
print(marks.index(67))  # Outputs: 2
print(marks)  # Outputs: [23, 56, 67, 76]


# If you try to find the index of a value that does not exist in the list, you will get an error:
# Uncomment below line and test
# print(marks.index(69)) # this will throw an error


#Iterating Through a List :
for mark in marks:
    print(mark)

# to create a function that checks whether there is any number greater than a given value in a list of numbers.
def has_greater_element(numbers, value):
    for number in numbers: 
        if(number>value):
            return True
    return False

numbers=[10,20,30]    
print(has_greater_element(numbers, 15)) # True

numbers=[5,7,8]    
print(has_greater_element(numbers, 10)) # False 

numbers=[]    
print(has_greater_element(numbers, 5)) # False 

#Delete an element from the list: 
#del(mark[3])
#print(mark)

#Add multiple values to an already existing list
marks.extend([3,7,8])
print(marks)

#Another way to extend
marks += [11,12]
print(marks)


#Append to an existing list
marks.append(99)
print(marks)


# Sort and reverse
numbers = [4, 2, 9, 1]
numbers.sort() # sort 
print(numbers) 
numbers.reverse() # reverse 
print(numbers) 


# determine if the sum of the elements in two given lists is equal
def are_sums_equal(list1, list2):
    if not list1 or not list2:
        return False
    # return sum(list1)==sum(list2) # This is a given contraint and must not use this in this example
    sum1, sum2=0, 0
    for l1 in list1: 
        sum1+=l1
    for l2 in list2:
        sum2+=l2
    return sum1==sum2
        
print(are_sums_equal([10, 20, 30], [15, 25, 20]))  # Output: True
print(are_sums_equal([5, 10, 15], [5, 10, 14]))  # Output: False
print(are_sums_equal([1, 2, 3, 4], [4, 3, 2, 1]))  # Output: True
print(are_sums_equal([], [4, 3, -7]))  # Output: False
print(are_sums_equal([1, 2], []))  # Output: False



#Using the reverse() method, you can reverse a list in-place. This action directly modifies the original list.
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
numbers.reverse()
print(numbers)  # ['Nine', 'Eight', 'Seven', 'Six', 'Five', 'Four', 'Three', 'Two', 'One', 'Zero']


# The reversed() function allows you to access elements in reverse order without changing the original list.
for number in reversed(numbers):
    print(number)
# This prints the elements in reverse order.


# The sort() method provides in-place sorting of the list, thereby altering the original list.
numbers.sort()
print(numbers)  # ['Eight', 'Five', 'Four', 'Nine', 'One', 'Seven', 'Six', 'Three', 'Two', 'Zero']


# To gain access to elements in a sorted order without affecting the original list, use the sorted() function.
for number in sorted(numbers):
    print(number)
# This prints the elements in sorted order.


# For sorting based on custom criteria, such as the string length, you can pass the len function as the key argument.
for number in sorted(numbers, key=len):
    print(number)
# This prints elements ordered by their increasing length.


# If you wish to sort the elements in the reverse order of the sorting criterion, you can achieve this by passing reverse=True.
for number in sorted(numbers, key=len, reverse=True):
    print(number)
# This prints the elements in descending order of their length.


# to determine whether a provided list of integers is sorted in ascending order.
def is_list_sorted(list): 
    if not list:
        return True
    for i in range(len(list)-1): # range function will start from 0 
        if list[i]>list[i+1]:
            return False
    return True

print(is_list_sorted([10, 20, 30]))   # Output: True
print(is_list_sorted([10, 30, 20]))   # Output: False
print(is_list_sorted([30, 20, 10]))   # Output: False
print(is_list_sorted([]))             # Output: True


# Invert a given list
def reverse_list(list): 
    start=0
    end=len(list)-1
    
    while (start<end):
        list[start], list[end] = list[end], list[start]
        start+=1 
        end-=1 
    return list 

# Testing the function with examples
print(reverse_list([10, 20, 30]))      # Output: [30, 20, 10]
print(reverse_list([5, 15, 25, 35]))   # Output: [35, 25, 15, 5]
print(reverse_list([1]))               # Output: [1]


# Function to determine all the factors of a given integer.
def find_factors(number): 
    factors=[]
    for i in range(1, number+1):
        if number%i==0 :
            factors.append(i)
    return factors

print(find_factors(12))  # Output: [1, 2, 3, 4, 6, 12]
print(find_factors(15))  # Output: [1, 3, 5, 15]
print(find_factors(7))   # Output: [1, 7]
    

# Function to determine all the multiples of a given number that are less than a provided limit
def find_multiples(number, limit):
    multiples=[]
    for i in range(number, limit, number): 
        multiples.append(i) 
    return multiples 

# Testing the function
print(find_multiples(3, 10))  # Expected Output: [3, 6, 9]
print(find_multiples(5, 22))  # Expected Output: [5, 10, 15, 20]
print(find_multiples(7, 50))  # Expected Output: [7, 14, 21, 28, 35, 42, 49]




