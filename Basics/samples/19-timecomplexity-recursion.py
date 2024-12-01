# Constant time complexity means that the running time of the algorithm does not depend on the size of the input.
# Accessing an element from an array - O(1)

def get_first_element(arr):
    return arr[0]
print(get_first_element([1, 2, 3]))  # Output: 1


# Linear time complexity means the running time grows linearly with the size of the input.
# Finding the maximum element in an array - O(n)

def find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val
print(find_max([1, 2, 3]))  # Output: 3


# Quadratic time complexity means the running time grows with the square of the size of the input.
# A nested loop iterating over an array - O(n^2)

def print_pairs(arr):
    for i in arr:
        for j in arr:
            print(i, j)
# Try it out
print_pairs([1, 2, 3])
# Output: 
# 1 1
# 1 2
# 1 3
# 2 1
# 2 2
# 2 3
# 3 1
# 3 2
# 3 3


# Recursion - Factorial Example
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)
# Example usage
result = factorial(5)
print("Factorial of 5 is:", result)


# Recursion - Sum of a list Example
def sum_of_list(lst, n):
    print(f"Calculating sum of first {n} elements of {lst}")
    if n == 0:
        print("Reached base case: sum is 0")
        return 0
    result = lst[n-1] + sum_of_list(lst, n-1)
    print(f"Sum of first {n} elements is {result}")
    return result
 
# Running the step-by-step example
result = sum_of_list([1, 2, 3], 3)
print(f"Final Output: Sum of list is {result}")

