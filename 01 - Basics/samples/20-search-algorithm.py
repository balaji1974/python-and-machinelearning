# Linear search - O(n)
def linear_search(lst, target):
    for index, value in enumerate(lst):  # enumeration returns both index and value 
        if value == target:
            return index
    return None
 
result = linear_search([5, 3, 8, 1, 4, 6, 7, 2, 9], 4)
print("Found at index:", result)

result = linear_search(['apple', 'banana', 'cherry', 'date', 'fig', 'grape'], 'date')
print("Found at index:", result)

result = linear_search(['apple', 'banana', 'cherry', 'date', 'fig', 'grape'], 'mango')
print("Found at index:", result)


# Binary Search - O(log n) - Iterative approach
def binary_search(lst, target):
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return None
 
result = binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
print("Found at index:", result)

result = binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9], 10)
print("Found at index:", result)

result = binary_search([], 4)
print("Found at index:", result)


# Binary Search - O(log n) - Recursive approach
def binary_search_recursive(lst, target, low=0, high=None):
    if high is None:
        high = len(lst) - 1
    if low > high:
        return None
    mid = (low + high) // 2
    if lst[mid] == target:
        return mid
    elif lst[mid] < target:
        return binary_search_recursive(lst, target, mid + 1, high)
    else:
        return binary_search_recursive(lst, target, low, mid - 1)
 
# Example usage
result = binary_search_recursive([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
print("Found at index:", result)

