# List comprehension - Example 
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
numbers_length_four = []
numbers_length_four = [number for number in numbers]
print(numbers_length_four) # ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
numbers_length_four = [len(number) for number in numbers]
print(numbers_length_four) # [4, 3, 3, 5, 4, 4, 3, 5, 5, 4]
numbers_length_four = [number for number in numbers if len(number) == 4]
print(numbers_length_four)  # ['Zero', 'Four', 'Five', 'Nine']

# Samples - List comprehension
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
result = [name for name in names if 'e' in name]
print(result)  # ['Alice', 'Charlie', 'Eve']

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = [num for num in numbers if num % 2 == 0]
print(result)  # [2, 4, 6, 8, 10]

sentence = "The quick brown fox jumps over the lazy dog"
result = [char for char in sentence if char.lower() in 'aeiou']
print(result)  # ['e', 'u', 'i', 'o', 'o', 'u', 'o', 'e', 'e', 'a', 'o']

original_list = [1, 2, 3, 4, 5]
result = [num**2 for num in original_list]
print(result)  # [1, 4, 9, 16, 25]

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = [num**2 for num in numbers if num % 2 != 0]
print(result)  # [1, 9, 25, 49, 81]

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = ['even' if num % 2 == 0 else 'odd' for num in numbers]
print(result)  # ['odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even']

sentence = "Hello World! How are you doing?"
result = [len(word) for word in sentence.split()] 
print(result) # [5, 6, 3, 3, 3, 6]


# slice and double 
# Given a list of integers and two indices, you'll extract a portion of the list using these indices and then double the numbers in that section. 
# Make sure to handle cases where the indices are outside the list's boundaries.
def slice_and_double(numbers, a, b):
    # handle edge cases 
    if(a<0):
        a=0
    # handle edge cases 
    if(b>len(numbers)):
        b=len(numbers)
    sliced_part=numbers[a:b]
    #print(sliced_part)
    doubled_slice=[ x*2 for x in sliced_part]
    #print(doubled_slice)
    numbers[a:b]=doubled_slice
    return numbers
    
print(slice_and_double([1, 2, 3, 4, 5], 1, 4))  # Output: [1, 4, 6, 8, 5]
print(slice_and_double([10, 11, 12], 0, 2))  # Output: [20, 22, 12]
print(slice_and_double([7, 8, 9, 10], 1, 5))  # Output: [7, 16, 18, 20]
print(slice_and_double([3, 6, 9, 12], -1, 3))  # Output: [6, 12, 18, 12]
print(slice_and_double([15, 20, 25, 30], 2, 10))  # Output: [15, 20, 50, 60]


# filter all the odd numbers from a given list of integers.
def extract_odd_numbers(values):
    result = [value for value in values if value % 2 == 1]
    return result

print(extract_odd_numbers([3, 6, 9, 1, 4, 15, 6, 3]))  # Output: [3, 9, 1, 15, 3]
print(extract_odd_numbers([10, 22, 33, 40, 55, 60]))  # Output: [33, 55]
print(extract_odd_numbers([]))  # Output: []



