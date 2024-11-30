# List slicing - examples

numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
print(numbers[3:8]) # ['Three', 'Four', 'Five', 'Six', 'Seven']
 
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
print(numbers[3:]) # ['Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
print(numbers[::4]) # ['Zero', 'Four', 'Eight']
 
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
print(numbers[2:9:2]) # ['Two', 'Four', 'Six', 'Eight']
 
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
print(numbers[::-2]) # ['Nine', 'Seven', 'Five', 'Three', 'One']
 
numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
numbers[1:6] = [1, 2, 3, 4, 5]
print(numbers) # ['Zero', 1, 2, 3, 4, 5, 'Six', 'Seven', 'Eight', 'Nine']


# fetch all the even-indexed elements originating from a list of integers.
def slice_alternate_elements(numbers):
    return numbers[::2]

print(slice_alternate_elements([10, 20, 30, 40, 50, 60]))  # Expected Output: [10, 30, 50]
print(slice_alternate_elements([]))  # Expected Output: []


# reverse every three elements in a given list of integers.
def reverse_chunks(numbers):
    for i in range(0, len(numbers), 3):
        numbers[i:i+3] = numbers[i:i+3][::-1]
    return numbers

print(reverse_chunks([1, 2, 3, 4, 5, 6, 7, 8, 9]))   # Output: [3, 2, 1, 6, 5, 4, 9, 8, 7]


# creating a function that sorts a list of strings based on their length in descending order and then removes the middle element(s). 
# If the list has an odd number of elements, you'll remove the exact middle element. 
# If the list has an even number of elements, you'll remove the two middle elements.
def reorder_and_eliminate_middle(words): 
    if(not words or len(words)<=2):
        return []
    sorted_words = sorted(words, key=len, reverse=True)
    
    middle = len(sorted_words)//2 
    if len(sorted_words) % 2 ==0:
        del sorted_words[middle-1:middle+1]
    else :
        del sorted_words[middle]
    return sorted_words   
        
print(reorder_and_eliminate_middle(["apple", "banana", "kiwi", "grapes", "mango"]))  
# Output: ["banana", "grapes", "mango", "kiwi"]
 
print(reorder_and_eliminate_middle(["apple", "banana", "kiwi", "grapes"]))  
# Output: ["banana", "kiwi"]
 
print(reorder_and_eliminate_middle([]))  
# Output: []
 
print(reorder_and_eliminate_middle(["apple"]))  
# Output: []

