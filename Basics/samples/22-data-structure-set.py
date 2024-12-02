# # Set creation
numbers = [1,2,3,2,1]
print(numbers)  # Output: [1, 2, 3, 2, 1]
numbers_set = set(numbers)
print(numbers_set)  # Output: {1, 2, 3}

# Adding to a set 
numbers_set.add(3)
print(numbers_set)  # Output: {1, 2, 3}
numbers_set.add(4)
print(numbers_set)  # Output: {1, 2, 3, 4}
numbers_set.add(0)
print(numbers_set)  # Output: {0, 1, 2, 3, 4}

# Removing from a set 
numbers_set.remove(0)
print(numbers_set)  # Output: {1, 2, 3, 4}

# Set does not support accessing from an index 
# numbers_set[0]  # This will raise TypeError: 'set' object does not support indexing

# Check if an element exist in a set or not 
print(1 in numbers_set)  # Output: True
print(5 in numbers_set)  # Output: False

# Aggregate operations on a set 
print(min(numbers_set))  # Output: 1
print(max(numbers_set))  # Output: 4
print(sum(numbers_set))  # Output: 10
print(len(numbers_set))  # Output: 4


# Union operation of two sets
numbers_1_to_5_set = set(range(1,6))
print(numbers_1_to_5_set)  # Output: {1, 2, 3, 4, 5}
numbers_4_to_10_set = set(range(4,11))
print(numbers_4_to_10_set)  # Output: {4, 5, 6, 7, 8, 9, 10}
print(numbers_1_to_5_set | numbers_4_to_10_set)  # Output: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

# Intersection operation on two sets
print(numbers_1_to_5_set & numbers_4_to_10_set)  # Output: {4, 5}

# subtraction operation of two sets
print(numbers_1_to_5_set - numbers_4_to_10_set)   # Output: {1, 2, 3}
print(numbers_4_to_10_set - numbers_1_to_5_set)  # Output: {6, 7, 8, 9, 10}

# use the * operator to unpack elements from a list and pass them as arguments to a function.
def print_values(num1, num2, num3):
    print(num1)
    print(num2)
    print(num3)
 
# Unpacking elements from a list  
numbers = [10, 20, 30]
print_values(*numbers)

# Unpacking elements from a set 
scores = {85, 90, 75}
print_values(*scores)


# Common Samples 
# Union operation (s1.union(s2) or s1 | s2) combines elements from two sets.
s1 = {1, 2, 3}
s2 = {3, 4, 5}
result = s1.union(s2) # Same as s1 | s2
print(result)
 
 
# Use set.union()
list_of_sets_1 = [set([1, 2]), set([3, 4]), set([5, 6])]
list_of_sets_2 = [set([5, 6]), set([7, 8])]
 
union_result = set.union(*list_of_sets_1, *list_of_sets_2)
#similar functions exist for union and difference
print(union_result)


# Difference operation (s1.difference(s2) or s1 - s2) removes common elements from the first set. 
s1 = {1, 2, 3}
s2 = {3, 4, 5}
result = s1.difference(s2) # Same as s1 - s2
print(result)
 
# Set comprehension ({x**2 for x in numbers if x % 2 == 0}) creates a set of squared numbers for even elements. 
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = {x**2 for x in numbers if x % 2 == 0}
print(result)
 
num = 5
result = {x for x in range(num, 101, num)}
print(result)
 
# Using set() efficiently removes duplicates from a list, as sets only store unique elements. 
my_list = [1, 2, 3, 2, 1, 4, 5, 3]
result = set(my_list)
print(result)
 
# Combining unique characters from strings with set(string_1) | set(string_2).
string_1 = "hello"
string_2 = "world"
 
result = set(string_1) | set(string_2)
print(result)

# Finding common elements in a list of sets with set.intersection(*list_of_sets). 
list_of_sets = [{1, 2, 3}, {3, 4, 5}, {5, 6, 7}]
result = set.intersection(*list_of_sets)
print(result)


# Find the intersection of the multiples of two numbers within a given range. 
def find_intersection(num1, num2, limit): 
    if num1==0 or num2==0:
        return set() # return empty set 
    multiples_num1={i for i in range(num1, limit + 1, num1)}
    multiples_num2={i for i in range(num2, limit + 1, num2)}
    return multiples_num1 & multiples_num2

print(find_intersection(4, 6, 30))
print(find_intersection(3, 5, 20))


# Function that identifies and returns the colors that are unique to two different sets of color palettes.
def unique_colors(palette1, palette2):
    if not palette1:
        return palette2
    if not palette2:
        return palette1
    return (palette1 | palette2) - (palette1 & palette2)

# Examples
print(unique_colors({"red", "blue"}, {"blue", "green"}))         
# Output: {"red", "green"}
 
print(unique_colors({"purple", "yellow"}, {"yellow", "pink"}))   
# Output: {"purple", "pink"}
 
print(unique_colors({"orange", "cyan"}, {"cyan", "magenta"}))    
# Output: {"orange", "magenta"}


# Merge number of shopping list into one 
def merge_shopping_lists(*lists):
    if(not lists):
        return set()
    return set.union(*lists)

list1 = {"apples", "bananas", "cherries"}
list2 = {"bananas", "dates", "eggs"}
list3 = {"cherries", "dates", "figs"}
 
print(merge_shopping_lists(list1, list2, list3))
# Output: {"apples", "bananas", "cherries", "dates", "eggs", "figs"}
 
list4 = {"bread", "milk"}
list5 = {"milk", "eggs", "juice"}
print(merge_shopping_lists(list4, list5))
# Output: {"bread", "milk", "eggs", "juice"}





