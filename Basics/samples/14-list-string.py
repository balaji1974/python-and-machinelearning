# A function that takes a list of strings and a positive integer n as inputs. 
# The function should rotate the list n times to the right and return the rotated list.
def rotate_strings(strings, n):
    if not strings or n==0:
        return strings
    for cnt in range(n): 
        last=strings[-1] # this will get the last element of the list  
        for i in range(len(strings)-1,0, -1):
            strings[i]=strings[i-1]
        strings[0]=last
    return strings
    
    
print(rotate_strings(['a', 'b', 'c'], 2))      # Output: ['b', 'c', 'a']
print(rotate_strings(['apple', 'banana', 'cherry', 'date'], 1))   # Output: ['date', 'apple', 'banana', 'cherry']
print(rotate_strings(['hello', 'world'], 3))    # Output: ['world', 'hello']



# find the next character based on ASCII values for a given character.
def next_character(char):
    next_ascii_val = ord(char) + 1
    next_char = chr(next_ascii_val)
    return next_char
 
char = 'Y'
print(f"Given character: {char}")
print(f"The next character is: {next_character(char)}")


# Function takes a list of strings as input. 
#  The function should replace each string in the list with a new string where each character is replaced by the next character in the ASCII table. 
# Return the encoded list.
def encode_strings(strings): 
    if not strings:
        return strings
    
    encoded_list=[]
    for string in strings:
        encoded_string=''
        for char in string:
            if(char=='z'): 
                next_char='a'
            elif(char=='Z'):
                next_char='A'
            else: 
                next_char = chr(ord(char) + 1)
            encoded_string+=next_char
        encoded_list.append(encoded_string)
    return encoded_list

print(encode_strings(['abc', 'def']))  # Output: ['bcd', 'efg']
print(encode_strings(['hello', 'WORLD']))  # Output: ['ifmmp', 'XPSME']
print(encode_strings(['zoo']))  # Output: ['app']
print(encode_strings(['']))  # Output: ['']

# Merging two lists by alternating elements from each list. 
# If one list is longer than the other, the excess elements should be added to the end of the merged list.
def alternate_merge(list1, list2): 
    merged_list=[]
    long_list=max(len(list1), len(list2))
    for current_index in range(long_list): 
        if(current_index<len(list1)): 
            merged_list.append(list1[current_index])
        if(current_index<len(list2)): 
            merged_list.append(list2[current_index])
    return merged_list
    
print(alternate_merge(['a', 'b'], ['c', 'd', 'e']))  
# Output: ['a', 'c', 'b', 'd', 'e']
print(alternate_merge(['x', 'y', 'z'], ['1', '2']))  
# Output: ['x', '1', 'y', '2', 'z']
print(alternate_merge(['apple', 'banana'], ['grape', 'pineapple', 'blueberry']))  
# Output: ['apple', 'grape', 'banana', 'pineapple', 'blueberry']
print(alternate_merge([], ['a', 'b', 'c']))  
# Output: ['a', 'b', 'c']
print(alternate_merge(['short', 'words'], ['a_very_long_word', 'tiny']))  
# Output: ['short', 'a_very_long_word', 'words', 'tiny']


# Create country class
class Country:
 
    def __init__(self, name, population, area): # constructor 
        self.name = name
        self.population = population
        self.area = area
 
    def __repr__(self): # string representation of the class - equivalent to toString() in java
        return repr((self.name,self.population,self.area))

# Creating instance of the class
countries = [Country('India',1200,100),
             Country('China', 1400, 200),
             Country('USA', 120, 300)]
 
countries.append(Country('Russia',80,900))


# Sorting the class - based on population in reverse order 
from operator import attrgetter
countries.sort(key=attrgetter('population'), reverse=True)
print(countries)

# finding max, max of population and area 
print(max(countries, key=attrgetter('population')))
print(min(countries, key=attrgetter('population')))
print(min(countries, key=attrgetter('area')))
print(max(countries, key=attrgetter('area')))

# Variable arguments - Finding maximum 
def find_max(*args):
    if not args:
        return None
    max_value = args[0]
    for num in args:
        if num > max_value:
            max_value = num
    return max_value
result = find_max(3, 8, 2, 10, 5)
print(result)  # Outputs: 10

# Variable argument - combining strings
def combine_strings(*args):
    return ' '.join(args)
result = combine_strings('Hello', 'world', 'from', 'Python')
print(result)  # Outputs: Hello world from Python


# create a Python function named is_anagram that checks if two given strings are anagrams of each other
# Anagrams are words or phrases that are formed by rearranging the letters of another word or phrase, using all the original letters exactly once.
def is_anagram(string1, string2): 
    if len(string1) != len(string2):
        return False
    list1=[0] * 26
    list2=[0] * 26
    for c in string1: 
        list1[ord(c) - ord('a')]+=1 
    for c in string2: 
        list2[ord(c) - ord('a')]+=1
    return list1==list2
    
print(is_anagram("listen", "silent"))  # Output: True
print(is_anagram("hello", "hey"))      # Output: False
print(is_anagram("apple", "ppale"))    # Output: True





