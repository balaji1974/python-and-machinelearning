# Dictionary Init 
occurances = dict(a=5, b=6, c=8)
print(occurances)  # {'a': 5, 'b': 6, 'c': 8}
print(type(occurances))  # <class 'dict'>


# You can access and modify values using keys.
occurances['d'] = 15
print(occurances)  # {'a': 5, 'b': 6, 'c': 8, 'd': 15}
occurances['d'] = 10
print(occurances)  # {'a': 5, 'b': 6, 'c': 8, 'd': 10}
print(occurances['d'])  # 10


# Accessing a non-existent key raises a KeyError.
# print(occurances['e'])  # Uncommenting this will raise KeyError: 'e'
# To avoid this, use the get() method, providing a default value if needed.
print(occurances.get('d'))  # 10
print(occurances.get('e'))  # None
print(occurances.get('e', 10))  # 10


# Dictionary methods
print(occurances.keys())  # dict_keys(['a', 'b', 'c', 'd'])
print(occurances.values())  # dict_values([5, 6, 8, 10])
print(occurances.items())  # dict_items([('a', 5), ('b', 6), ('c', 8), ('d', 10)])


# You can iterate through key-value pairs in a dictionary.
for (key, value) in occurances.items():
    print(f"{key} {value}")  # Expected output:
                             # a 5
                             # b 6
                             # c 8
                             # d 10


# You can delete a specific key-value pair using the del keyword.
occurances['a'] = 0
del occurances['a']
print(occurances)  # {'b': 6, 'c': 8, 'd': 10}


# Change value and print 
user_info = {'name': 'John', 'age': 30, 'city': 'New York'}
user_info['age'] = 31
print(user_info) # {'name': 'John', 'age': 31, 'city': 'New York'}


# character count
text = "hello"
char_count = {}
for char in text:
    if char in char_count:
        char_count[char] += 1
    else:
        char_count[char] = 1
print(char_count) # {'h': 1, 'e': 1, 'l': 2, 'o': 1}
 

# Finding common keys 
dict_1 = {'a': 1, 'b': 2, 'c': 3}
dict_2 = {'c': 3, 'd': 4, 'e': 5}
common_keys = set(dict_1.keys()) & set(dict_2.keys())
print(common_keys) # {'c'}
 

# Combining 2 sets to form a dictionary using 'zip' 
keys = ['a', 'b', 'c']
values = [1, 2, 3]
result = {k: v for k, v in zip(keys, values)}
print(result) # {'a': 1, 'b': 2, 'c': 3}
 

# Dictionary within a dictionary and accessing its elements 
users = {
    'user1': {'name': 'John', 'age': 30},
    'user2': {'name': 'Jane', 'age': 25}
}
print(users['user1']['name']) # John
 

# Creating dictionary with squares as values of keys 
squares = {x: x**2 for x in range(1, 6)}
print(squares) # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
 

# Finding common values in dictionary containing sets  
students = {"Alice": {"Math", "English"}, "Bob": {"Math", "Science"}}
common_subjects = set.intersection(*students.values())
print(common_subjects) # {'Math'}


# Count the character occurances of a given string 
def count_characters(str):
    char_occurances ={}
    for char in str:
        char_occurances[char] = char_occurances.get(char, 0) + 1
    return char_occurances

print(count_characters("apple")) # {'a': 1, 'p': 2, 'l': 1, 'e': 1}
print(count_characters("banana")) # {'b': 1, 'a': 3, 'n': 2}
print(count_characters("This is an awesome occasion.")) # {'T': 1, 'h': 1, 'i': 3, 's': 4, ' ': 4, 'a': 3, 'n': 2, 'w': 1, 'e': 2, 'o': 3, 'm': 1, 'c': 2, '.': 1}   


# Count word occurances of a given string 
def count_words(str):
    word_occurrences={}
    words=str.split()
    for word in words:
        word_occurrences[word] = word_occurrences.get(word, 0) + 1
    return word_occurrences
    
print(count_words("This is an example.")) # {'This': 1, 'is': 1, 'an': 1, 'example.': 1}
print(count_words("Hello world! Hello everyone.")) # {'Hello': 2, 'world!': 1, 'everyone.': 1}
print(count_words("This is an awesome occasion. This has never happened before.")) # {'This': 2, 'is': 1, 'an': 1, 'awesome': 1, 'occasion.': 1, 'has': 1, 'never': 1, 'happened': 1, 'before.': 1}


# create a function that returns a dictionary containing the squares of the first n natural numbers.
def squares_map(n):
    return {x: x**2 for x in range(1, n+1)}

print(squares_map(10))  # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100}


# function that identifies the subjects that are studied by all students across three different grades.
def common_subjects(grade1, grade2, grade3):
    if(not grade1 or not grade2 or not grade3):
        return set()
    return set.intersection(*grade1.values(),*grade2.values(),*grade3.values())
    
grade1 = {"Alice": {"Math", "English"}, "Bob": {"Math", "Science"}}
grade2 = {"Charlie": {"Math", "History"}, "David": {"Math", "English"}}
grade3 = {"Eva": {"Math", "Music"}, "Frank": {"Math", "Science"}}
grade4 = {}
grade5 = {"Gina": {}, "Hank": {"History"}}
 
print(common_subjects(grade1, grade2, grade3))  # Output: {"Math"}
print(common_subjects(grade1, grade2, grade4))  # Output: set()

