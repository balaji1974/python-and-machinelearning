# String in Python
print("Hello World")  # Output: Hello World
print('Hello World')  # Output: Hello World


#The type() method allows you to find the type of a variable.
message = "Hello World"
print(type(message))  # Output: <class 'str'>


#Converting to uppercase and lowercase
message = "hello"
print(message.upper())  # Output: HELLO
print(message.lower())  # Output: hello
print("hello".capitalize())  # Output: Hello
print('hello'.capitalize())  # Output: Hello


#Checking lower case, title case, and upper case
print('hello'.islower())  # Output: True
print('Hello'.islower())  # Output: False
print('Hello'.istitle())  # Output: True
print('hello'.istitle())  # Output: False
print('hello'.isupper())  # Output: False
print('Hello'.isupper())  # Output: False
print('HELLO'.isupper())  # Output: True


#Checking if a string is a numeric value
print('123'.isdigit())  # Output: True
print('A23'.isdigit())  # Output: False
print('2 3'.isdigit())  # Output: False
print('23'.isdigit())   # Output: True


#Checking if a string only contains alphabets or alphabets and numerals
print('23'.isalpha())   # Output: False
print('2A'.isalpha())   # Output: False
print('ABC'.isalpha())  # Output: True
print('ABC123'.isalnum())  # Output: True
print('ABC 123'.isalnum())  # Output: False


#Checking if a string ends or starts with a specific substring
print('Hello World'.endswith('World'))   # Output: True
print('Hello World'.endswith('ld'))      # Output: True
print('Hello World'.endswith('old'))     # Output: False
print('Hello World'.endswith('Wo'))      # Output: False
print('Hello World'.startswith('Wo'))    # Output: False
print('Hello World'.startswith('He'))    # Output: True
print('Hello World'.startswith('Hell0')) # Output: False
print('Hello World'.startswith('Hello')) # Output: True


#Finding a substring within a string
print('Hello World'.find('Hello'))   # Output: 0
print('Hello World'.find('ello'))    # Output: 1
print('Hello World'.find('Ello'))    # Output: -1
print('Hello World'.find('bello'))   # Output: -1
print('Hello World'.find('Ello'))    # Output: -1


#You can use the in keyword to check whether a character or sequence of characters exists within a specific set.
print('Hello' in 'Hello World')   # Output: True
print('ello' in 'Hello World')    # Output: True
print('Ello' in 'Hello World')   # Output: False
print('bello' in 'Hello World')    # Output: False

# accessing string index values
message = "Hello World"
print(message[0])  # Output: 'H'
print(type(message[0]))  # Output: <class 'str'>
print(type(message))    # Output: <class 'str'>


# Printing all characters in a string
message = "Hello World"
for ch in message:
    print(ch)

# import the string module
import string

print(string.ascii_letters)        # Output: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
print(string.ascii_lowercase)      # Output: abcdefghijklmnopqrstuvwxyz
print(string.ascii_uppercase)      # Output: ABCDEFGHIJKLMNOPQRSTUVWXYZ
print(string.digits)               # Output: 0123456789
print(string.hexdigits)            # Output: 0123456789abcdefABCDEF
print(string.punctuation)          # Output: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

#Comparing Strings
str1 = "test"
str2 = "test1"
print(str1)  # Output: test
print(str2)  # Output: test1
print(str1 == str2)  # Output: False

# convert string to its ascii value 
str1='A'
print(ord(str1))

# Check if a character is vowel or not 
def is_vowel(char):
    return char in 'aeiouAEIOU'

print(is_vowel('a'))  # Output: True
print(is_vowel('b'))  # Output: False


# Count uppercase characters in a given string 
import string
def count_uppercase_letters(text) :
    count=0
    for i in text:
        if(i.isupper()) :
            count+=1 
    return count

print(count_uppercase_letters('Hello WORLD'))  # Output: 6

# check if a given string has at least two consecutive identical characters.
def has_consecutive_identical_characters(text) :
    strlength=len(text)
    for i in range (0,strlength-2):
        if(text[i]==text[i+1]):
            return True
    return False
 
print(has_consecutive_identical_characters('Hello World'))  # Output: True
print(has_consecutive_identical_characters('I love Switzerland'))  # Output: False


#Code to return the right-most digit found in a given string.
import string
def find_right_most_digit(text) :
    for i in reversed(text):
        if(i.isdigit()): 
            return int(i)
    return -1
print(find_right_most_digit('The value is 42'))  # Output: 2
print(find_right_most_digit('No digits here'))  # Output: -1

# function that identifies and returns the left-most longest word in a provided text
import string
def find_longest_word(text) :
    x=text.split()
    long_word=''
    length_long_word=0
    for ch in x:
        if(len(ch)>length_long_word) :
            length_long_word=len(ch)
            long_word=ch
    return long_word

print(find_longest_word('The quick brown fox jumps over the lazy dog'))  # Output: 'quick'
print(find_longest_word(''))  # Output: ''


# checks whether two given strings are anagrams of each other. 
# An anagram is a word or phrase formed by rearranging the letters of a different word or phrase.
import string 
def is_anagram(string1, string2) :
    return sorted(string1)==sorted(string2)

print(is_anagram('listen', 'silent'))  # Output: True
print(is_anagram('hello', 'world'))  # Output: False



# checks whether a given string consists solely of valid hexadecimal characters.  
def is_hex_string(string): 
    if(string =='') :
        return False
    hex_digits='0123456789abcdefABCDEF'
    for t in string: 
        if(t not in hex_digits) :
            return False
    return True
            
print(is_hex_string('1a2f4C'))  # Output: True
print(is_hex_string('1g2f4C'))  # Output: False
print(is_hex_string(''))        # Output: False

# create a function that takes a string as input and returns the reversed version of the string.
def reverse_word(word) : 
    reversed_word=''
    for w in word :
        reversed_word=w+reversed_word
    return reversed_word
print(reverse_word("Python"))  # Output: "nohtyP"





