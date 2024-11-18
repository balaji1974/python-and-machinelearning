# In nested for loop the outer loop runs for the as many iterations of the inner loop
for i in range(1,3):
    for j in range(1,3):
        print(f"i = {i}, j = {j}")


# Printing a pattern in Python
for i in range(5): # Range will run from 0 to 5 
	for j in range(5):
		print("*", end="") # will not print newline at the end, but will print an empty character  
	print() # will print empty line


# Printing another pattern in Python
for i in range(5): # Range will run from 0 to 5 
	for j in range(i+1):
		print("*", end="") # will not print newline at the end, but will print an empty character  
	print() # will print empty line

