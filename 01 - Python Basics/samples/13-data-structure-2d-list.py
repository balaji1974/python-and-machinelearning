# Initilize a two-D list
two_d_list = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# Initialize a 2D list with 3 rows and 4 columns using nested for-loops
rows, cols = 3, 4
two_d_list = []
for i in range(rows):
    row = []
    for j in range(cols):
        row.append(i * 10 + j)
    two_d_list.append(row)
 
print("Initial 2D list:", two_d_list)  
# Initial 2D list: [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]



# Access a specific element in the 2D list
print(f"Element at position [1][2]: {two_d_list[1][2]}")


# Set elements at specific positions
two_d_list[0][0] = 1
two_d_list[0][1] = 2
two_d_list[1][2] = 3
two_d_list[2][3] = 4
print("After updation:", two_d_list)  


# Traverse and print the 2D list
print("2D list traversal:")
for i in range(rows):
    for j in range(cols):
        print(two_d_list[i][j], end=' ')
    print()  


# Sample 2D list
rows = 3
cols = 4
 
two_d_list = []
 
for i in range(rows):
    row = []
    for j in range(cols):
        row.append(i * 10 + j)
    two_d_list.append(row)
 
two_d_list[2][3] = two_d_list[2][3] * 2
 
two_d_list[1][2] = two_d_list[1][2] * 2
 
two_d_list[0][0] = 1
two_d_list[0][1] = 2
two_d_list[1][2] = 3
two_d_list[2][3] = 4
 
#print(two_d_list)
 
 
for i in range(rows):
    for j in range(cols):
        print(two_d_list[i][j], end=' ')
    print()


# function that searches for an element in a 2D list (list of lists) and returns its index. 
# If the element is not present, the function should return (-1, -1)
def search_element(list_2D,target):
    for i in range(len(list_2D)): 
        for j in range(len(list_2D[i])): 
            if list_2D[i][j]==target:
                return i,j
    return -1,-1

# Test the function
print(search_element([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 5))  # Output: (1, 1)
print(search_element([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 10))  # Output: (-1, -1)

# Calculate the sum of each row in a 2D list and store the results in a new list.
def sum_of_rows(list_2D): 
    row_sums=[]
    for i in range(len(list_2D)):
        sum=0
        for j in range(len(list_2D[i])):
            sum+= list_2D[i][j]
        row_sums.append(sum)
    return row_sums
# Test the function
print(sum_of_rows([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # Output: [6, 15, 24]
print(sum_of_rows([[1], [2], [3]]))  # Output: [1, 2, 3]
print(sum_of_rows([[], [], []]))  # Output: [0, 0, 0]

# function that adds two matrices of the same size. If either of the matrices is empty, return an empty list.
def add_matrices(matrix1,matrix2):
    if not matrix1 or not matrix2: 
        return []
    matrix3=[]
    for i in range(len(matrix1)):
        row=[]
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j]+matrix2[i][j])
        matrix3.append(row)
    return matrix3

# Testing the function
print(add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]))  # Output: [[6, 8], [10, 12]]
print(add_matrices([], []))  # Output: []
