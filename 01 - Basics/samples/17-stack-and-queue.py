# A stack is a LIFO (Last In, First Out) data structure. This means the last element you insert is the first one you take out.

numbers = []
# Push - add elements to the top of the stack  
numbers.append(1)
numbers.append(2)
numbers.append(3)
numbers.append(4)
 
print("Stack after pushes:", numbers)  # Output: Stack after pushes: [1, 2, 3, 4]

# remove from top of the stack
print("Popped element:", numbers.pop())  # Output: Popped element: 4

# Inspect the top of the stack 
print("Top of stack:", numbers[-1])  # Output: Top of stack: 3

# Check if the stack is empty 
print("Is stack empty?", not bool(numbers))  # Output: Is stack empty? False


# Build a stack class
# Initialize the Stack class
class Stack:
    def __init__(self):
        self.items = []  # Initialize an empty list

    # Add an element to the top of the stack
    def push(self, item):
        self.items.append(item)  # Append the item to the list

    # Check if the stack is empty
    def is_empty(self):
        return len(self.items) == 0  # True if empty, otherwise False

    # Remove and return the top element
    def pop(self):
        if not self.is_empty():  # Check if the stack is not empty
            return self.items.pop()  # Remove and return the last element

    # Look at the top element without removing it
    def top(self):
        if not self.is_empty():  # Check if the stack is not empty
            return self.items[-1]  # Return the last element

    # Display the elements in the stack
    def display(self):
        print(self.items)  # Print the list items


# Create a Stack
s = Stack()
 
# Push elements
s.push(1)  # Adding 1 to the stack
s.push(2)  # Adding 2 to the stack
s.push(3)  # Adding 3 to the stack
 
# Display the stack
s.display()  # Output: [1, 2, 3]
 
# Pop elements
print(s.pop())  # Output: 3 (removing top element)
s.display()  # Output: [1, 2]
 
# Top element
print(s.top())  # Output: 2 (top element)
 
# Check if stack is empty
print(s.is_empty())  # Output: False (stack is not empty)


# Enqueue (Add to rear of queue)
# Use the append() method to add elements.
numbers = []
numbers.append(1)
numbers.append(2)
numbers.append(3)
numbers.append(4)
 
print("Queue after enqueues:", numbers)  # Output: Queue after enqueues: [1, 2, 3, 4]

# Dequeue (Remove front of queue)
# Use pop(0) method to remove and return the first element.
print("Dequeued element:", numbers.pop(0))  # Output: Dequeued element: 1


# Front (Inspect front of queue)
# Look at the first element without removing it.
print("Front of queue:", numbers[0])  # Output: Front of queue: 2


# IsEmpty (Check if queue is empty)
# Use the length of the list to check.
print("Is queue empty?", not bool(numbers))  # Output: Is queue empty? False

# Build a Queue class
# Queue class
class Queue:
    def __init__(self):
        self.items = []  # Initialize an empty list
    
    def enqueue(self, item):
        self.items.append(item)  # Append the item to the list
        
    def dequeue(self):
        if not self.is_empty():  # Check if the queue is not empty
            return self.items.pop(0)  # Remove and return the first element
        
    def front(self):
        if not self.is_empty():  # Check if the queue is not empty
            return self.items[0]  # Return the first element
    
    def is_empty(self):
        return len(self.items) == 0  # True if empty, otherwise False
    
    def display(self):
        print(self.items)  # Print the list items
 
# Create a Queue
q = Queue()
 
# Enqueue elements
q.enqueue(1)  # Adding 1 to the queue
q.enqueue(2)  # Adding 2 to the queue
q.enqueue(3)  # Adding 3 to the queue
 
# Display the queue
q.display()  # Output: [1, 2, 3]
 
# Dequeue elements
print(q.dequeue())  # Output: 1 (removing front element)
q.display()  # Output: [2, 3]
 
# Front element
print(q.front())  # Output: 2 (front element)
 
# Check if queue is empty
print(q.is_empty())  # Output: False (queue is not empty)

