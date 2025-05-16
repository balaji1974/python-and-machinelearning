
# Fast API

## Introduction
```xml 
What is FAST API
------------------
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 
based on standard Python type hints.

Installation
------------
pip install "fastapi[standard]"


```


## Run the first application - helloworld.py
```xml 
# import FastAPI library
from fastapi import FastAPI
app = FastAPI()

# use the imported library
@app.get("/")
def read_root():
    return {"Greetings" : "Hello World"}

# Run the server
fastapi dev helloworld.py

# Test the API
curl --location 'http://127.0.0.1:8000'

# Open documentation - Swagger
# This is useful for testing 
http://localhost:8000/docs

```

## Create Basic HTTP - GET, POST, PUT and DELETE - basicrest.py
```xml 

# Run the server
fastapi dev basicrest.py

# Setup a basic array to manipulate within the application
all_todos= [
    {'todo_id':1, 'todo_name':'Sports', 'todo_description':'Go to Gym'},
    {'todo_id':2, 'todo_name':'Read', 'todo_description':'Read 10 pages every day'},
    {'todo_id':3, 'todo_name':'Shop', 'todo_description':'Go shopping'},
    {'todo_id':4, 'todo_name':'Study', 'todo_description':'Study for exam'},
    {'todo_id':5, 'todo_name':'Mediate', 'todo_description':'Meditate 20 minutes'}
]

# GET - Return an index sent as path parameter
@app.get('/todos/{todo_id}')
def get_todo(todo_id : int): # note type of the path parameter is specified or else default is string
    for todo in all_todos: 
        if(todo['todo_id'] == todo_id):
            return todo

# GET - Return the full array or until the size sent as param value
@app.get('/todos')
def get_all_todos(first_n: int = None):
    if(first_n): 
        return all_todos[:first_n]
    else: 
        return all_todos

curl --location 'http://localhost:8000/todos?first_n=2'
or 
curl --location 'http://localhost:8000/todos'


# POST - Create a new todo record 
@app.post('/todos')
def create_todo(todo : dict):
    new_todo_id =  max(todo['todo_id'] for todo in all_todos) +1
    new_todo = {
        'todo_id': new_todo_id, 
        'todo_name': todo['todo_name'], 
        'todo_description': todo['todo_description']
    }
    all_todos.append(new_todo)
    return new_todo

curl --location 'http://localhost:8000/todos' \
--header 'Content-Type: application/json' \
--data '{
    "todo_name" :"Grocery", 
    "todo_description": "Go to Market"
}'

# PUT - Update an existing todo record
@app.put('/todos/{todo_id}')
def update_todo(todo_id : int, updated_todo: dict):
    for todo in all_todos: 
        if(todo['todo_id'] == todo_id):
            todo['todo_name'] = updated_todo['todo_name']
            todo['todo_description'] = updated_todo['todo_description']
            return todo
    return {"Error", "Not Found"}

curl --location --request PUT 'http://localhost:8000/todos/1' \
--header 'Content-Type: application/json' \
--data '{
    "todo_name" :"Technology", 
    "todo_description": "Buy GPUs"
}'


# DELETE - Delete an existing todo record
@app.delete('/todos/{todo_id}')
def delete_todo(todo_id : int):
    for index, todo in enumerate(all_todos): 
        if(todo['todo_id'] == todo_id):
            deleted_todo=all_todos.pop(index)
            return deleted_todo
    return {"Error", "Not Found"}

curl --location --request DELETE 'http://localhost:8000/todos/7'

```


### Reference
```xml
https://www.youtube.com/watch?v=rvFsGRvj9jo
https://fastapi.tiangolo.com/

```
