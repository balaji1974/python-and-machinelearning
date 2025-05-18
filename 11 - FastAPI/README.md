
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

## Advance REST - advrest.py
```xml 
# import basic packages needed 
from fastapi import FastAPI
from typing import List, Optional
from enum import IntEnum
from pydantic import BaseModel, Field

# Setup app and Models needed
app = FastAPI()

class Priority(IntEnum):
    LOW=3
    MEDIUM=2
    HIGH=1 

class TodoBase(BaseModel):
    todo_name: str = Field(..., min_length=3, max_length=512, description='Name of todo')
    todo_description: str = Field(..., description='Description of todo')
    todo_priority: Priority = Field(default=Priority.LOW, description='Priority of todo')

class TodoCreate(TodoBase):
    pass

class Todo(TodoBase):
    todo_id: int = Field(..., description='Unique indetifier of todo')

class TodoUpdate(BaseModel):
    todo_name: Optional[str] = Field(None, min_length=3, max_length=512, description='Name of todo')
    todo_description: Optional[str] = Field(None, description='Description of todo')
    todo_priority: Optional[Priority] = Field(default=Priority.LOW, description='Priority of todo')

# Setup the list object
all_todos= [
    Todo(todo_id=1, todo_name='Sports', todo_description='Go to Gym', todo_priority=Priority.MEDIUM),
    Todo(todo_id=2, todo_name='Read', todo_description='Read 10 pages every day', todo_priority=Priority.HIGH),
    Todo(todo_id=3, todo_name='Shop', todo_description='Go to Shopping', todo_priority=Priority.LOW),
    Todo(todo_id=4, todo_name='Study', todo_description='Study for exam', todo_priority=Priority.HIGH),
    Todo(todo_id=5, todo_name='Meditate', todo_description='Meditate 20 minutes', todo_priority=Priority.MEDIUM)
]

# GET by id
@app.get('/todos/{todo_id}', response_model=Todo)
def get_todo(todo_id : int):
    for todo in all_todos: 
        if todo.todo_id == todo_id:
            return todo

curl --location 'http://localhost:8000/todos/3'


# GET by id or GET all
@app.get('/todos', response_model=List[Todo])
def get_all_todos(first_n: int = None):
    if(first_n): 
        return all_todos[:first_n]
    else: 
        return all_todos

curl --location --request GET 'http://localhost:8000/todos?first_n=3' \
--header 'Content-Type: application/json' --data ''

(or) 

curl --location 'http://localhost:8000/todos'


# POST
@app.put('/todos/{todo_id}', response_model=Todo)
def update_todo(todo_id : int, updated_todo: TodoUpdate):
    for todo in all_todos: 
        if(todo.todo_id == todo_id):
            if updated_todo.todo_name is not None:
                todo.todo_name = updated_todo.todo_name
            if updated_todo.todo_description is not None:
                todo.todo_description = updated_todo.todo_description
            if updated_todo.todo_priority is not None:
                todo.todo_priority = updated_todo.todo_priority
            return todo
    raise HTTPException(status_code=404, detail='Todo not found')

curl --location 'http://localhost:8000/todos' \
--header 'Content-Type: application/json' \
--data '{
    "todo_name" :"Grocery", 
    "todo_description": "Go to Lulu",
    "todo_priority": "1"
}'


# PUT
@app.put('/todos/{todo_id}', response_model=Todo)
def update_todo(todo_id : int, updated_todo: TodoUpdate):
    for todo in all_todos: 
        if(todo.todo_id == todo_id):
            todo.todo_name = updated_todo.todo_name
            todo.todo_description = updated_todo.todo_description
            todo.todo_priority = updated_todo.todo_priority
            return todo
    return {"Error":"Not found"}

curl --location --request PUT 'http://localhost:8000/todos/6' \
--header 'Content-Type: application/json' \
--data '{
    "todo_name" :"Technology", 
    "todo_description": "Buy GPUs",
    "todo_priority": 1
}'


# DELETE 
@app.delete('/todos/{todo_id}', response_model=Todo)
def delete_todo(todo_id : int):
    for index, todo in enumerate(all_todos): 
        if(todo.todo_id == todo_id):
            deleted_todo=all_todos.pop(index)
            return deleted_todo
    return {"Error", "Not Found"}

curl --location --request DELETE 'http://localhost:8000/todos/6'

```

## Advance REST with HTTP Exceptions - advrestwithhttpexception.py
```xml 
Changes needed:
1. from fastapi import HTTPException
2. raise HTTPException(status_code=404, detail='Todo not found')

# Check below for implementation
from fastapi import FastAPI, HTTPException

@app.get('/todos/{todo_id}', response_model=Todo)
def get_todo(todo_id : int):
    for todo in all_todos: 
        if todo.todo_id == todo_id:
            return todo
    raise HTTPException(status_code=404, detail='Todo not found')

```

## Async Requests with FastAPI - asyncrest.py
```xml 
Check the python program asyncrest.py 
It has incode comments with all the explainations 
```

### Reference
```xml
https://www.youtube.com/watch?v=rvFsGRvj9jo
https://fastapi.tiangolo.com/
https://www.youtube.com/watch?v=tGD3653BrZ8&t=10s
```
