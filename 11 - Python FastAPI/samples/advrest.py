from fastapi import FastAPI
from typing import List, Optional
from enum import IntEnum
from pydantic import BaseModel, Field

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


all_todos= [
    Todo(todo_id=1, todo_name='Sports', todo_description='Go to Gym', todo_priority=Priority.MEDIUM),
    Todo(todo_id=2, todo_name='Read', todo_description='Read 10 pages every day', todo_priority=Priority.HIGH),
    Todo(todo_id=3, todo_name='Shop', todo_description='Go to Shopping', todo_priority=Priority.LOW),
    Todo(todo_id=4, todo_name='Study', todo_description='Study for exam', todo_priority=Priority.HIGH),
    Todo(todo_id=5, todo_name='Meditate', todo_description='Meditate 20 minutes', todo_priority=Priority.MEDIUM)
]



@app.get('/todos/{todo_id}', response_model=Todo)
def get_todo(todo_id : int):
    for todo in all_todos: 
        if todo.todo_id == todo_id:
            return todo


@app.get('/todos', response_model=List[Todo])
def get_all_todos(first_n: int = None):
    if(first_n): 
        return all_todos[:first_n]
    else: 
        return all_todos


@app.post('/todos', response_model=Todo)
def create_todo(todo : TodoCreate):
    new_todo_id =  max(todo.todo_id for todo in all_todos) + 1
    new_todo = Todo(
                todo_id=new_todo_id, 
                todo_name=todo.todo_name, 
                todo_description=todo.todo_description,
                todo_priority = todo.todo_priority
                )
    all_todos.append(new_todo)
    return new_todo


@app.put('/todos/{todo_id}', response_model=Todo)
def update_todo(todo_id : int, updated_todo: TodoUpdate):
    for todo in all_todos: 
        if(todo.todo_id == todo_id):
            todo.todo_name = updated_todo.todo_name
            todo.todo_description = updated_todo.todo_description
            todo.todo_priority = updated_todo.todo_priority
            return todo
    return {"Error":"Not found"}


@app.delete('/todos/{todo_id}', response_model=Todo)
def delete_todo(todo_id : int):
    for index, todo in enumerate(all_todos): 
        if(todo.todo_id == todo_id):
            deleted_todo=all_todos.pop(index)
            return deleted_todo
    return {"Error", "Not Found"}