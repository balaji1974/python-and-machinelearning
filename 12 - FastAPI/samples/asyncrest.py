import time, asyncio
from fastapi import FastAPI

app = FastAPI()



"""
All request run in main thread 
No awaitable operations, cannot be paused 
Sequential Order
Request 1
    Hello World
    Bye
Request 2
    Hello World
    Bye  
"""
@app.get('/1')
async def first_endpoint(): # Processed Sequentially
    print("Hello World")
    time.sleep(5) # Blocking IO operation, cannot be awaited
    # Function excection cannot be paused, so event loop is blocked 
    print("Bye")



"""
All request run in main thread 
Awaitable operations, can be paused 
Concurrent Order
Request 1
    Hello World
Request 2
    Hello World
Request 1
    Bye
Request 2
    Bye
"""
@app.get('/2')
async def second_endpoint(): # Processed Concurrently 
    print("Hello World")
    await asyncio.sleep(5) # Non blocking IO operation, awaited
    # Function excection paused, so event loop is not blocked 
    print("Bye")



"""
All request run in seperate thread 
Parallel Order
No particular sequence is followed
"""
@app.get('/3') # Processed Parallely 
def third_endpoint(): # Processed 
    print("Hello World")
    time.sleep(5)
    print("Bye")