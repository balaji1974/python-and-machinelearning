# import create engine from sql alchemy which is used for database connections
# import text a function for wrapping the SQL commands
from sqlalchemy import create_engine, text 

# Use the create engine to set to a database URL
# with echo as true for debugging in development environment
engine = create_engine("sqlite:///mydatabase.db", echo=True)

# Use the created engine and connect it to the db. 
connection = engine.connect()

# Create the DDL table script using execute method
connection.execute(text("CREATE TABLE IF NOT EXISTS people(name str, age int)"))

#Commit the change
connection.commit()


from sqlalchemy.orm import Session  

# Create a session to interact with the database
session = Session(engine)

# Create a new person object
new_person = {"name": "John Doe", "age": 30}

# Add the new person to the session
session.execute(
    text("INSERT INTO people (name, age) VALUES (:name, :age)"),
    new_person
)

# Commit the changes to the database
session.commit()

# Query the database to retrieve all people
result = session.execute(text("SELECT * FROM people")).fetchall()

# Print the results
for row in result:
    print(row)

