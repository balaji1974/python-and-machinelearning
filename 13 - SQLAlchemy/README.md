
# SQLAlchemy

## Introduction
```xml 
SQLAlchemy is the Python SQL toolkit and Object Relational Mapper 
that gives application developers the full power and flexibility of SQL.

It provides a full suite of well known enterprise-level persistence patterns, 
designed for efficient and high-performing database access, 
adapted into a simple and Pythonic domain language.


Installing SQLAlchmey
pip install SQLAlchemy


# Checking the SQLAlchmey version (version.py)
import sqlalchemy as sa
print(sa.__version__)

```


## Basic connection and sql (basics.py)
```xml 
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

# Run the python progam
python basics.py

# Check if the database is created in the folder where you run the python script from
ls my*

# Connect to the sqllite db created
sqlite3 mydatabase.db

# Check for the table
.table 

# Check table info
PRAGMA table_info(people);

#Quit and come out
.q


# Import session from SQLAlchemy ORM for database querying
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

```

## SQLAlchemy Core (core.py)
```xml 
rm mydatabase.db


# import create engine from sql alchemy which is used for database connections
# import MetaData to hold the schema information
# import Table to create a table object
# import Column to define the columns of the table
# import Integer and String to define the data types of the columns
# import String to define the data types of the columns
# import insert to insert data into the table
# import Float to define the data types of the columns
# import ForeignKey to define foreign key constraints
# import Select to select data from the table
# import func to use SQL functions
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, insert, Float, ForeignKey, func


# Use the create engine to set to a database URL
# with echo as true for debugging in development environment
# Create an engine to connect to the SQLite database
engine = create_engine("sqlite:///mydatabase.db", echo=True)

# Create a MetaData instance to hold the schema information
meta = MetaData()

# Define a table object for the "people" table
# with columns "id", "name", and "age"
# The "id" column is the primary key
# The "name" column is a string and cannot be null
# The "age" column is an integer
people = Table(
    "people", meta,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("age", Integer)
)

things = Table(
    "things", meta,
    Column("id", Integer, primary_key=True),
    Column("description", String, nullable=False),
    Column("value", Float),
    Column("owner", Integer, ForeignKey("people.id"), nullable=False)
)

# Create the table in the database
meta.create_all(engine)  

# Create a connection to the database
conn = engine.connect()

# Insert a new person into the "people" table
# with the name "John Doe" and age 30
# insert = people.insert().values(name="John Doe", age=30)
# or
insert_people = people.insert().values(
    [
        {"name": "John Doe", "age": 30},
        {"name": "Jane Smith", "age": 25},
        {"name": "Alice Johnson", "age": 28},
        {"name": "Bob Brown", "age": 35},
        {"name": "Charlie Davis", "age": 22},
        {"name": "David Wilson", "age": 40},
        {"name": "Eve Thompson", "age": 27},
    ]
)

insert_things = things.insert().values(
    [
        {"description": "Laptop", "value": 1200.50, "owner": 1},
        {"description": "Smartphone", "value": 800.00, "owner": 2},
        {"description": "Tablet", "value": 500.00, "owner": 3},
        {"description": "Headphones", "value": 150.00, "owner": 4},
        {"description": "Smartwatch", "value": 250.00, "owner": 5},
        {"description": "Camera", "value": 900.00, "owner": 6},
        {"description": "Monitor", "value": 300.00, "owner": 7}
    ]
)    

# Execute the insert statement
result = conn.execute(insert_people)


# Commit the changes to the database
conn.commit()

# Execute the insert statement
result = conn.execute(insert_things)

# Commit the changes to the database
conn.commit()

# Query the database to retrieve all people
# Select all rows from the "people" table
# select = people.select()
# or
# Select all rows from the "people" table where age is greater than 25
select = people.select().where(people.c.age > 25)

# Execute the select statement
result = conn.execute(select)

# Fetch all rows from the result
rows = result.fetchall()

# Print the results
for row in rows:
    print(row)


# Select all rows from the "things" table 
select = things.select()

# Execute the select statement
result = conn.execute(select)

# Fetch all rows from the result
rows = result.fetchall()

# Print the results
for row in rows:
    print(row)


# Update the age of the person with name "John Doe" to 31
update = people.update().where(people.c.name == "John Doe").values(age=31)

# Execute the update statement
result = conn.execute(update)

# Commit the changes to the database
conn.commit()

# select all rows from the "people" table
select = people.select()

# Execute the select statement
result = conn.execute(select)

# Fetch all rows from the result
rows = result.fetchall()

# Print the results
for row in rows:
    print(row)

# Join the "people" and "things" tables on the owner column
join = people.join(things, people.c.id == things.c.owner)

# Select columns from both tables
select = people.select().with_only_columns(people.c.name, things.c.description).select_from(join)

# Execute the select statement
result = conn.execute(select)

# Fetch all rows from the result
rows = result.fetchall()

# Print the results
for row in rows:
    print(row)


# Group by the owner column and count the number of things owned by each person
group_by = things.select().group_by(things.c.owner).with_only_columns(things.c.owner, func.sum(things.c.value))

# Execute the select statement
result = conn.execute(group_by)

# Fetch all rows from the result
rows = result.fetchall()

# Print the results
for row in rows:
    print(row)


# Group by the owner column and count the number of things owned by each person having value greater than 5000
group_by = things.select().group_by(things.c.owner).with_only_columns(things.c.owner, func.sum(things.c.value)).having(func.sum(things.c.value) > 5000)

# Execute the select statement
result = conn.execute(group_by)

# Fetch all rows from the result
rows = result.fetchall()

# Print the results
for row in rows:
    print(row)

# Close the connection
conn.close()


# Close the metadata
meta.clear()

# Close the engine
engine.dispose()

```

## SQLAlchemy ORM (orm.py) 
```xml 

rm mydatabase.db

# import create_engine from sql alchemy which is used for database connections
# import column to define the columns of the table
# import Integer and String to define the data types of the columns
# import Float to define the data types of the columns
# import ForeignKey to define foreign key constraints
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, func

# import declarative_base to create a base class for the ORM models
# import sessionmaker to create a session factory
# import relationship to define relationships between tables
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Use the create engine to set to a database URL
# with echo as true for debugging in development environment
engine=create_engine("sqlite:///mydatabase.db", echo=True)

# Base class for the ORM models
Base = declarative_base()

# Define the Person class as a model for the "people" table
# The class inherits from the Base class
# The __tablename__ attribute specifies the name of the table in the database
# The class attributes define the columns of the table
# The id column is an Integer and is the primary key
# The name column is a String and cannot be null
# The age column is an Integer
# The things attribute defines a relationship to the Thing class
# The relationship is bidirectional, meaning that the Thing class will also have a reference to the Person class
# The relationship is defined using the relationship function
# The back_populates argument specifies the name of the attribute in the Thing class that refers back to the Person class
class Person(Base):
    __tablename__ = "people"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    # Define a relationship to the Thing class  
    things = relationship("Thing", back_populates="person")

# Define the Thing class as a model for the "things" table
# The class inherits from the Base class
# The __tablename__ attribute specifies the name of the table in the database
# The class attributes define the columns of the table
# The id column is an Integer and is the primary key
# The description column is a String and cannot be null
# The value column is a Float
# The owner column is an Integer and is a foreign key that references the id column in the "people" table
# The owner attribute defines a relationship to the Person class
# The relationship is bidirectional, meaning that the Person class will also have a reference to the Thing class
# The relationship is defined using the relationship function
# The back_populates argument specifies the name of the attribute in the Person class that refers back to the Thing class
# The person attribute is used to access the owner of the thing
class Thing(Base):
    __tablename__ = "things"
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=False)
    value = Column(Float)
    owner = Column(Integer, ForeignKey("people.id"), nullable=False)
    # Define a relationship to the Person class
    person = relationship("Person", back_populates="things")   

# Create the tables in the database
Base.metadata.create_all(engine)  

# Create a session factory
Session = sessionmaker(bind=engine)

# Create a session to interact with the database
session = Session()

# Create a new person object
person = Person(name="Charlie", age=70)

# Add the new person to the session
session.add(person)
session.flush()  # Flush the session to get the new person's ID 

# Create a new thing object
new_thing = Thing(description="Laptop", value=1000.0, owner=person.id)

# Add the new thing to the session
session.add(new_thing)

# Commit the changes to the database
session.commit()

# Query the database to retrieve all people
result = session.query(Person).all()

# Print the results
for person in result:
    print(f"ID: {person.id}, Name: {person.name}, Age: {person.age}")
    for thing in person.things:
        print(f"  Thing ID: {thing.id}, Description: {thing.description}, Value: {thing.value}")

# Sample query with group by 
result = session.query(Thing.description, func.sum(Thing.value)).group_by(Thing.description).all()
print (result)


# join query
result = session.query(Person.name, Thing.description, Thing.value).join(Thing).all()
print (result)

# Close the session
session.close()

```

## SQLAlchemy - SQL to Dataframes and viseversa (pddataframe.py) 
```xml 
# import pandas
import pandas as pd

# import create_engine
from sqlalchemy import create_engine

# Use the create engine to set to a database URL
# with echo as true for debugging in development environment
engine=create_engine("sqlite:///mydatabase.db", echo=True)

# Create a dataframe from the "people" table
df= pd.read_sql("SELECT * FROM people", con=engine)

# Print the dataframe
print(df)

# Create a new dataframe with the data to be inserted
new_data = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})

# Insert the new data into the "people" table
new_data.to_sql('people', con=engine, if_exists='append', index=False)

# Print the updated dataframe\
df = pd.read_sql("SELECT * FROM people", con=engine)

# Print the updated dataframe
print(df)

```



## Also check 
```xml 
Check:
simple.py -> Basic connection, create a record and display it using in-memory SQLlite DB
oop.py -> Create records using Object Orientied Programing way
rel.py -> Creates the relationship using OOP

```


### Reference
```xml
https://www.sqlalchemy.org/
https://www.youtube.com/watch?v=aAy-B6KPld8
https://github.com/ArjanCodes/examples/tree/main/2024/sqlalchemy

https://www.youtube.com/watch?v=529LYDgRTgQ

```
