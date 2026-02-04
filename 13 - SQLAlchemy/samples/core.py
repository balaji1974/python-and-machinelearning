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





