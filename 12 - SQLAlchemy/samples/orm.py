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
