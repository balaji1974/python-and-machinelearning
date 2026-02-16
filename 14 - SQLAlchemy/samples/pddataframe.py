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

