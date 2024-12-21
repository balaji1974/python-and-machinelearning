
# Pandas 



## Pandas - Intro
```xml

Pandas is a data analysis library built on top of Python 
It is short form for "Panel Data Structure" meaning 
"data sets with observations over time"

It is also termed as “Excel for Python”

Pandas documentation
--------------------
https://pandas.pydata.org/pandas-docs/stable/

10 Minutes to Pandas
--------------------
https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min


Why Pandas? 
Pandas allows us to analyze big data and 
make conclusions based on statistical theories. 
Pandas can clean messy data sets, and make them readable and relevant. 
Relevant data is very important in data science

Pandas are Data analysis library built on top of the 
Python programming language
It is a robust toolkit for analyzing, filtering, manipulating, 
aggregating, merging, pivoting, and cleaning data

```

## Series - Creation and Assignment
```xml

Series are one Dimensional or a single column of data 
A Series combines the best features of a list and a dictionary.
A Series maintains a single collection of ordered values 
(i.e. a single column of data).
We can assign each value an identifier, which does not have to be unique.
If the identifier is not assigned the series automatically assigns 
unique numbers as identifier. 

# Import Panda library 
import pandas as pd  

# Create a series with data from list  
series = pd.Series(["BMW","Toyota","Honda"]) 
#print the series that was created 
series 

colors = pd.Series(["Red", "Yello", "Blue"])
colors

ice_cream = ["Chocolate", "Vanilla", "Strawberry", "Rum Raisin"]
pd.Series(ice_cream)

# Create a series from a dictionary
sushi = {
    "Salmon": "Orange",
    "Tuna": "Red",
    "Eel": "Brown"
}
pd.Series(sushi)


Exercise
--------
# Import the pandas library and assign it its "pd" alias
import pandas as pd 

# Create a list with 4 countries - United States, France, Germany, Italy
# Create a new Series by passing in the list of countries
# Assign the Series to a "countries" variable
country=["United States", "France", "Germany", "Italy"]
countries=pd.Series(country)

# Create a list with 3 colors - red, green, blue
# Create a new Series by passing in the list of colors
# Assign the Series to a "colors" variable
color=["red", "green", "blue"]
colors=pd.Series(color)

# Given the "recipe" dictionary below,
# create a new Series by passing in the dictionary as the data source
# Assign the resulting Series to a "series_dict" variable
recipe = {
  "Flour": True,
  "Sugar": True,
  "Salt": False
}
series_dict=pd.Series(recipe)

```

## Series - Methods
```xml
# Create a series and assign float values
prices = pd.Series([2.99, 4.45, 1.36])
prices

# Calculates the sum of the values in the series
prices.sum()

# Calculates the product of the values in the series
prices.product()

# Calculates the mean of the values in the series
prices.mean()

# Calculates the standard deviation of the values in the series
prices.std()

```

## Series - Attributes 
```xml

An attribute is a piece of data that lives on an object.
An attribute is a fact, a detail, a characteristic of the object.
Access an attribute with `object.attribute` syntax.

# Create a series and assign values
adjectives = pd.Series(["Smart", "Handsome", "Charming", "Brilliant", "Humble", "Smart"])
adjectives

# The `size` attribute returns a count of the number of values in the Series.
adjectives.size

# The `is_unique` attribute returns True if the Series has no duplicate values.
adjectives.is_unique

# The `values` and `index` attributes return the underlying objects 
# that holds the Series' values and index labels.

# return type is numpy.ndarray -> relies on numpy library
adjectives.values

# return type is pandas.core.indexes.range.RangeIndex
adjectives.index

Exercise
--------
import pandas as pd

# The Series below stores the number of home runs
# that a baseball player hit per game
home_runs = pd.Series([3, 4, 8, 2])

# Find the total number of home runs (i.e. the sum) and assign it
# to the total_home_runs variable below
total_home_runs = home_runs.sum()

# Find the average number of home runs and assign it
# to the average_home_runs variable below
average_home_runs = home_runs.mean()


```

## Series - Parameter and Arguments 
```xml
A parameter is the name for an expected input to a function/method/class instantiation.
An argument is the concrete value we provide for a parameter during invocation.
We can pass arguments either sequentially (based on parameter order) 
or with explicit parameter names written out.
The first two parameters for the Series constructor are `data` and `index`, 
which represent the values and the index labels.

fruits = ["Apple", "Orange", "Plum", "Grape", "Blueberry", "Watermelon"]
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Monday"]

# Value is fruits
pd.Series(fruits)

# Value is weekdays
pd.Series(weekdays)

# Value is fruits and index is weekdays
pd.Series(fruits, weekdays)

# Value is weekdays and index is fruits
pd.Series(weekdays, fruits)

# All the 3 below will produce the exact same result 
# where data is fruits and index is weekdays
pd.Series(data=fruits, index=weekdays)
pd.Series(index=weekdays, data=fruits)
pd.Series(fruits, index=weekdays)

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# The code below defines a list of delicious foods
# and some dipping sauces to dip them in
import pandas as pd

foods = ["French Fries", "Chicken Nuggets", "Celery", "Carrots"]
dipping_sauces = ["BBQ", "Honey Mustard", "Ranch", "Sriracha"]

# Create a Series and assign it to the s1 variable below. 
# Assign the foods list as the data source
# and the dipping_sauces list as the Series index 
# For this solution, use positional arguments (i.e. feed in the arguments sequentially)
s1 = pd.Series(foods,dipping_sauces)

# Create a Series and assign it to the s2 variable below. 
# Assign the dipping_sauces list as the data source
# and the foods list as the Series index 
# For this solution, use keyword arguments (i.e. provide the parameter names
# alongside the arguments)
s2 = pd.Series(data=dipping_sauces, index=foods)

```

## Series - Import from csv
```xml

A CSV is a plain text file that uses line breaks to separate rows 
and commas to separate row values.

Pandas ships with many different `read_` functions for different types of files.
The `read_csv` function accepts many different parameters. 
The first one specifies the file name/path.
The `read_csv` function will import the dataset as a DataFrame, a 2-dimensional table.
The `usecols` parameter accepts a list of the column(s) to import.
The `squeeze` method converts a DataFrame to a Series.

# Make sure pokemom.csv file is in the same directory as Jupyter notebook
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
pokemon

# Another example
google = pd.read_csv("google_stock_price.csv", usecols=["Price"]).squeeze("columns")
google

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd

# We have a foods.csv CSV file with 3 columns: Item Number, Menu Item, Price
# You can explore the data by clicking into the foods.csv file on the left
# Import the CSV file into a pandas Series object
# The Series should have the standard pandas numeric index
# The Series values should be the string values from the "Menu Item" column
# Assign the Series object to a "foods" variable
foods=pd.read_csv("foods.csv",usecols=["Menu Item"]).squeeze("columns")

```

## Series - Head & Tail
```xml

The `head` method returns a number of rows from the top/beginning of the `Series`.
The `tail` method returns a number of rows from the bottom/end of the `Series`.

# Read data into a series
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
google = pd.read_csv("google_stock_price.csv", usecols=["Price"]).squeeze("columns")

# Will return first 5 elements
pokemon.head()

# Same as above
pokemon.head(5)

# Same as above
pokemon.head(n=5)

# Will return first 'n' elements 
pokemon.head(8)
pokemon.head(1)

# This is exact same like head but will display data from the bottom of the series
google.tail()
google.tail(5)
google.tail(n=5)

google.tail(7)
google.tail(n=2)

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd

# We have a roller_coasters.csv CSV file with 4 columns: Name, Park, Country, and Height.
# You can explore the data by clicking into the CSV file on the left
# Import the CSV file into a pandas Series object
# The Series should have the standard pandas numeric index
# The Series values should be the string values from the "Name" column
# Assign the Series object to a "coasters" variable
coasters=pd.read_csv("roller_coasters.csv", usecols=["Name"]).squeeze("columns")

# I only want to ride the top 3 roller coasters on the list.
# Starting with the "coasters" Series, extract the first 3 rows in a new Series.
# Assign the new Series to a "top_three" variable.
top_three=coasters.head(3)

# I'm now curious about some of the last entries on the coaster list.
# Starting with the "coasters" Series, extract the last 4 rows in a new Series.
# Assign the new Series to a "bottom_four" variable.
bottom_four=coasters.tail(4)

```

## Passing series to python built-in functions
```xml
# Read data into series 
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
google = pd.read_csv("google_stock_price.csv", usecols=["Price"]).squeeze("columns")

# The `len` function returns the length of the Series.
len(pokemon)

# The `type` function returns the type of an object
type(pokemon)

# The `list` function converts the Series to a list.
list(pokemon)

# The `dict` function converts the Series to a dictionary.
dict(pokemon)

# The `sorted` function converts the Series to a sorted list.
sorted(pokemon)
type(sorted(pokemon))
sorted(google)

# The `max` function returns the largest value in the Series.
# The `min` function returns the smalllest value in the Series.
max(google)
min(google)

max(pokemon)
min(pokemon)

```

## Series - Inclusions with 'in' key word
```xml

# import from csv file 
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
google = pd.read_csv("google_stock_price.csv", usecols=["Price"]).squeeze("columns")
pokemon.head()

# The `in` keyword checks if a value exists within an object.
"car" in "racecar"
2 in [3, 2, 1]

# The `in` keyword will look for a value in the Series's index by default. 
"Bulbasaur" in pokemon
0 in pokemon

# Use the `index` and `values` attributes to access "nested" objects within the Series.
5 in pokemon.index

# Combine the `in` keyword with `values` to search within the Series's values.
"Bulbasaur" in pokemon.values
"Pikachu" in pokemon.values
"Nonsense" in pokemon.values


```

## Series - sort_values Method
```xml

# import csv file 
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
google = pd.read_csv("google_stock_price.csv", usecols=["Price"]).squeeze("columns")

google.head()

# The `sort_values` method sorts a Series values in order.
# By default, pandas applies an ascending sort (smallest to largest).
# Customize the sort order with the `ascending` parameter.
google.sort_values()
google.sort_values(ascending=True)
google.sort_values(ascending=False)
google.sort_values(ascending=False).head()


pokemon.sort_values()
pokemon.sort_values(ascending=True)
pokemon.sort_values(ascending=False)
pokemon.sort_values(ascending=False).tail()

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd

# Below, we have a list of delicious tortilla chip flavors
flavors = ["Spicy Sweet Chili", "Cool Ranch", "Nacho Cheese", "Salsa Verde"]

# Create a new Series object, passing in the flavors list defined above
# Assign it to a 'doritos' variable. The resulting Series should look like this:
#
# 
#   0    Spicy Sweet Chili
#   1           Cool Ranch
#   2         Nacho Cheese
#   3          Salsa Verde
#   dtype: object
doritos=pd.Series(flavors)

# Below, sort the doritos Series in descending order.
# Assign the sorted a Series to a 'sorted_doritos' variable.
# The sorted Series should like this:
#
#   0    Spicy Sweet Chili
#   3          Salsa Verde
#   2         Nacho Cheese
#   1           Cool Ranch
#   dtype: object
sorted_doritos=pd.Series(flavors).sort_values(ascending=False)

```

## Series - sort_index Method
```xml
# import csv file
pokemon = pd.read_csv("pokemon.csv", index_col="Name").squeeze("columns")
pokemon.head()

# The `sort_index` method sorts a Series by its index.
pokemon.sort_index()

The `sort_index` method also accepts an `ascending` parameter to set sort order.
pokemon.sort_index(ascending=True)
pokemon.sort_index(ascending=False)

```

## Dataframes - Creation
```xml

Dataframes -> are two Dimensional

Create a dataframe: 
car_data=pd.DataFrame({"Car Make": series, "Color": colors })
car_data

Export a dataframe to csv file: 
car_sales.to_csv("exported_car_sales.csv", index=False) -> 
with false, excel's row index will not be imported as a column 

Import dataframe directly from URL: 
heart_disease = 
  pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")


```

## Descibe Data
```xml

# Tab to find all options available on data
car_sales.<press tab key>


# Describe the attribute type
car_sales.dtypes

# List the columns
car_sales.columns

# List the index of the row
# The index of a DataFrame is a series of labels that identify each row.
car_sales.index


# Statical information about dataframe
car_sales.describe()

# Gives complete information of the dataframe 
car_sales.info()

# Finding mean 
car_sales.mean(numeric_only=True)

#Finding sum 
car_sales.sum()

#Finding sum of single column
car_sales["Doors"].sum()

# Finding the length of the dataframe
len(car_sales)

```

## View and select data 
```xml
# Returns the top 5 rows of the dataset, this is the default option
car_sales.head() 

# Returns the top 7 rows of the dataset 
car_sales.head(7)

# Returns the bottom 5 rows of the dataset, this is the default option
car_sales.tail()

# Returns the bottom 7 rows of the dataset 
car_sales.tail(7)

# Loc refers to the index number of the record in the dataset 
car_sales.loc[3]

# iLoc refers to the postiion of the record in the dataset 
car_sales.iloc[3]

# Slicing with iLoc - will give rows between 0 to 3 (exclusive)
car_sales.iloc[:3]

# Two ways of selecting column
car_sales.Make
car_sales["Make"]

# Will select rows that have "Make" equal to "Toyota"
car_sales[car_sales["Make"]=="Toyota"]

# Will select rows that have "Odometer (KM)" greater than 100000
car_sales[car_sales["Odometer (KM)"]>100000]

# Multiple conditions in a dataframe 
car_sales[
  (car_sales["Make"]=="Toyota") 
    & 
  (car_sales["Odometer (KM)"]>100000)
]

# Cross tab between 2 different columns 
pd.crosstab(car_sales['Make'], car_sales['Doors'])

# Group by make and find mean of numeric columns 
car_sales.groupby(["Make"]).mean(numeric_only=True)

# To plot a graph on a column value 
car_sales["Odometer (KM)"].plot()

# To draw histogram - to visulaize the spread of data 
car_sales["Odometer (KM)"].hist()

# Change Price column to integers
car_sales["Price"] = 
  car_sales["Price"].str.replace('[\\$\\,\\.]', '', regex=True).astype(int)

# Divide by 100 as decimal was removed in previous step 
# and round result by 2 decimials
car_sales["Price"] = car_sales["Price"].div(100).round(2)


```


## Manipulating Data
```xml

# Printing the values of a string in lower case 
car_sales["Make"].str.lower()


# Import a file into panda dataframe
car_sales_missing=pd.read_csv("car-sales-missing-data.csv")

# Fill the missing values in Odometer with mean of available values
car_sales_missing["Odometer"]=
  car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean())

# Drop all the rows with NA values
car_sales_missing_droped=car_sales_missing.dropna()


# Create a new column
seats_column=pd.Series([5,5,5,5,5])
car_sales["Seats"]=seats_column
car_sales
car_sales["Seats"]=car_sales["Seats"].fillna(5) -> This will fix the 'na' values 
car_sales


# Create column from list 
fuel_economy=[5.9, 6.2,3.6, 2.1,9.7, 8.2, 3.7, 5.4, 6.7, 9.3]
car_sales["Fuel Per 100Km"]=fuel_economy
car_sales


# Column computation 
car_sales["Total fuel used"]=
  car_sales["Odometer (KM)"]/100 * car_sales["Fuel Per 100Km"]

# Create column from a single value
car_sales["No of Wheels"]=4
car_sales["Pass Through Safety"]= True


# Remove or drop a column
car_sales=car_sales.drop("Pass Through Safety", axis=1)


# Randamonize the data order by taking a % of data
car_sales.sample(frac=0.5) -> here we have taken 50% data as sample
car_sales.sample(frac=1) -> will shuffle 100% of the data

# Reset back the randomized column index data
car_sales_shuffled.reset_index(drop=True, inplace=True)

# Apply Labda functions to change column values 
# This creates a new column called Odometer (Miles) 
# and converts kilometer into miles
car_sales_shuffled["Odometer (Miles)"]=car_sales_shuffled["Odometer (KM)"]
  .apply(lambda x : x/1.4)


```

## Pandas - SQL Connection (MySQL)
```xml
Requirement: Make sure MySQL is installed and valid connection establised.

# Enviroment import requirement 
install mysql-connector-python

# Run the following code:  
import mysql.connector as connection
import pandas as pd
try:
    mydb = connection.connect(host="localhost", database = 'myworld',user="myworld", passwd="myworld123",use_pure=True)
    query = "select * from student;"
    result_dataFrame = pd.read_sql(query,mydb)
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))

This runs and makes a successful connection and fetches data in 
result_dataFrame

But we will get a warning as 
" UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) 
or database string URI or sqlite3 DBAPI2 connection. 
Other DBAPI2 objects are not tested. Please consider using SQLAlchemy."

# Much cleaner approach
# To avoid this import the following libraries: 
import sqlalchemy
import pymysql


# Run the following code:
import pandas as pd
from sqlalchemy import create_engine
connect_string = 'mysql+pymysql://myworld:myworld123@localhost/myworld'
sql_engine = create_engine(connect_string)
query = "select * from student"
df = pd.read_sql_query(query, sql_engine)
df 

```

### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/
https://pandas.pydata.org/pandas-docs/stable/
https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html
https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html

https://www.udemy.com/course/data-analysis-with-pandas


```