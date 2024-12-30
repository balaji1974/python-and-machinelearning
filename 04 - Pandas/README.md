
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

# Pandas - Series 
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
# import csv file with index column as name 
pokemon = pd.read_csv("pokemon.csv", index_col="Name").squeeze("columns")
pokemon.head()

# The `sort_index` method sorts a Series by its index.
pokemon.sort_index()

The `sort_index` method also accepts an `ascending` parameter to set sort order.
pokemon.sort_index(ascending=True)
pokemon.sort_index(ascending=False)

1. Exercise
-----------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work
import pandas as pd

# Below, we have a list of delicious drink flavors
# We create a sorted Series of strings and assign it to a 'gatorade' variable 
flavors = ["Red", "Blue", "Green", "Orange"]
gatorade = pd.Series(flavors).sort_values()

# I'd like to return the Series to its original order 
# (sorted by the numeric index in ascending order). 
# Sort the gatorade Series by index.
# Assign the result to an 'original' variable.
original=gatorade.sort_index()

2. Exercise
-----------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd

# This challenge includes a coffee.csv with 2 columns: 
# Coffee and Calories. Import the CSV. Assign the Coffee
# column to be the index and the Calories column to be the
# Series' values. Assign the Series to a 'coffee' variable.
coffee=pd.read_csv("coffee.csv", index_col="Coffee").squeeze("columns")

# Check whether the coffee 'Flat White' is present in the data.
# Assign the result to a `flat_white` variable
flat_white="Flat White" in coffee

# Check whether the coffee 'Cortado' is present in the data.
# Assign the result to a `cortado` variable
cortado="Cortado" in coffee

# Check whether the coffee 'Blackberry Mocha' is present in the data.
# Assign the result to a `blackberry_mocha` variable
blackberry_mocha='Blackberry Mocha' in coffee

# Check whether the value 221 is present in the data.
# Assign the result to a 'high_calorie' variable.
high_calorie=221 in coffee.values

# Check whether the value 400 is present in the data.
# Assign the result to a 'super_high_calorie' variable.
super_high_calorie=400 in coffee.values 

```

## Series - Extract value by index location
```xml
# import csv file 
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
pokemon.head()

# Use the `iloc` accessor to extract a Series value by its index position.
# `iloc` is short for "index location".
pokemon.iloc[0]
pokemon.iloc[500]
# pokemon.iloc[1500]
pokemon.iloc[[100, 200, 300]]
# pokemon.iloc[[100, 200, 300, 1500]]


# Python's list slicing syntaxes (slices, slices from start, slices to end, etc.) are supported with **Series** objects.
pokemon.iloc[27:36]
pokemon.iloc[0:7]
pokemon.iloc[:7]

pokemon.iloc[700:1010]
pokemon.iloc[700:]
pokemon.iloc[700:5000]

pokemon.iloc[-1]
pokemon.iloc[-10]

pokemon.iloc[-10:-5]
pokemon.iloc[-8:]

```

## Series - Extract value by index label
```xml
# import csv file 
pokemon = pd.read_csv("pokemon.csv", index_col="Name").squeeze("columns")
pokemon.head()

## Extract Series Value by Index Label
# Use the `loc` accessor to extract a Series value by its index label.
pokemon.loc["Bulbasaur"]
pokemon.iloc[0] # both extract the same value

# Pass a list to extract multiple values by index label.
pokemon.loc["Mewtwo"]
pokemon.loc[["Charizard", "Jolteon", "Meowth"]]

# If one index label/position in the list does not exist, Pandas will raise an error.
# pokemon.loc["Digimon"]
# pokemon.loc[["Pikachu", "Digimon"]]

# Exercise
----------

import pandas as pd

# I have a dictionary that maps guitar types to their colors
guitars_dict = {
    "Fender Telecaster": "Baby Blue",
    "Gibson Les Paul": "Sunburst",
    "ESP Eclipse": "Dark Green"
}

# Create a new Series object, passing in the guitars_dict dictionary as the data source.
# Assign the resulting Series to a "guitars" variable.
guitars=pd.Series(guitars_dict)

# Access the value for the index position of 0 within the "guitars" Series.
# Assign the value to a "fender_color" variable.
fender_color=guitars[0]

# Access the value for the index label of "Gibson Les Paul" in the "guitars" Series.
# Assign the value to a "gibson_color" variable.
gibson_color=guitars.loc["Gibson Les Paul"]

# Access the value for the index label of "ESP Eclipse" in the "guitars" Series.
# Assign the value to a "esp_color" variable.
esp_color=guitars.loc["ESP Eclipse"]

```

## Series - Get Method
```xml
# import csv file
pokemon = pd.read_csv("pokemon.csv", index_col="Name").squeeze("columns")
pokemon.head()

# The `get` method extracts a Series value by index label. 
# It is an alternative option to square brackets.
pokemon.get("Moltres")
pokemon.loc["Moltres"]
# pokemon.loc["Digimon"]
pokemon.get("Digimon")

# The `get` method's second argument sets the fallback value to 
# return if the label/position does not exist.
pokemon.get("Digimon", "Nonexistent")
pokemon.get("Moltres", "Nonexistent")
pokemon.get(["Moltres", "Digimon"], "One of the values in the list was not found")

```

## Series - Overwrite a Series Value
```xml
# Use the `loc/iloc` accessor to target an index label/position, 
# then use an equal sign to provide a new value.

# import from csv file
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
pokemon.head()


pokemon.iloc[0] = "Borisaur"
pokemon.head()

pokemon.iloc[[1, 2, 4]] = ["Firemon", "Flamemon", "Blazemon"]
pokemon.loc["Bulbasaur"] = "Awesomeness"
pokemon.head()

pokemon.iloc[1] = "Silly"
pokemon.head()

```

## Series - Copy Method
```xml
# A copy is a duplicate/replica of an object.
# Changes to a copy do not modify the original object.
# A view is a different way of looking at the same data.
# Changes to a view do modify the original object.
# The `copy` method creates a copy of a pandas object.

# import from csv file - first into a dataframe and 
# then into a series using the copy methid
pokemon_df = pd.read_csv("pokemon.csv", usecols=["Name"])
pokemon_series = pokemon_df.squeeze("columns").copy()

# display the dataframe
pokemon_df

# change the series
pokemon_series[0] = "Whatever"

# look at the changed element
pokemon_series.head()

# display the data frame 
pokemon_df


```

## Series - Math Methods on Series Objects
```xml
# import from csv file
google = pd.read_csv("google_stock_price.csv", usecols=["Price"]).squeeze("columns")
google.head()

# The `count` method returns the number of values in the Series. 
# It excludes missing values; the `size` attribute includes missing values.
google.count()

# The `sum` method adds together the Series's values.
google.sum()

# The `product` method multiplies together the Series's values.
google.product() # will return INF (infinity) error as the value is too big
pd.Series([1, 2, 3, 4]).product()

# The `mean` method calculates the average of the Series's values.
google.mean()

# The `std` method calculates the standard deviation of the Series's values.
google.std()

# The `max` method returns the largest value in the Series.
google.max()

# The `min` method returns the smallest value in the Series.
google.min()

# The `median` method returns the median of the Series (the value in the middle).
google.median()

# The `mode` method returns the mode of the Series (the most frequent value).
google.mode()
pd.Series([1, 2, 2, 2, 3]).mode()

# The `describe` method returns a summary with various mathematical calculations.
google.describe()

```

## Series - Math Methods on Series Objects
```xml
# Broadcasting describes the process of applying an arithmetic operation to an array 
# (i.e., a Series), by applying the same operation to every element of the array

# import from csv file
google = pd.read_csv("google_stock_price.csv", usecols=["Price"]).squeeze("columns")
google.head()


# We can combine mathematical operators with a Series to apply the mathematical operation to every value.
# There are also methods to accomplish the same results (`add`, `sub`, `mul`, `div`, etc.)

# This adds 10 to every element of the array
google.add(10)
# This is the same as above
google + 10

# They both multiple 30 to every element of the array
google.sub(30)
google - 30

google.mul(1.25)
google * 1.25
1.25 * google

google.div(2)
google / 2

```

## Series - The value_counts Method
```xml
# import from csv file
pokemon = pd.read_csv("pokemon.csv", index_col="Name").squeeze("columns")
pokemon.head()

# The `value_counts` method returns the number of times each unique value occurs in the Series.
pokemon.value_counts()
pokemon.value_counts(ascending=True)

# The `normalize` parameter returns the relative frequencies or 
# percentage of times that value occurs instead of the counts.
pokemon.value_counts(normalize=True)
pokemon.value_counts(normalize=True) * 100

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd

# We have a hot_dogs.csv CSV file with 2 columns: Year and Winner.
# The dataset stores the winner of the world-famous Nathan's Hot Dog Eating
# contest for each year since 1967. You can explore the data by clicking into 
# the CSV file on the left.
#
# Import the CSV file into a pandas Series object
# The Series should have the standard pandas numeric index
# The Series values should be the string values from the "Winner" column
# Assign the Series object to a "hot_dogs" variable
hot_dogs=pd.read_csv("hot_dogs.csv",usecols=["Winner"]).squeeze("columns")

# I'm curious how many times each winner has won the hot dog-eating contest.
# Create a new Series that shows each person's name (index labels) 
# and the number of times they've won (the values). What method can
# help you generate this Series?
# Asssign the Series to a "names_and_wins" variable.
names_and_wins=hot_dogs.value_counts()

```

## Series - The apply Method
```xml
# import from csv file
pokemon = pd.read_csv("pokemon.csv", usecols=["Name"]).squeeze("columns")
pokemon.head()

# The `apply` method accepts a function. It invokes that function on every `Series` value.
# In the below example it applies the len function to every element of the series
pokemon.apply(len)

# below function counts 'a' in every element of the series
def count_of_a(pokemon):
    return pokemon.count("a")

pokemon.apply(count_of_a)

```

## Series - The map Method
```xml
## The `map` method "maps" or connects each Series values to another value.

# import from csv file
pokemon = pd.read_csv("pokemon.csv", index_col="Name").squeeze("columns")
pokemon

# We can pass the method a dictionary or a Series. Both types connects keys to values.
attack_powers = {
    "Grass": 10,
    "Fire": 15,
    "Water": 15,
    "Fairy, Fighting": 20,
    "Grass, Psychic": 50
}
attack_powers

# The `map` method uses our argument to connect or bridge together the values.
pokemon.map(attack_powers)


# Does the same thing as above, we pass Series instead of dictionary
attack_powers = pd.Series({
    "Grass": 10,
    "Fire": 15,
    "Water": 15,
    "Fairy, Fighting": 20,
    "Grass, Psychic": 50
})
attack_powers
pokemon.map(attack_powers)

```

# Pandas - Dataframes

## Dataframes - Introduction
```xml
# A DataFrame is a 2-dimensional table consisting of rows and columns.

# import from csv file
# Pandas uses a `NaN` designation for cells that have a missing value. 
# It is short for "not a number". Most operations on `NaN` values will produce `NaN` values.
# Like with a Series, Pandas assigns an index position/label to each DataFrame row.
nba = pd.read_csv("nba.csv")
nba

# Create a sample series 
s = pd.Series([1, 2, 3, 4, 5])
s

# Methods and Attributes between Series and DataFrames
# The DataFrame and Series have common and exclusive methods/attributes.
nba.head()
nba.head(n=5)
nba.head(8)

nba.tail()
nba.tail(n=7)
nba.tail(1)

# Returns the index range  
s.index
nba.index

# Returns the values 
s.values
nba.values

# Returns the dimensions 
s.shape
nba.shape

# Returns the data types 
s.dtypes
nba.dtypes

# The `hasnans` attribute exists only in a Series. 
# It returns true if Series has any missing values
s.hasnans
# nba.hasnans # this attribute does not exist in Dataframe

# The `columns` attribute exists only in a DataFrame.
# It returns the column names for the DataFrame
nba.columns
# s.columns # This attribute does not exist in Series

# Some methods/attributes will return different types of data.
s.axes # This returns the range index 
nba.axes # This returns the range index and column index of the Dataframe

# The `info` method returns a summary of the pandas object.
s.info()
nba.info()

# Differences between Shared Methods
# import from csv 
revenue = pd.read_csv("revenue.csv", index_col="Date")
revenue

# Create a sample series 
s = pd.Series([1, 2, 3])
s.sum(axis="index") # The `sum` method adds a Series's values.


# On a DataFrame, the `sum` method defaults to adding the values 
# by traversing the index (row values). 
revenue.sum()

# The `axis` parameter customizes the direction that we add across. 
revenue.sum(axis="index") # will add the column totals 

# Pass `"columns"` or `1` to add "across" the columns.
revenue.sum(axis="columns") # will give row totals
revenue.sum(axis="columns").sum() # will total rows totals 

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

## Dataframes - Descibe Data
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

## Dataframes - View and select data 
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

## Dataframes - Select One Column from a DataFrame
```xml
# import from csv
nba = pd.read_csv("nba.csv")
nba.head()

# We can use attribute syntax (`df.column_name`) to select a column from a DataFrame. 
nba.Team
nba.Salary
nba.Name
# nba.name # Column name is case sensitive

# Pandas extracts a column from a DataFrame as a Series.
type(nba.Name)

# The syntax will not work if the column name has spaces.
# We can also use square bracket syntax (`df["column name"]`) which will work for any column name.
nba["Team"]
nba["Salary"]

# The Series is a view, so changes to the Series will affect the DataFrame.
# Pandas will display a warning if you mutate the Series. Use the `copy` method to create a duplicate.
names = nba["Name"].copy()
names

names.iloc[0] = "Whatever"

names.head()

nba.head()

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd 

# This challenge includes a cruise_ships.csv with 4 columns: 
# Name, Operator, Year, and Tonnage
# Import the cruise_ships.csv DataFrame and assign it to
# a cruise_ships variable
cruise_ships=pd.read_csv("cruise_ships.csv")

# Extract the "Operator" column from the DataFrame
# and assign it to an "operators" variable.
operators=cruise_ships["Operator"]

# Extract the "Tonnage" column from the DataFrame
# and assign it to an "tonnages" variable.
tonnages=cruise_ships["Tonnage"]

# Extract the "Name" column from the DataFrame
# and assign it to an "cruise_names" variable.
cruise_names=cruise_ships["Name"]

```

## Dataframes - Select Multiple Columns
```xml

# import from csv
nba = pd.read_csv("nba.csv")
nba.head()

# Use square brackets with a list of names to extract multiple DataFrame columns.
nba[["Name", "Team"]]
nba[["Team", "Name"]]
nba[["Salary", "Team", "Name"]]

# Pandas stores the result in a new DataFrame (a copy).
columns_to_select = ["Salary", "Team", "Name"]
nba[columns_to_select]

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd 


# This challenge includes a "chicken_restaurants.csv" dataset with 6 columns:
# Name, Original Location, Year, Headquarters, Locations, Areas Served
# Import the CSV into a DataFrame and assign it to a "chicken" variable.
chicken=pd.read_csv("chicken_restaurants.csv")

# Extract the "Year" and "Locations" columns (in that order) into
# their own DataFrame. Assign the DataFrame to a "years_and_locations" variable.
years_and_locations=chicken[["Year","Locations"]]

# Extract the "Locations", "Name", and "Headquarters" columns (in that order)
# into their own DataFrame. Assign the DataFrame to a 
# "interesting_facts" variable.
interesting_facts=chicken[["Locations","Name","Headquarters"]]

```

## Dataframes - Add New Column to DataFrame
```xml

# import from csv
nba = pd.read_csv("nba.csv")
nba.head()

# Use square bracket extraction syntax with an equal sign 
# to add a new Series to a DataFrame.
nba["Sport"] = "Basketball"


# The insert method allows us to insert an element at a specific column index.
# nba.insert(loc=3, column="Sport", value="Basketball") # Another way 
nba["Salary"] * 2
nba["Salary"].mul(2)

nba["Salary Doubled"] = nba["Salary"].mul(2)

# On the right-hand side, we can reference an existing DataFrame column and 
# perform a broadcasting operation on it to create the new Series.
nba["Salary"] - 5000000
nba["Salary"].sub(5000000)

nba["New Salary"] = nba["Salary"].sub(5000000)

```

## Dataframes - A Review of the value_counts Method
```xml
# import from csv
nba = pd.read_csv("nba.csv")
nba.head()

# The value_counts method counts the number of times that 
# each unique value occurs in a Series.
nba["Team"].value_counts()

nba["Position"].value_counts()
nba["Position"].value_counts(normalize=True) # This results in relative percentage 
nba["Position"].value_counts(normalize=True) * 100 

nba["Salary"].value_counts()

```

## Dataframes - Drop Rows with Missing Values
```xml
# import csv
nba = pd.read_csv("nba.csv")
nba

# Pandas uses a NaN designation for cells that have a missing value.
# The dropna method deletes rows with missing values. 
# Its default behavior is to remove a row if it has any missing values.
nba.dropna()
nba.dropna(how="any") # same as above - default parameter 

# Pass the how parameter an argument of "all" to delete rows 
# where all the values are NaN.
nba.dropna(how="all")

# The subset parameters customizes/limits the columns that 
# pandas will use to drop rows with missing values.
nba.dropna(subset=["College"])
nba.dropna(subset=["College", "Salary"]) 

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work
import pandas as pd
# The data.csv file contains a dataset of random numbers.
# The dataset has 4 columns: A, B, C, and D.
# Import pandas and use it to parse the CSV.
# Assign the imported DataFrame to a variable called 'data'.
data=pd.read_csv("data.csv")

# Filter the dataset to remove rows where ALL the
# values are missing. Assign the resulting DataFrame
# to a "no_empty_rows" variable.
no_empty_rows=data.dropna(how="all")

# Filter the dataset to remove rows that have a missing value
# in either the "B" or "D" columns.
# Assign the resulting DataFrame to a "result" variable.
result=data.dropna(subset=["B", "D"])

```

## Dataframes - Fill in Missing Values with the fillna Method
```xml
# import csv
nba = pd.read_csv("nba.csv").dropna(how="all")
nba

# The fillna method replaces missing NaN values with its argument.
# The fillna method is available on both DataFrames and Series.
nba.fillna(0)
nba["Salary"] = nba["Salary"].fillna(0)
nba

# An extracted Series is a view on the original DataFrame, 
# but the fillna method returns a copy.
nba["College"] = nba["College"].fillna(value="Unknown")
nba

```

## Dataframes - Manipulating Data
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

## Pandas - The astype Method
```xml
# read csv and fill NA 
nba = pd.read_csv("nba.csv").dropna(how="all")
nba["Salary"] = nba["Salary"].fillna(0)
nba["Weight"] = nba["Weight"].fillna(0)
nba

# The dtypes attribute returns a Series with the DataFrame's columns and their types.
nba.dtypes

# The astype method converts a Series's values to a specified type.
# Pass in the specified type as either a string or the core Python data type.
# Pandas cannot convert NaN values to numeric types, 
# so we need to eliminate/replace them before we perform the conversion.
# There are multiple ways to convert and is listed below: (copy and not permenant change)
nba["Salary"].astype("int")
nba["Salary"].astype(int)

# To make the change permenant 
nba["Salary"] = nba["Salary"].astype(int)
nba["Weight"] = nba["Weight"].astype(int)
nba

# read csv
nba = pd.read_csv("nba.csv").dropna(how ="all")
nba.tail()

# The nunique method will return a Series with the number of unique values in each column.
nba["Team"].nunique()
nba.nunique()
nba.info() # Check and note the memory consumption 

# The category type is ideal for columns with a limited number of unique values.
# With categories, pandas does not create a separate value in memory for each "cell". 
# Rather, the cells point to a single copy for each unique value.
# The total memory of the dataframe reduces by converting certain columns to category 
nba["Position"] = nba["Position"].astype("category")
nba["Team"] = nba["Team"].astype("category")
nba.info() # Check that the size has been reduced by converting to category 

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Import the pandas library and assign it an alias of 'pd'.
import pandas as pd

# Import the health.csv file and assign it to a 'health' variable.
# The resulting DataFrame will have 3 columns: Weight, Height, and Blood Type
health = pd.read_csv("health.csv")
# Convert the values in the Weight Series to strings and overwrite the original column
health["Weight"]=health["Weight"].astype(str)
# Convert the values in the Height Series to integers and overwrite the original column
health["Height"]=health["Height"].astype(int)
# Convert the values in the Blood Type Series to categories and overwrite the original column
health["Blood Type"]=health["Blood Type"].astype("category")


```

## Pandas - Sort a DataFrame with the sort_values method
```xml
# read csv
nba = pd.read_csv("nba.csv")
nba.tail()

# The sort_values method sorts a DataFrame by the values in one or more columns. 
# The default sort is an ascending one (alphabetical for strings).
# The first parameter (by) expects the column(s) to sort by.
# If sorting by a single column, pass a string with its name.
nba.sort_values("Name")
nba.sort_values(by="Name")

# The ascending parameter customizes the sort order.
nba.sort_values(by="Name", ascending=True)
nba.sort_values(by="Name", ascending=False)
nba.sort_values("Salary")
nba.sort_values("Salary", ascending=False)

# The na_position parameter customizes where pandas places NaN values.
nba.sort_values("Salary", na_position="last")
nba.sort_values("Salary", na_position="first")
nba.sort_values("Salary", na_position="first", ascending=False)

# To sort by multiple columns, pass the by parameter a list of column names. 
# Pandas will sort in the specified column order (first to last).
nba.sort_values(by=["Team", "Name"])

# Pass the ascending parameter a Boolean to sort all columns in a consistent order 
# (all ascending or all descending).
nba.sort_values(by=["Team", "Name"], ascending=True)
nba.sort_values(by=["Team", "Name"], ascending=False)

# Pass ascending a list to customize the sort order per column. 
# The ascending list length must match the by list.
nba.sort_values(by=["Team", "Name"], ascending=[True, False])

# Another set of examples
nba.sort_values(["Position", "Salary"])
nba.sort_values(["Position", "Salary"], ascending=True)
nba.sort_values(["Position", "Salary"], ascending=False)
nba.sort_values(["Position", "Salary"], ascending=[True, False])
nba.sort_values(["Position", "Salary"], ascending=[False, True])

# Assign the sorted values to the dataframe back
nba = nba.sort_values(["Position", "Salary"], ascending=[False, True])
nba


# read data once again from csv
nba = pd.read_csv("nba.csv")
nba = nba.sort_values(["Team", "Name"])
nba

# Sort a DataFrame by its Index
# The sort_index method sorts the DataFrame by its index positions/labels.
nba.sort_index()
nba.sort_index(ascending=True)
nba.sort_index(ascending=False)
nba = nba.sort_index(ascending=False)
nba

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd 

# This challenge includes a s&p500.csv with 6 columns: 
# Symbol, Security, Sector, Industry, HQ, Founded
# Import the s&p500.csv DataFrame and assign it to
# a companies variable.
companies=pd.read_csv("s&p500.csv")

# Sort the DataFrame by the values in the "Industry" column in ascending order
# Assign the new DataFrame to a "by_industry" variable.
by_industry=companies.sort_values(by="Industry")

# Sort the DataFrame by the values in the "HQ" column in descending order
# Assign the new DataFrame to a "by_headquarters_descending" variable.
by_headquarters_descending=companies.sort_values(by="HQ", ascending=False)

# Sort the DataFrame by two conditions:
#  - by the values in the "Sector" column in descending order
#  - THEN by the values in the "Security" column in ascending order
# Assign the new DataFrame to a 'by_sector_and_security' variable
by_sector_and_security=companies.sort_values(by=["Sector","Security"],ascending=[False,True])

```

## Dataframe - Rank Values with the rank Method
```xml
# read csv
nba = pd.read_csv("nba.csv").dropna(how="all")
nba["Salary"] = nba["Salary"].fillna(0).astype(int)
nba

# The rank method assigns a numeric ranking to each Series value.
# Pandas will assign the same rank to equal values and 
# create a "gap" in the dataset for the ranks.
nba["Salary"].rank()
nba["Salary"].rank(ascending=True)
nba["Salary"].rank(ascending=False).astype(int)

nba["Salary Rank"] = nba["Salary"].rank(ascending=False).astype(int)
nba

nba.sort_values("Salary", ascending=False).head(10)

```

## Dataframe - Date/Time and type conversions (Memory Optimization)
```xml
# import data 
employees = pd.read_csv("employees.csv")
employees.info()
employees.head()

# The pd.to_datetime method converts a Series to hold datetime values.
# The format parameter informs pandas of the format that the times are stored in.
# We pass symbols designating the segments of the string. 
# For example, %m means "month" and %d means day.
employees["Start Date"] = pd.to_datetime(employees["Start Date"], format="%m/%d/%Y")

# The dt attribute reveals an object with many datetime-related attributes and methods.
# The dt.time attribute extracts only the time from each value in a datetime Series.
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time

# Use the astype method to convert the values in a Series to another type.
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.info()

# The parse_dates parameter of read_csv is an alternate way to parse strings as datetimes.
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

```

## Dataframe - Filter A DataFrame Based On A Condition 
```xml

# import data and convert dates/time, boolean and category 
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

import datetime as dt
employees[employees["Gender"] == "Male"]
employees[employees["Team"] == "Finance"]

on_finance_team = employees["Team"] == "Finance"
employees[on_finance_team]

employees[employees["Senior Management"]].head()
employees[employees["Salary"] > 110000]
employees[employees["Bonus %"] < 1.5]
employees[employees["Start Date"] < "1985-01-01"]
employees[employees["Last Login Time"] < dt.time(12, 0, 0)

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd

# This challenge includes a the_office.csv dataset.
# It is a listing of all episodes in the popular American sitcom The Office.
# The dataset has 7 columns:
# Season, Episode, Name, Director, Writer, Airdate, Viewership
# Import the the_office.csv fille into a DataFrame. 
# Tell pandas to parse the values in the Airdate column as datetime values.
# Finally, assign the imported DataFrame to an 'office' variable.
office = pd.read_csv("the_office.csv", parse_dates=["Airdate"])

# CHALLENGE 1:
# Find all episodes with a Writer of "Greg Daniels"
# Assign the resulting DataFrame to a 'written_by_greg' variable.
written_by_greg=office[office["Writer"]=="Greg Daniels"]

# CHALLENGE 2:
# Find all episodes BEFORE season 8 (not including season 8)
# Assign the resulting DataFrame to a 'good_episodes' variable
good_episodes=office[office["Season"]<8]

# CHALLENGE 3:
# Find all episodes that aired before 1/1/2008.
# Assign the resulting DataFrame to an 'early_episodes' variable.
early_episodes=office[office["Airdate"]<"1/1/2008"]


```
## Dataframe - Filter with More than One Condition using (AND) and (OR)
```xml

# import data and convert dates/time, boolean and category 
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

# AND Logic
# True, True -> True
# True, False -> False
# False, True -> False
# False, False -> False

# Use the & operator in between two Boolean Series to filter by both conditions.

# female employees who work in Marketing who earn over $100k a year
is_female = employees["Gender"] == "Female"
is_in_marketing = employees["Team"] == "Marketing"
salary_over_100k = employees["Salary"] > 100000

# Pandas needs a Series of Booleans to perform a filter.
# Pass the Boolean Series inside square brackets after the DataFrame.
# We can generate a Boolean Series using a wide variety of operations 
# (equality, inequality, less than, greater than, inclusion, etc)
is_female & is_in_marketing & salary_over_100k
employees[is_female & is_in_marketing & salary_over_100k]

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd

# This challenge includes a the_office.csv dataset.
# It is a listing of all episodes in the popular American sitcom The Office.
# The dataset has 7 columns:
# Season, Episode, Name, Director, Writer, Airdate, Viewership
# Import the the_office.csv file into a DataFrame. 
# Tell pandas to parse the values in the Airdate column as datetime values.
# Finally, assign the imported DataFrame to an 'office' variable.
office=pd.read_csv("the_office.csv", parse_dates=["Airdate"])

# CHALLENGE 1:
# Find all episodes with a Viewership greater than 10
# who are also directed by Jeffrey Blitz
# Assign the resulting DataFrame to a 'jeffs_episodes' variable.
v1=office["Viewership"]>10
v2=office["Director"]=="Jeffrey Blitz"
jeffs_episodes=office[v1 & v2]

# CHALLENGE 2:
# Find all episodes in season 5 that have an episode number
# greater than or equal to 13.
# Assign the resulting DataFrame to a "second_half_of_season_5" variable.
v3=office["Season"]==5
v4=office["Episode"]>=13
second_half_of_season_5=office[v3 & v4]

# CHALLENGE 3:
# Find all episodes that were the 6th episode of their season
# and also aired before 01/01/2010.
# Assign the resulting DataFrame to a "sixth_episodes_of_early_seasons" variable.
v5=office["Episode"]==6
v6=office["Airdate"]<"1/1/2010"
sixth_episodes_of_early_seasons=office[v5 & v6]
-----------------------


# OR logic 
# True, True -> True
# True, False -> True
# False, True -> True
# False, False -> False

# Use the | operator in between two Boolean Series to filter by either condition.

# Employees who are either senior management OR started before January 1st, 1990
is_senior_management = employees["Senior Management"]
started_in_80s = employees["Start Date"] < "1990-01-01"
employees[is_senior_management | started_in_80s]

# First Name is Robert who work in Client Services OR Start Date after 2016-06-01
is_robert = employees["First Name"] == "Robert"
is_in_client_services = employees["Team"] == "Client Services"
start_date_after_june_2016 = employees["Start Date"] > "2016-06-01"
employees[(is_robert & is_in_client_services) | start_date_after_june_2016]
# or 
employees[((employees["First Name"] == "Robert") & (employees["Team"] == "Client Services")) 
  | (employees["Start Date"] > "2016-06-01")]

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd

# This challenge includes a the_office.csv dataset.
# It is a listing of all episodes in the popular American sitcom The Office.
# The dataset has 7 columns:
# Season, Episode, Name, Director, Writer, Airdate, Viewership
# Import the the_office.csv file into a DataFrame. 
# Tell pandas to parse the values in the Airdate column as datetime values.
# Finally, assign the imported DataFrame to an 'office' variable.
office=pd.read_csv("the_office.csv", parse_dates=["Airdate"])

# CHALLENGE 1:
# Find all episodes that were EITHER in Season 4
# OR directed by Harold Ramis
# Assign the resulting DataFrame to a 'season_4_or_harold' variable.
v1=office["Season"]==4 
v2=office["Director"]=="Harold Ramis"
season_4_or_harold=office[v1 | v2]

# CHALLENGE 2:
# Find all episodes that EITHER had a Viewership less than 4
# OR aired on/after January 1st, 2013.
# Assign the resulting DataFrame to a 'low_viewership_or_later_airdate' variable.
v3=office["Viewership"]<4 
v4=office["Airdate"]>='1/1/2013'
low_viewership_or_later_airdate=office[v3 | v4]

# CHALLENGE 3:
# Find all episodes that EITHER the 9th episode of their season
# OR had an episode Name of "Niagara"
# Assign the resulting DataFrame to a 'ninth_or_niagara' variable.
v5=office["Episode"]==9
v6=office["Name"]=="Niagara"
ninth_or_niagara=office[v5 | v6]


```

## Dataframe - The isin Method
```xml
# import data and convert dates/time, boolean and category 
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

# Legal Team or Sales Team or Product Team
legal_team = employees["Team"] == "Legal"
sales_team = employees["Team"] == "Sales"
product_team = employees["Team"] == "Product"
employees[legal_team | sales_team | product_team]

# The isin Series method accepts a collection object like a list, tuple, or Series.
# The method returns True for a row if its value is found in the collection.
target_teams = employees["Team"].isin(["Legal", "Sales", "Product"])
employees[target_teams]

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd 

# This challenge includes a billboard.csv dataset.
# It is a list of the top-selling albums of all time according to Billboard
# The dataset has 5 columns:
# Artist,Album,Released,Genre,Sales
# Import the billboard.csv file into a DataFrame. 
# Assign the imported DataFrame to an 'billboard' variable.
billboard=pd.read_csv("billboard.csv")

# CHALLENGE 1
# Find all records with an Artist of either Michael Jackson,
# Whitney Houston, or Celine Dion. Assign the resulting 
# DataFrame to a "trios" DataFrame.
trios=billboard[billboard["Artist"].isin(["Michael Jackson", "Whitney Houston", "Celine Dion"])]

# CHALLENGE 2
# Find all records with Sales of either 25, 35, or 45
# million copies. Note that the 'Sales' column's integers
# reflect album sales in millions. Assign the resulting DataFrame
# to a 'fives' DataFrame
fives=billboard[billboard["Sales"].isin([25,35,45])]

# CHALLENGE 3
# Find all records released in either 1979, 1989, or 1999.
# Assign the resulting DataFrame to a 'end_of_decade' DataFrame.
end_of_decade=billboard[billboard["Released"].isin([1979,1989,1999])]

```

## Dataframe - The isnull, notnull, between Methods
```xml
# import data and convert dates/time, boolean and category 
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

# The isnull method returns True for NaN values in a Series.
employees[employees["Team"].isnull()]

# The notnull method returns True for present values in a Series.
employees[employees["Team"].notnull()]

employees[employees["First Name"].isnull() & employees["Team"].notnull()]

# The between method returns True if a Series value is found within its range.
import datetime as dt
employees[employees["Salary"].between(60000, 70000)]
employees[employees["Bonus %"].between(2.0, 5.0)]
employees[employees["Start Date"].between("1991-01-01", "1992-01-01")]
employees[employees["Last Login Time"].between(dt.time(8, 30), dt.time(12, 0))]

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd

# This challenge includes a weather.csv dataset. It includes
# temperature measurements (high and low) for the month of April.
# The dataset has 3 columns:
# Day, Low Temp, High Temp
# Import the weather.csv file into a DataFrame. 
# Tell pandas to parse the values in the Day column as datetime values.
# Finally, assign the imported DataFrame to a 'weather' variable.
weather=pd.read_csv("weather.csv", parse_dates=["Day"])

# CHALLENGE 1
# I want to see the temperature for the days between April 15, 2022 and April 22, 2022.
# Extract those rows to a new DataFrame and assign it to a 'week_of_weather' variable
week_of_weather=weather[weather["Day"].between("2022-04-15","2022-04-22")]

# CHALLENGE 2
# Extract the rows where the value in the Low Temp column is between 30 and 50.
# Assign the new DataFrame to a "cold_days" variable.
cold_days=weather[weather["Low Temp"].between(30,50)]

# CHALLENGE 3
# Extract the rows where the value in the High Temp column is between 50 and 75.
# Assign the new DataFrame to a "warm_days" variable.
warm_days=weather[weather["High Temp"].between(50,75)]

```

## Dataframe - The duplicated Method
```xml
# import data and convert dates/time, boolean and category 
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

# The duplicated method returns True if a Series value is a duplicate.
# Pandas will mark one occurrence of a repeated value as a non-duplicate.
employees[employees["First Name"].duplicated()]

# Use the keep parameter to designate whether the first or last occurrence 
# of a repeated value should be considered the "non-duplicate".
employees[employees["First Name"].duplicated(keep="first")]
employees[employees["First Name"].duplicated(keep="last")]

# Pass False to the keep parameter to mark all occurrences of repeated values as duplicates.
employees[employees["First Name"].duplicated(keep=False)]

# Use the tilde symbol (~) to invert a Series's values. 
# Trues will become Falses, and Falses will become trues.
employees[~employees["First Name"].duplicated(keep=False)]

```

## Dataframe - The drop_duplicates Method
```xml
# import data and convert dates/time, boolean and category 
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

# The drop_duplicates method deletes rows with duplicate values.
# By default, it will remove a row if all of its values are shared with another row.
employees.drop_duplicates()

# Remove a row if a column values is shared with same column value in another row.
# By default the first occurance is marked as non duplicate and kept
employees.drop_duplicates("Team")
employees.drop_duplicates("Team", keep="first")

# Keep last will mark the last occurance as non duplicate and will be kept
employees.drop_duplicates("Team", keep="last")

# Keep false will remove all occurance of the duplicates 
employees.drop_duplicates("Team", keep=False)
employees.drop_duplicates("First Name", keep=False)

# The subset parameter configures the columns to look for duplicate values within.
# Pass a list to subset parameter to look for duplicates across multiple columns.
employees.drop_duplicates(["Senior Management", "Team"]).sort_values("Team")
employees.drop_duplicates(["Senior Management", "Team"], keep="last").sort_values("Team")

```

## Dataframe - The unique and nunique Methods
```xml
# import data and convert dates/time, boolean and category 
employees = pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")
employees["Last Login Time"] = pd.to_datetime(employees["Last Login Time"], format="%H:%M %p").dt.time
employees["Senior Management"] = employees["Senior Management"].astype(bool)
employees["Gender"] = employees["Gender"].astype("category")
employees.head()

# The unique method on a Series returns a collection of its unique values. 
# The method does not exist on a DataFrame.
# The return object may vary between Catagory or DArray 
employees["Gender"].unique()
type(employees["Gender"].unique())
employees["Team"].unique()
type(employees["Team"].unique())

# The nunique method returns a count of the number of unique values in the Series/DataFrame.
employees["Team"].nunique()

# The dropna parameter configures whether to include or exclude missing (NaN) values in the count.
employees["Team"].nunique(dropna=True)
employees["Team"].nunique(dropna=False)

# There is no unique method on the dataframe level 
# employees.unique() # This will return error 
# But nunique return the count of unique records in each column 
employees.nunique()
```



# Pandas - Data Import
## Pandas - Import from csv
```xml
# read csv and fill NA 
nba = pd.read_csv("nba.csv").dropna(how="all")

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
# Connection String -> mysql+pymysql://<user_id>:<passwd>@<host_name/ip>/<db_name>
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
