
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

## Dataframe - The set_index and reset_index Methods
```xml
# import panda
import pandas as pd

# import data
bond = pd.read_csv("jamesbond.csv")
bond.head()

# The index serves as the collection of primary identifiers/labels/entrypoints for the rows.
# The fastest way to extract a row is from a sorted index by position/label.
# Pandas uses index labels/values when merging different objects together.
# The set_index method sets an existing column as the index of the DataFrame.
bond = bond.set_index("Film")
# bond = pd.read_csv("jamesbond.csv", index_col="Film") # same as above 
bond.head()

# The reset_index method will remove the index that was set before
# and reset it to the standand ascending column index (default by panda)
# combining it with set index will set a new index by dropping the old index 
bond = bond.reset_index().set_index("Year")
bond.head()

```

## Dataframe - Retrieve Rows by Index Position with iloc Accessor
```xml
# import data
bond = pd.read_csv("jamesbond.csv")
bond.head()

# The iloc accessor retrieves one or more rows by index position.
# Provide a pair of square brackets after the accessor.
# iloc accepts single values, lists, and slices.

bond.iloc[5]
bond.iloc[[15, 20]]
bond.iloc[4:8]
bond.iloc[0:6]
bond.iloc[:6]

bond.iloc[20:]

```

## Dataframe - Retrieve Rows by Index Label with loc Accessor
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film")
bond.head()

# The loc accessor retrieves one or more rows by index label.
# Provide a pair of square brackets after the accessor.

bond.loc["Goldfinger"]
bond.loc["GoldenEye"]
bond.loc["Casino Royale"]
# bond.loc["Sacred Bond"]

bond.loc[["Octopussy", "Moonraker"]]
bond.loc[["Moonraker", "Octopussy"]]
bond.loc[["Moonraker", "Octopussy", "Casino Royale"]]
# bond.loc[["Moonraker", "Octopussy", "Casino Royale", "Gold Bond"]]

bond.loc["Diamonds Are Forever":"Moonraker"]
bond.loc["Moonraker":"Diamonds Are Forever"]

bond.loc["GoldenEye":]
bond.loc[:"On Her Majesty's Secret Service"]

# bond.loc[:"Casino Royale"] # Since it has 2 records this cannot be done
# bond.loc["Casino Royale":] # Since it has 2 records this cannot be done

```

## Dataframe - Second Arguments to loc and iloc Accessors
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# The second value inside the square brackets targets the columns.
# The loc requires labels for rows and columns.
bond.loc["Diamonds Are Forever", "Director"]
bond.loc[["Octopussy", "GoldenEye"], "Director"]
bond.loc[["Octopussy", "GoldenEye"], "Director":"Budget"]
bond.loc["GoldenEye":"Octopussy", "Director":"Budget"]
bond.loc["GoldenEye":"Octopussy", ["Actor", "Bond Actor Salary", "Year"]]

# The iloc requires numeric positions for rows and columns.
bond.iloc[0, 2]

bond.iloc[3, 5]

bond.iloc[[0, 2], 3]
bond.iloc[[0, 2], [3, 5]]

bond.iloc[:7, :3]

```

## Dataframe - Overwrite Value in a DataFrame
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# Use the iloc or loc accessor on the DataFrame to target a value, 
# then provide the equal sign and a new value.
bond.loc["Diamonds Are Forever", "Actor"] = "Sir Sean Connery"
bond

```

## Dataframe - Overwrite Multiple Values in a DataFrame
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# The replace method replaces all occurrences of a Series value 
# with another value (think of it like "Find and Replace").
bond["Actor"] = bond["Actor"].replace("Sean Connery", "Sir Sean Connery")

# To overwrite multiple values in a DataFrame, 
# remember to use an accessor on the DataFrame itself.
# This must not be used as we will be working on a copy
# bond[bond["Actor"] == "Sean Connery"].loc[:, "Actor"] = "Sir Sean Connery"

# Accessors like loc and iloc can accept Boolean Series. 
# Use them to target the values to overwrite.
is_sean_connery = bond["Actor"] == "Sean Connery"
bond.loc[is_sean_connery, "Actor"] = "Sir Sean Connery"

```

## Dataframe - Rename Index Labels or Columns in a DataFrame
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# The rename method accepts a dictionary for either its columns or index parameters.
# The dictionary keys represent the existing names and the values represent the new names.
bond = bond.rename(columns={ "Year": "Year of Release", "Box Office": "Revenue" })
bond.head()
# or
swaps = {
    "Dr. No": "Dr No",
    "GoldenEye": "Golden Eye",
    "The World Is Not Enough": "Best Bond Movie Ever"
}
bond = bond.rename(index=swaps)
bond.head()

# We can replace all columns by overwriting the DataFrame's columns attribute.
bond.columns = ["Year", "Bond Guy", "Camera Dude", "Revenues", "Cost", "Salary"]
bond.head()

# This will not work 
# bond.columns[3] = "The Money"

```

## Dataframe - Delete Rows or Columns from a DataFrame
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# The drop method deletes one or more rows/columns from a DataFrame.
bond.drop(columns=["Box Office", "Budget"])

# Pass the index or columns parameters a list of the column names to remove.
bond.drop(index=["No Time to Die", "Casino Royale"])
bond.drop(index=["No Time to Die", "Casino Royale"], columns=["Box Office", "Budget"])

The pop method removes and returns a single Series (it mutates the DataFrame in the process).
actor = bond.pop("Actor")
actor.head()

# Python's del keyword also removes a single Series.
del bond["Year"]
del bond["Director"]
bond.head()

```

## Dataframe - Create Random Sample with the sample Method
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# The sample method returns a specified one or more random rows from the DataFrame.
bond.sample()
bond.sample(n=5)
bond.sample(n=3, axis="rows")

# Customize the axis parameter to extract random columns.
bond.sample(n=2, axis="columns")
bond.sample(n=2, axis="columns").head()

```

## Dataframe - The nsmallest and nlargest Methods
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# Retrieve the 4 films with the highest/largest Box Office gross
bond.sort_values("Box Office", ascending=False).head(4)

# The nlargest and nsmallest methods are more efficient than sorting the entire DataFrame.
# The nlargest method returns a specified number of rows with the largest values from a given column.
bond.nlargest(n=4, columns="Box Office")
bond["Box Office"].nlargest(4)

# The nsmallest method returns rows with the smallest values from a given column.
bond.nsmallest(3, columns="Bond Actor Salary")
bond["Bond Actor Salary"].nsmallest(3)

```

## Dataframe - Filtering with the where Method
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# Filter
actor_is_sean_connery = bond["Actor"] == "Sean Connery"
bond[actor_is_sean_connery]
bond.loc[actor_is_sean_connery]

# Similar to square brackets or loc, 
# the where method filters the original DataFrame with a Boolean Series.
# Pandas will populate rows that do not match the criteria with NaN values.
# Leaving in the NaN values can be advantageous for certain merge and visualization operations.
bond.where(actor_is_sean_connery)

```

## Dataframe - The apply Method with DataFrames
```xml
# import data and index with Film column
bond = pd.read_csv("jamesbond.csv", index_col="Film").sort_index()
bond.head()

# The apply method invokes a function on every column or every row in the DataFrame.
bond["Actor"].apply(len)
bond.head()

# Pass the uninvoked function as the first argument to the apply method.
# Pass the axis parameter an argument of "columns" to invoke the function on every row.
# Pandas will pass in the row's values as a Series object. 
# We can use accessors like loc and iloc to extract the column's values for that row.
# MOVIE RANKING SYSTEM
#
# CONDITION      -> DESIGNATION
# 80s movie      -> "Great 80's flick"
# Pierce Brosnan -> "The best Bond ever"
# Budget > 100   -> "Expensive movie, fun"
# Others         -> "No comment"

def rank_movie(row):
    year = row.loc["Year"]
    actor = row.loc["Actor"]
    budget = row.loc["Budget"]

    if year >= 1980 and year < 1990:
        return "Great 80's flick!"

    if actor == "Pierce Brosnan":
        return "The best Bond ever!"

    if budget > 100:
        return "Expensive movie, fun"

    return "No comment"

bond.apply(rank_movie, axis="columns")

```

## Working with Text Data - Common String Methods
```xml

# This Module's Dataset
# This module's dataset (chicago.csv) is a collection of public sector employees in the city of Chicago.
# Each row inclues the employee's name, position, department, and salary.
chicago = pd.read_csv("chicago.csv").dropna(how="all") # remove rows with all missing value 
chicago.head() # display first 5 rows
chicago.info() # display information about the dataset 
chicago.nunique() # no. of unique values in each column 
chicago["Department"] = chicago["Department"].astype("category") # make department as category column 

# Apply all together
chicago = pd.read_csv("chicago.csv").dropna(how="all")
chicago["Department"] = chicago["Department"].astype("category")
chicago.head()

# A Series has a special str attribute that exposes an object with string methods.
# Access the str attribute, then invoke the string method on the nested object.
# Most method names will match their Python method equivalents (upper, lower, title, etc).
chicago["Position Title"].str.lower() # lower case
chicago["Position Title"].str.upper() # upper case
chicago["Position Title"].str.title() # capatilize every word
chicago["Position Title"].str.len() # length 
chicago["Position Title"].str.title().str.len() # chain multiple str methods
chicago["Position Title"].str.strip() # remove white spaces on left and right of the string
chicago["Position Title"].str.lstrip() # remove white spaces on left of the string
chicago["Position Title"].str.rstrip() # remove white spaces on right of the string

chicago["Department"].str.replace("MGMNT", "MANAGEMENT").str.title() # replace matching string and chain with title method

Excerise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd

# This challenge includes a data.csv dataset.
# The dataset has 3 columns: Name, Title, Cryptocurrency
# Import the dataset and assign the DataFrame to a "data" variable.
data = pd.read_csv("data.csv")


# Unfortunately, some of the text data has been incorrectly exported.

# The strings in the Name column are all lowercased.
# Capitalize the names properly (so the first letter of the person's first and last names are uppercase)
# Then, overwrite the Name column with the new Series.
data["Name"]=data["Name"].str.title()

# Lowercase the strings in the Title column.
# Then, overwrite the Title column with the new Series.
data["Title"]=data["Title"].str.lower()

# Uppercase the strings in the Cryptocurrency column.
# Then, overwrite the Cryptocurrency column with the new Series.
data["Cryptocurrency"]=data["Cryptocurrency"].str.upper()

```

## Working with Text Data - Filtering with String Methods
```xml
# import data and drop 'na' and change 'Department' column as category
chicago = pd.read_csv("chicago.csv").dropna(how="all")
chicago["Department"] = chicago["Department"].astype("category")
chicago.head()

# The str.contains method checks whether a substring exists anywhere in the string.
water_workers = chicago["Position Title"].str.lower().str.contains("water")
chicago[water_workers]

# The str.startswith method checks whether a substring exists at the start of the string.
starts_with_civil = chicago["Position Title"].str.lower().str.startswith("civil")
chicago.loc[starts_with_civil]

# The str.endswith method checks whether a substring exists at the end of the string.
ends_with_iv = chicago["Position Title"].str.lower().str.endswith("iv")
chicago[ends_with_iv]

```

## Working with Text Data - String Methods on Index and Columns
```xml
# import data and drop 'na', change 'Department' column as category and add 'Name' as index column
chicago = pd.read_csv("chicago.csv", index_col="Name").dropna(how="all").sort_index()
chicago["Department"] = chicago["Department"].astype("category")
chicago.head()

# Use the index and columns attributes to access the DataFrame index/column labels.
# These objects support string methods via their own str attribute.
chicago.index = chicago.index.str.strip().str.title()
chicago.columns = chicago.columns.str.upper()
chicago.head()

```

## Working with Text Data - The split Method
```xml
# import data and drop 'na' and change 'Department' column as category
chicago = pd.read_csv("chicago.csv").dropna(how="all")
chicago["Department"] = chicago["Department"].astype("category")
chicago.head()

# The str.split method splits a string by the occurrence of a delimiter. 
# Pandas returns a Series of lists.
# The most common first word in our job positions/titles
# Use the str.get method to access a nested list element by its index position.
chicago["Position Title"].str.split(" ").str.get(0).value_counts()

# More Practice with Splits
# Finding the most common first name among the employees
chicago["Name"].str.title().str.split(", ").str.get(1).str.strip().str.split(" ").str.get(0).value_counts()

# The expand parameter returns a DataFrame instead of a Series of lists.
chicago[["Last Name", "First Name"]] = chicago["Name"].str.split(",", expand=True)
chicago.head()

# The n parameter limits the number of splits.
chicago[["Primary Title", "Secondary Title"]] = chicago["Position Title"].str.split(" ", expand=True, n=1)
chicago.head()
```

## Multi-Index - Create a MultiIndex
```xml
# import data and parse date
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d")
bigmac.head()
bigmac.dtypes
bigmac.info()

# A MultiIndex is an index
# with multiple levels or layers.
# import data and parse date, index columns and sort index 
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d", index_col=["Date", "Country"]).sort_index()
bigmac.head()

# Pass the set_index method a list of colum names to create a multi-index DataFrame.
# The order of the list's values will determine the order of the levels.
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d")
bigmac.set_index(keys=["Date", "Country"])
bigmac.set_index(keys=["Country", "Date"]).sort_index()
bigmac.nunique()

bigmac = bigmac.set_index(keys=["Date", "Country"])
bigmac.head()

# Alternatively, we can pass the read_csv function's index_col parameter a list of columns.
bigmac.index.names
bigmac.index[0]
# type(bigmac.index[0])

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd
# This challenge includes a subway_locations.csv dataset.
# It is a listing of all US locations of the Subway fast food restaurant chain.
# The dataset has 4 columns:
# city,state,latitude,longitude
# Import the subway_locations.csv file into a DataFrame. 
# Assign the imported DataFrame to a 'subway' variable.

subway=pd.read_csv("subway_locations.csv")
# CHALLENGE 1:
# Create a MultiIndex with the levels coming from the 'state' and 'city' columns (in that order)
# Assign the resulting DataFrame to a 'multi_df' variable.
# Do not mutate the original DataFrame.
multi_df=subway.set_index(keys=["state", "city"])

# CHALLENGE 2:
# Using your MultiIndex DataFrame, sort the index by both levels' values
# Assign the resulting DataFrame to a 'sorted_multi_df' variable.
# Do not mutate any previous DataFrames.
sorted_multi_df=multi_df.sort_index()

```

## Multi-Index - Extract Index Level Values
```xml
# import data, parse date and index columns and sort index 
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d", index_col=["Date", "Country"]).sort_index()
bigmac.head()

# The get_level_values method extracts an Index with the values from one level in the MultiIndex.
# Invoke the get_level_values on the MultiIndex, not the DataFrame itself.
# The method expects either the level's index position or its name.
bigmac.index.get_level_values("Date")
bigmac.index.get_level_values(0)
bigmac.index.get_level_values("Country")
bigmac.index.get_level_values(1)

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd

# This challenge includes a subway_locations.csv dataset.
# It is a listing of all US locations of the Subway fast food restaurant chain.
# The dataset has 4 columns:
# city,state,latitude,longitude

# CHALLENGE 1:
# Import the subway_locations.csv file into a DataFrame. 
# Create a MultiIndex with the levels coming from the 'state' and 'city' columns (in that order)
# Sort the index (all columns must be sorted) and assign the resulting DataFrame to a 'subway' variable.
subway=pd.read_csv("subway_locations.csv", index_col=["state","city"]).sort_index()

# CHALLENGE 2:
# Using your DataFrame, access the Index holding the values from 
# the 'city' level. Assign the Index to a `city_index` variable.
city_index=subway.index.get_level_values("city")

# CHALLENGE 3:
# Using your DataFrame, access the Index holding the values from 
# the 'state' level. Assign the Index to a `state_index` variable.
state_index=subway.index.get_level_values("state")

```

## Multi-Index - Rename Index Levels
```xml
# import data, parse date and index columns and sort index 
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d", index_col=["Date", "Country"]).sort_index()
bigmac.head()

# Invoke the set_names method on the MultiIndex to change one or more level names.
# Use the names and level parameter to target a nested index at a given level.
bigmac.index.set_names(names="Time", level=0)
bigmac.index.set_names(names="Country", level=1)
bigmac.index.set_names(names=["Time", "Location"])

# Alternatively, pass names a list of strings to overwrite all level names.
# The set_names method returns a copy, so replace the original index to alter the DataFrame.
bigmac.index = bigmac.index.set_names(names=["Time", "Location"])

bigmac.head()

```
## Multi-Index - The sort_index Method on a MultiIndex DataFrame
```xml
# import data, parse date and index columns  
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d", index_col=["Date", "Country"])
bigmac.head()

# Using the sort_index method, we can target all levels or specific levels of the MultiIndex.
# To apply a different sort order to different levels, pass a list of Booleans.
bigmac.sort_index()
bigmac.sort_index(ascending=True)
bigmac.sort_index(ascending=False)

bigmac.sort_index(ascending=[True, False])
bigmac.sort_index(ascending=[False, True])

```

## Multi-Index - Extract Rows from a MultiIndex DataFrame
```xml
# import data, parse date and index columns  
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d", index_col=["Date", "Country"]).sort_index()
bigmac.head()

# A tuple is an immutable list. It cannot be modified after creation.
# Create a tuple with a comma between elements. 
# The community convention is to wrap the elements in parentheses.
1,
1, 2
(1, 2)
type((1, 2)) # this is tuple

# type([1, 2]) # this is list

# The iloc and loc accessors are available to extract rows by index position or label.
bigmac.iloc[2]

# For the loc accessor, pass a tuple to hold the labels from the index levels.
bigmac.loc["2000-04-01"]

bigmac.loc["2000-04-01", "Canada"]
bigmac.loc["2000-04-01", "Price in US Dollars"]

bigmac.loc[("2000-04-01", "Canada")]

start = ("2000-04-01", "Hungary")
end = ("2000-04-01", "Poland")
bigmac.loc[start:end]

bigmac.loc[("2019-07-09", "Hungary"):]

bigmac.loc[("2012-01-01", "Brazil"): ("2013-07-01", "Turkey")]

bigmac.loc[("2012-01-01", "Brazil"): ("2013-07-01", "Turkey"), "Price in US Dollars"]

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

import pandas as pd

# This challenge includes a subway_locations.csv dataset.
# It is a listing of all US locations of the Subway fast food restaurant chain.
# The dataset has 4 columns:
# city,state,latitude,longitude

# CHALLENGE 1:
# Import the subway_locations.csv file into a DataFrame. 
# Use the 'index_col' parameter to attach a MultiIndex with levels
# coming from the 'state' and 'city' columns (in that order)
# Sort the DataFrame by both index levels (in regular ascending order)
# Assign the resulting DataFrame to a 'subway' variable.
subway=pd.read_csv("subway_locations.csv", index_col=["state","city"]).sort_index()

# CHALLENGE 2:
# Extract the row(s) with a 'state' level value of 'OK' and a
# 'city' level value of 'Broken Arrow'. Assign the result to a
# 'broken_arrow' variable.
broken_arrow=subway.loc[("OK","Broken Arrow")]

# CHALLENGE 3:
# Extract the row(s) with a 'state' level value of 'FL' and a
# 'city' level value of 'Winter Park'. Assign the result to a
# 'winter_park' variable.
winter_park=subway.loc[("FL","Winter Park")]

```

## Multi-Index - The transpose Method
```xml
# import data, parse date, index columns and sort index
bigmac = pd.read_csv("bigmac.csv", parse_dates=["Date"], date_format="%Y-%m-%d", index_col=["Date", "Country"]).sort_index()
bigmac.head()


The transpose method inverts/flips the horizontal and vertical axes of the DataFrame.
start = ("2018-01-01", "China")
end = ("2018-01-01", "Denmark")

bigmac.loc[start:end].transpose()

```

## Multi-Index - The stack Method
```xml
# import data, index columns and sort index
world = pd.read_csv("worldstats.csv", index_col=["year", "country"]).sort_index()
world.head()

# The stack method moves the column index to the row index.
# Pandas will return a MultiIndex Series.
# Think of it like "stacking" index levels for a MultiIndex.
world.stack()

type(world.stack()) # This will be a series

world.stack().to_frame() # will convert series to dataframe

```

## Multi-Index - The unstack Method
```xml
# import data, index columns, sort index and stack
world = pd.read_csv("worldstats.csv", index_col=["year", "country"]).sort_index().stack()
world.head()


# The unstack method moves a row index to the column index (the inverse of the stack method).
# By default, the unstack method will move the innermost index.
world.unstack()
world.unstack().unstack().columns

# We can customize the moved index with the level parameter.
# The level parameter accepts the level's index position or its name. It can also accept a list of positions/names.
world.unstack(level=0)
world.unstack(level="year")
world.unstack(level=-3)

world.unstack(level=1)
world.unstack(level="country")
world.unstack(level=-2)
world.unstack(level=2)

world.unstack([1, 0])
world.unstack(["country", "year"])

world.unstack([0, 1])
world.unstack(["year", "country"])

world.unstack(["year", "country"]).sort_index(axis=1)
```

## Multi-Index - The pivot Method
```xml
# import data
sales = pd.read_csv("salesmen.csv")
sales

# The pivot method reshapes data from a tall format to a wide format.
# Ask yourself which direction the data will expand in if you add more entries.
# A tall/long format expands down. A wide format expands out.
# The index parameter sets the horizontal index of the pivoted DataFrame.
# The columns parameter sets the column whose values will be the columns in the pivoted DataFrame.
# The values parameter set the values of the pivoted DataFrame. 
# Pandas will populate the correct values based on the index and column intersections.
sales.pivot(index="Date", columns="Salesman", values="Revenue")

```

## Multi-Index - The melt Method
```xml
# import data
quarters = pd.read_csv("quarters.csv")
quarters

# The melt method is the inverse of the pivot method.
# It takes a 'wide' dataset and converts it to a 'tall' dataset.
# The melt method is ideal when you have multiple columns storing the same data point.
# Ask yourself whether the column's values are a type of the column header. 
# If they're not, the data is likely stored in a wide format.
# The id_vars parameters accepts the column whose values will be repeated for every column.
# The var_name parameter sets the name of the new column for the varying values (the former column names).
# The value_name parameter set the new name of the values column (holding the values from the original DataFrame).
quarters.melt(id_vars="Salesman", var_name="Quarter", value_name="Revenue")

Exercise
--------
# If you see a test failure when checking your solution,
# note that [left] refers to YOUR code while [right]
# refers to the correct code that the computer is comparing
# to your work

# Let's start by importing pandas below
import pandas as pd
# This challenge includes a weather.csv dataset.
# It is a listing of temperatures across 4 seasons in several cities
# The dataset has 4 columns: City,Fall,Winter,Spring,Summer
# Notice that the Fall, Winter, Spring, and Summer columns are storing
# the same data point -- the temperature. This makes them good candidates
# for melting into a single column.
# Import the weather.csv file into a DataFrame. 
# Assign the imported DataFrame to a 'weather' variable.
weather=pd.read_csv("weather.csv")

# CHALLENGE 1:
# Create a new DataFrame using the 'weather' DataFrame by 'melting'
# the season columns' values into a single one. The 4 Season values
# should be stored in a column called 'Season'. The temperature
# values should be stored in a column called 'Temperature'.
# Your goal is a DataFrame that looks like this:
# 
# City      Season   Temperature
# London  Fall   68
# London  Winter   94
# London  Spring   103
# London  Summer   21
# Paris     Fall   46
# Paris     Winter   86
# Paris     Spring   26
# Paris     Summer   70
# ... more rows
#
# Assign this DataFrame to a 'melted' variable
melted=weather.melt(id_vars="City", var_name="Season", value_name="Temperature")

```

## Multi-Index - The pivot_table Method
```xml
# import data
foods = pd.read_csv("foods.csv")
foods.head()

# The pivot_table method operates similarly to the Pivot Table feature in Excel.
# A pivot table is a table whose values are aggregations of groups of values from another table.
# The values parameter accepts the numeric column whose values will be aggregated.
# The index parameter sets the index labels of the pivot table. MultiIndexes are permitted.
foods.pivot_table(values="Spend", index="Gender")

# The aggfunc parameter declares the aggregation function (the default is mean/average).
foods.pivot_table(values="Spend", index="Gender", aggfunc="mean")
foods.pivot_table(values="Spend", index="Gender", aggfunc="sum")
foods.pivot_table(values="Spend", index="Item", aggfunc="sum")
foods.pivot_table(values="Spend", index=["Gender", "Item"], aggfunc="sum")

# The columns parameter sets the column labels of the pivot table. MultiIndexes are permitted.
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns="City", aggfunc="sum")
foods.pivot_table(values="Spend", index="Item", columns=["Gender", "City"], aggfunc="sum")
foods.pivot_table(values="Spend", index="Item", columns=["Gender", "City"], aggfunc="mean")
foods.pivot_table(values="Spend", index="Item", columns=["Gender", "City"], aggfunc="count")
foods.pivot_table(values="Spend", index="Item", columns=["Gender", "City"], aggfunc="max")
foods.pivot_table(values="Spend", index="Item", columns=["Gender", "City"], aggfunc="min")

```

## The GroupBy Object - The groupby Method
```xml
# import data
fortune = pd.read_csv("fortune1000.csv", index_col="Rank")
fortune.head()

Grouping is a way to organize/categorize/group the data based on a column's values.
The groupby method returns a DataFrameGroupBy object. 
It resembles a group/collection of DataFrames in a dictionary-like structure.
The DataFrameGroupBy object can perform aggregate operations on each group within it.
sectors = fortune.groupby("Sector")
sectors

len(sectors)
sectors.size()
sectors.first()
sectors.last()

```

## The GroupBy Object - Retrieve a Group with the get_group Method
```xml
# import data
fortune = pd.read_csv("fortune1000.csv", index_col="Rank")
sectors = fortune.groupby("Sector")
fortune.head(5)

# The get_group method on the DataFrameGroupBy object retrieves 
# a nested DataFrame belonging to a specific group/category.
sectors.get_group("Energy")
sectors.get_group("Technology")

```

## The GroupBy Object - Methods on the GroupBy Object
```xml
# import data
fortune = pd.read_csv("fortune1000.csv", index_col="Rank")
sectors = fortune.groupby("Sector")
fortune.head(5)

# Use square brackets on the DataFrameGroupBy object to "extract" a column from the original DataFrame.
# The resulting SeriesGroupBy object will have aggregation methods available on it.
sectors["Revenue"].sum()
sectors["Employees"].sum()
sectors["Profits"].max()
sectors["Profits"].min()

sectors["Employees"].mean()
sectors["Employees"].min()

# Pandas will perform the calculation on every group within the collection.
# For example, the sum method will sum together the Revenues for every row by group/category.
sectors[["Revenue", "Profits"]].sum()
sectors[["Revenue", "Profits"]].mean()

```

## The GroupBy Object - Grouping by Multiple Columns
```xml
# import data
fortune = pd.read_csv("fortune1000.csv", index_col="Rank")
sectors = fortune.groupby(["Sector", "Industry"])
fortune.head(5)


# Pass a list of columns to the groupby method to group by pairings of values across columns.
# Target a column to retrieve the SeriesGroupBy object, then perform an aggregation with a method.
# Pandas will return a MultiIndex Series where the levels will be the original groups.
sectors.size()
sectors["Revenue"].sum()
sectors["Employees"].mean().head(20)

```

## The GroupBy Object - The agg Method
```xml
# import data
fortune = pd.read_csv("fortune1000.csv", index_col="Rank")
sectors = fortune.groupby("Sector")
fortune.head(5)

# The agg method applies different aggregation methods on different columns.
# Invoke the agg method directly on the DataFrameGroupBy object.
# Pass the method a dictionary where the keys are the columns and 
# the values are the aggregation operations.
sectors.agg({ "Revenue":"sum", "Profits":"max", "Employees":"mean" })

```

## The GroupBy Object - Iterating through Groups
```xml
# import data
fortune = pd.read_csv("fortune1000.csv", index_col="Rank")
sectors = fortune.groupby("Sector")
fortune.head(5)

# The DataFrameGroupBy object supports the apply method (just like a Series and a DataFrame do).
# The apply method invokes a function on every nested DataFrame in the DataFrameGroupBy object.
# It captures the return values of the functions and collects them in a new DataFrame (the return value).
# Find the two companies in each sector with the most employees

def top_two_companies_by_employee_count(sector):
    return sector.nlargest(2, "Employees")

sectors.apply(top_two_companies_by_employee_count)

```

## Merging DataFrames - The pd.concat Function
```xml
# import 4 different data sets
# Our datasets are spread across multiple files in this section. 
# Each file has a restaurant_ prefix.
# The foods.csv file stores our restaurant's menu items.
foods = pd.read_csv("restaurant_foods.csv")
# The customers.csv file stores our restaurant's customers.
customers = pd.read_csv("restaurant_customers.csv")
# The week_1_sales and week_2_sales files store our orders.
week1 = pd.read_csv("restaurant_week_1_sales.csv")
week2 = pd.read_csv("restaurant_week_2_sales.csv")

week1.head()
week2.head()
len(week1)
len(week2)

# The concat function concatenates one DataFrame to the end of another.
# The original index labels will be kept by default. 
pd.concat([week1, week2])

# Set ignore_index to True to generate a new index.
pd.concat([week1, week2], ignore_index=False)
pd.concat([week1, week2], ignore_index=True)

# The keys parameter create a MultiIndex using the specified keys/labels.
pd.concat([week1, week2], keys=["Week 1", "Week 2"]).sort_index()

df1 = pd.DataFrame([1, 2, 3], columns=["A"])
df1

df2 = pd.DataFrame([4, 5, 6], columns=["B"])
df2

# Pandas will concatenate the DataFrames along the row/index axis.
# Pandas will include all columns that exist in either DataFrame. 
# If there are no matching values, pandas will use NaN values.
pd.concat([df1, df2])
pd.concat([df1, df2], axis="index")

# We can pass the axis parameter an argument of "columns" to concatenate on the column axis.
pd.concat([df1, df2], axis="columns")

```

## Merging DataFrames - Joins 
```xml
# import 4 different data sets
foods = pd.read_csv("restaurant_foods.csv")
customers = pd.read_csv("restaurant_customers.csv")
week1 = pd.read_csv("restaurant_week_1_sales.csv")
week2 = pd.read_csv("restaurant_week_2_sales.csv")
times = pd.read_csv("restaurant_week_1_times.csv")

Left Join
---------

week1.head()
foods.head(5)

# The merge method joins two DataFrames together based on shared values in a column or an index.
# A left join merges one DataFrame into another based on values in the first one.
# The "left" DataFrame is the one we invoke the merge method on.
# If the left DataFrame's value is not found in the right DataFrame, the row will hold NaN values
week1.merge(foods, how="left", on="Food ID")

The left_on and right_on Parameters
-----------------------------------

week2.head()
customers.head()

# The left_on and right_on parameters designate the column names 
# from each DataFrame to use in the merge.
week2.merge(customers, how="left", left_on="Customer ID", right_on="ID")
week2.merge(customers, how="left", left_on="Customer ID", right_on="ID").drop("ID", axis="columns")

Inner Joins
-----------

week1[week1["Customer ID"] == 155]
week2[week2["Customer ID"] == 155]

# Inner joins merge two tables based on shared/common values in columns.
# If only one DataFrame has a value, 
# pandas will exclude it from the final results set.
# If the same ID occurs multiple times, 
# pandas will store each possible combination of the values.
# The design of the join ensures that the results will be the same no matter 
# what DataFrame the merge method is invoked upon.

week1.merge(week2, how="inner", on="Customer ID", suffixes=[" - Week 1", " - Week 2"])


week1.head()
week2.head()

# We can pass multiple arguments to the on parameter of the merge method. 
# Pandas will require matches in both columns across the DataFrames.
week1.merge(week2, how="inner", on=["Customer ID", "Food ID"])

# To verify above join 
condition_one = week1["Customer ID"] == 578
condition_two = week1["Food ID"] == 5
week1[condition_one & condition_two]
condition_one = week2["Customer ID"] == 578
condition_two = week2["Food ID"] == 5
week2[condition_one & condition_two]

Full/Outer Join
---------------

week1.head()
week2.head()

# A full/outer joins values that are found in either DataFrame or both DataFrames.
# Pandas does not mind if a value exists in one DataFrame but not the other.
# If a value does not exist in one DataFrame, it will have a NaN.
week1.merge(week2, how="outer", on="Customer ID", suffixes=[" - Week 1", " - Week 2"])
week1.merge(week2, how="outer", on="Customer ID", suffixes=[" - Week 1", " - Week 2"], indicator=True)

merged = week1.merge(week2, how="outer", on="Customer ID", suffixes=[" - Week 1", " - Week 2"], indicator=True)
merged["_merge"].value_counts()

merged[merged["_merge"].isin(["left_only", "right_only"])]

Merging by Indexes with the left_index and right_index Parameters
-----------------------------------------------------------------

week1.head()
customers.head()
foods.head()

# Use the on parameter if the column(s) to be matched on 
# have the same names in both DataFrames.
# Use the left_on and right_on parameters if the column(s) to be matched on 
# have different names in the two DataFrames.
# Use the left_index or right_index parameters (set to True) to if the values 
# to be matched on are found in the index of a DataFrame.
week1.merge(
    customers, how="left", left_on="Customer ID", right_index=True
).merge(foods, how="left", left_on="Food ID", right_index=True)


The join Method
---------------

week1.head()
times.head()
week1.merge(times, how="left", left_index=True, right_index=True)

# The join method is a shortcut for concatenating two DataFrames when merging by index labels.
week1.join(times)

```

## Working with Dates & Time
```xml

Review of Python's datetime Module
----------------------------------

import pandas as pd
import datetime as dt

# The datetime module is built into the core Python programming language.
# The common alias for the datetime module is dt.
# A module is a Python source file; think of like an internal library that Python loads on demand.
# The datetime module includes date and datetime classes for representing dates and datetimes.
# The date constructor accepts arguments for year, month, and day. Python defaults to 0 for any missing values.
# The datetime constructor accepts arguments for year, month, day, hour, minute, and second.

someday = dt.date(2025, 12, 15)

someday.year
someday.month
someday.day

dt.datetime(2025, 12, 15)
dt.datetime(2025, 12, 15, 8)
dt.datetime(2025, 12, 15, 8, 13)
dt.datetime(2025, 12, 15, 8, 13, 59)

sometime = dt.datetime(2025, 12, 15, 8, 13, 59)
sometime.year
sometime.month
sometime.day
sometime.hour
sometime.minute
sometime.second


The Timestamp and DatetimeIndex Objects
----------------------------------------
# Pandas ships with several classes related to datetimes.
# The Timestamp is similar to Python's datetime object (but with expanded functionality).
# The Timestamp constructor accepts a string, a datetime object, 
# or equivalent arguments to the datetime clas.
pd.Timestamp(2027, 3, 12)
pd.Timestamp(2027, 3, 12, 18, 23, 49)
pd.Timestamp(dt.date(2028, 10, 23))
pd.Timestamp(dt.datetime(2028, 10, 23, 14, 35))
pd.Timestamp("2025-01-01")
pd.Timestamp("2025/04/01")
pd.Timestamp("2021-03-08 08:35:15")

pd.Series([pd.Timestamp("2021-03-08 08:35:15")]).iloc[0]

# A DatetimeIndex is an index of Timestamp objects.
pd.DatetimeIndex(["2025-01-01", "2025-02-01", "2025-03-01"])
index = pd.DatetimeIndex([
    dt.date(2026, 1, 10),
    dt.date(2026, 2, 20)
])

index[0]
type(index[0])


Create Range of Dates with pd.date_range Function
-------------------------------------------------
# The date_range function generates and returns a 
# DatetimeIndex holding a sequence of dates.
# The function requires 2 of the 3 following parameters: 
# start, end, and period.
# With start and end, Pandas will assume a daily period/interval.
# Every element within a DatetimeIndex is a Timestamp

pd.date_range(start="2025-01-01", end="2025-01-07") # interval will be one day 
pd.date_range(start="2025-01-01", end="2025-01-07", freq="D") # same as above 
pd.date_range(start="2025-01-01", end="2025-01-07", freq="2D") # interval set to 2 days
pd.date_range(start="2025-01-01", end="2025-01-07", freq="B") # business days - Monday-Friday 
pd.date_range(start="2025-01-01", end="2025-01-31", freq="W") # will contain weekly freqency - every sunday 
pd.date_range(start="2025-01-01", end="2025-01-31", freq="W-FRI") # will contain weekly freqency - every friday
pd.date_range(start="2025-01-01", end="2025-01-31", freq="W-THU") # will contain weekly freqency - every thursday

pd.date_range(start="2025-01-01", end="2025-01-31", freq="h") # interval will be hourly 
pd.date_range(start="2025-01-01", end="2025-01-31", freq="6h") # interval will be every 6 hours

pd.date_range(start="2025-01-01", end="2025-12-31", freq="ME") # month end interval 
pd.date_range(start="2025-01-01", end="2025-12-31", freq="MS") # month start interval
pd.date_range(start="2025-01-01", end="2050-12-31", freq="YS") # year start interval
pd.date_range(start="2025-01-01", end="2050-12-31", freq="YE") # year end interval

pd.date_range(start="2012-09-09", freq="D", periods=25) # start with a date, interval in day and total 25 days
pd.date_range(start="2012-09-09", freq="3D", periods=40) # start with a date, interval in 3 days and total 40 days
pd.date_range(start="2012-09-09", freq="B", periods=180) # start with a date, interval in business days and total 180 days

pd.date_range(end="2013-10-31", freq="D", periods=20) # end with a date, interval in day and total 20 days in reverse
pd.date_range(end="2016-12-31", freq="B", periods=75) # end with a date, interval in business days and total 75 days in reverse
pd.date_range(end="1991-04-12", freq="W-FRI", periods=75) # end with a date, interval every friday and total 75 days in reverse


The dt Attribute
----------------
# Create a DatetimeIndex between start and end of year with an interval of of 24 days and 3 hours between each date
bunch_of_dates = pd.Series(pd.date_range(start="2000-01-01", end="2020-12-31", freq="24D 3h"))
bunch_of_dates.head()

# The dt attribute reveals a DatetimeProperties object with attributes/methods 
# for working with datetimes. It is similar to the str attribute for string methods.
# The DatetimeProperties object has attributes like day, month, 
# and year to reveal information about each date in the Series.

bunch_of_dates.dt.day # will return day 
bunch_of_dates.dt.month # will return month
bunch_of_dates.dt.year # will return year 
bunch_of_dates.dt.hour # will return hour
bunch_of_dates.dt.day_of_year # will return day of the year 

# The day_name method returns the written day of the week.
bunch_of_dates.dt.day_name() # will return the day of the week 

# Attributes like is_month_end and is_quarter_start return Boolean Series.
bunch_of_dates.dt.is_month_end # will return true if it is month end  
bunch_of_dates[bunch_of_dates.dt.is_month_end] # will return all month end dates 
bunch_of_dates.dt.is_month_start # will return true if it is month start  
bunch_of_dates[bunch_of_dates.dt.is_month_start] # will return all month start dates

bunch_of_dates[bunch_of_dates.dt.is_quarter_start] # will return all quater start dates 

Selecting Rows from a DataFrame with a DateTimeIndex
----------------------------------------------------
# Read file, parse Date, and index it and sort this index  
stocks = pd.read_csv("ibm.csv", parse_dates=["Date"], index_col="Date").sort_index()
stocks.head()

# The iloc accessor is available for index position-based extraction.
stocks.iloc[300] 

# The loc accessor accepts strings or Timestamps to extract by index label/value. 
# Note that Python's datetime objects will not work.
stocks.loc["2014-03-04"]
stocks.loc[pd.Timestamp(2014, 3, 4)]
# Use list slicing to extract a sequence of dates. 
stocks.loc["2014-03-04":"2014-12-31"]
stocks.loc[pd.Timestamp(2014, 3, 4):pd.Timestamp(2014, 12, 31)]
# The truncate method is another alternative.
stocks.truncate("2014-03-04", "2014-12-31")

stocks.loc["2014-03-04", "Close"] 
stocks.loc["2014-03-04", "High":"Close"]

stocks.loc[pd.Timestamp(2014, 3, 4):pd.Timestamp(2014, 12, 31), "High":"Close"]

The DateOffset Object
---------------------
# Read file, parse Date, and index it and sort this index
stocks = pd.read_csv("ibm.csv", parse_dates=["Date"], index_col="Date").sort_index()
stocks.head()

# A DateOffset object adds time to a Timestamp to arrive at a new Timestamp.
stocks.index + pd.DateOffset(days=5)
stocks.index - pd.DateOffset(days=5)
stocks.index + pd.DateOffset(months=3)
stocks.index - pd.DateOffset(years=1)
stocks.index + pd.DateOffset(hours=7)

# The DateOffset constructor accepts days, weeks, months, years parameters, and more.
stocks.index + pd.DateOffset(years=1, months=3, days=2, hours=14, minutes=23, seconds=12)

# We can pass a DateOffset object to the freq parameter of the pd.date_range function.
# Find the IBM stock price on every one of my birthdays (April 12, 1991)
birthdays = pd.date_range(start="1991-04-12", end="2023-04-12", freq=pd.DateOffset(years=1))
birthdays
stocks[stocks.index.isin(birthdays)]


Specialized Date Offsets
------------------------
# Read file, parse Date, and index it and sort this index
stocks = pd.read_csv("ibm.csv", parse_dates=["Date"], index_col="Date").sort_index()
stocks.head()

# Pandas nests more specialized date offsets in pd.tseries.offsets.
# We can add a different amount of time to each date 
# (for example, month end, quarter end, year begin)
stocks.index + pd.tseries.offsets.MonthEnd()
stocks.index - pd.tseries.offsets.MonthEnd()

stocks.index + pd.tseries.offsets.QuarterEnd()
stocks.index - pd.tseries.offsets.QuarterEnd()

stocks.index + pd.tseries.offsets.QuarterBegin(startingMonth=1)
stocks.index - pd.tseries.offsets.QuarterBegin(startingMonth=1)

stocks.index + pd.tseries.offsets.YearEnd()
stocks.index + pd.tseries.offsets.YearBegin()


Timedeltas
----------
# Read file, parse Date, and index it and sort this index
stocks = pd.read_csv("ibm.csv", parse_dates=["Date"], index_col="Date").sort_index()
stocks.head()

# A Timedelta is a pandas object that represents a duration (an amount of time).
# Subtracting two Timestamp objects will yield a Timedelta object 
# (this applies to subtracting a Series from another Series).
pd.Timestamp("2023-03-31 12:30:48") - pd.Timestamp("2023-03-20 19:25:59")
pd.Timestamp("2023-03-20 19:25:59") - pd.Timestamp("2023-03-31 12:30:48")

# The Timedelta constructor accepts parameters for time as well as string descriptions
pd.Timedelta(days=3, hours=2, minutes=5)
pd.Timedelta("5 minutes")
pd.Timedelta("3 days 2 hours 5 minutes")

# Read file and parse Date
ecommerce = pd.read_csv("ecommerce.csv", index_col="ID", parse_dates=["order_date", "delivery_date"], date_format="%m/%d/%y")
ecommerce.head()

ecommerce["Delivery Time"] = ecommerce["delivery_date"] - ecommerce["order_date"]
ecommerce.head()

ecommerce["If It Took Twice As Long"] = ecommerce["delivery_date"] + ecommerce["Delivery Time"]
ecommerce.head()

ecommerce["Delivery Time"].max()
ecommerce["Delivery Time"].min()
ecommerce["Delivery Time"].mean()

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
