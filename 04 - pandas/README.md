
# Pandas 



## Pandas - Intro
```xml
Pandas documentation
--------------------
https://pandas.pydata.org/pandas-docs/stable/

10 Minutes to Pandas
--------------------
https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min


Why Pandas? 
Pandas allows us to analyze big data and make conclusions based on statistical theories. 
Pandas can clean messy data sets, and make them readable and relevant. 
Relevant data is very important in data science

```

## Series 
```xml

Series -> are one Dimensional

import pandas as pd -> Import Panda library 
series = pd.Series(["BMW","Toyota","Honda"]) -> Create a series with data 
series -> print the series that was created 

colors = pd.Series(["Red", "Yello", "Blue"])
colors

```

## Dataframes 
```xml

Dataframes -> are two Dimensional

Create a dataframe: 
car_data=pd.DataFrame({"Car Make": series, "Color": colors })
car_data

Export a dataframe to csv file: 
car_sales.to_csv("exported_car_sales.csv", index=False) -> 
with false, excel's row index will not be imported as a column 

Import dataframe directly from URL: 
heart_disease = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")


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
car_sales[(car_sales["Make"]=="Toyota") & (car_sales["Odometer (KM)"]>100000)]

# Cross tab between 2 different columns 
pd.crosstab(car_sales['Make'], car_sales['Doors'])

# Group by make and find mean of numeric columns 
car_sales.groupby(["Make"]).mean(numeric_only=True)

# To plot a graph on a column value 
car_sales["Odometer (KM)"].plot()

# To draw histogram - to visulaize the spread of data 
car_sales["Odometer (KM)"].hist()

# Change Price column to integers
car_sales["Price"] = car_sales["Price"].str.replace('[\\$\\,\\.]', '', regex=True).astype(int)

# Divide by 100 as decimal was removed in previous step and round result by 2 decimials
car_sales["Price"] = car_sales["Price"].div(100).round(2)


```


## Manipulating Data
```xml

# Printing the values of a string in lower case 
car_sales["Make"].str.lower()


# Import a file into panda dataframe
car_sales_missing=pd.read_csv("car-sales-missing-data.csv")

# Fill the missing values in Odometer with mean of available values
car_sales_missing["Odometer"]=car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean())

# Drop all the rows with NA values
car_sales_missing_droped=car_sales_missing.dropna()






```


### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/
https://pandas.pydata.org/pandas-docs/stable/
https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html
https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html

```