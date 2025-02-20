
# Scikit-learn (sklearn)

## Introduction
```xml 
Scikit-Learn, also referred to as sklearn, is an open-source Python machine learning library.

It's built on top on NumPy (Python library for numerical computing) 
and Matplotlib (Python library for data visualization).

Although the fields of data science and machine learning are vast, the main goal is 
finding patterns within data and then using those patterns to make predictions.
And there are certain categories which a majority of problems fall into.

Classification problem: 
----------------------
If you're trying to create a machine learning model to predict whether 
an email is spam and or not spam, you're working on a classification problem 
(whether something is one thing or another).

Regression problem: 
------------------
If you're trying to create a machine learning model to predict the price of houses 
given their characteristics, you're working on a regression problem (predicting a number).

Clustering problem: 
------------------
If you're trying to get a machine learning algorithm to group together similar samples 
(that you don't necessarily know which should go together), 
you're working on a clustering problem.

Once you know what kind of problem you're working on, 
there are also similar steps you'll take for each.

Steps like splitting the data into different sets, 
one for your machine learning algorithms to learn on (the training set) 
and another to test them on (the testing set).

Choosing a machine learning model and then evaluating whether 
or not your model has learned anything.

Scikit-Learn offers Python implementations for doing all of these kinds of tasks 
(from preparing data to modelling data). 
Saving you from having to build them from scratch.


```

## Scikit-learn imports 
```xml 

# Standard imports
# %matplotlib inline # No longer required in newer versions of Jupyter (2022+)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
print(f"Using Scikit-Learn version: {sklearn.__version__} (materials in this notebook require this version or newer).")
# Using Scikit-Learn version: 1.5.1 (materials in this notebook require this version or newer).

sklearn.show_versions() # This also shows the version information 
```


# Scikit-learn Workflow 

![alt text](https://github.com/balaji1974/python-and-machinelearning/blob/main/08%20-%20SciKit-Learn/images/sklearn-workflow-title.png?raw=true)

## Scikit-learn workflow - Steps
```xml 
1. Getting the data ready
2. Choosing the right maching learning estimator/aglorithm/model for your problem
3. Fitting your chosen machine learning model to data and using it to make a prediction
4. Evaluting a machine learning model
5. Improving predictions through experimentation (hyperparameter tuning)
6. Saving and loading a pretrained model
7. Putting it all together in a pipeline


```

## 1. Get the data ready 
```xml 
# The problem at hand is see if someone has heart disease of not.
# It is a classification problem 

# Import dataset
heart_disease = pd.read_csv("./heart-disease.csv")

# View the data
heart_disease.head()

# Create X (all the feature columns except target column)
X = heart_disease.drop("target", axis=1)

# Create y (the target column - label)
y = heart_disease["target"] # This is a result that says if the person has heart disease or not

```

## 2. Choosing the right model and hyper parameters
```xml 
# Random Forest Classifier (for classification problems)
from sklearn.ensemble import RandomForestClassifier

# Instantiating a Random Forest Classifier (clf short for classifier)
clf = RandomForestClassifier(n_estimators=100)

# We will keep the default hyperparameters 
# These are used for fine tuning the model 
clf.get_params() # checking the default hyperparameters

```

## 3. Fit the model to the training data 
```xml 
# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# View the data shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# All models/estimators have the fit() function built-in
clf.fit(X_train, y_train);

# Once fit is called, you can make predictions using predict()
y_preds = clf.predict(X_test)

```

## 4. Evaluate the model 
```xml 
# All models/estimators have a score() function
# On the training set
clf.score(X_train, y_train)

# On the test set (unseen)
clf.score(X_test, y_test)

# Import different classification metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Classification report
print(classification_report(y_test, y_preds))

# Confusion matrix
confusion_matrix(y_test, y_preds)

# Accuracy score
accuracy_score(y_test, y_preds)

```

## 5. Improve a model 
```xml 
# Try different numbers of estimators (n_estimators is a hyperparameter you can change)
np.random.seed(42)
for i in range(10,100,10):
    print(f"Trying model with {i} estimator....")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set : {clf.score(X_test, y_test) * 100:.2f}%")
    print("")

# The output of this will help us select the best estimator to use:
# select the one with the highest accuracy score

```
## 6. Save the model and load it
```xml 
# Saving a model with pickle
import pickle

# Save an existing model to file
pickle.dump(clf, open("rs_random_forest_model_1.pkl", "wb")) # wb- write binary

# Load a saved pickle model
loaded_pickle_model = pickle.load(open("rs_random_forest_model_1.pkl", "rb")) # read binary

# Evaluate loaded model
loaded_pickle_model.score(X_test, y_test)

```

# Scikit-learn Workflow - In Detail  


## 1. Getting the data ready
```xml 
1. Split the data into feature and labels (X and y)
2. Converting non-numerical values into numeric values (also known as feature encoding)
3. Filling (also known as imputing) or disregarding missing values 


1. Split the data into feature and labels (X and y)
---------------------------------------------------

# Import dataset
heart_disease = pd.read_csv("./resources/heart-disease.csv")

# View the data
heart_disease.head()

# Create X (all the feature columns except target column)
X = heart_disease.drop("target", axis=1)

# Create y (the target column - label)
y = heart_disease["target"] # This is a result that says if the person has heart disease or not

# import the train_test_split module from scikit-learn
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# View the data shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape


2. Converting non-numericial values into numeric values 
-------------------------------------------------------
# For this example lets import a new dataset containing non-numeric data 
# Import and view dataset 
car_sales = pd.read_csv("./resources/car-sales-extended.csv")
car_sales.head()

# Explore the dataset  
car_sales["Doors"].value_counts()
len(car_sales)
car_sales.dtypes

# Split into X/y
X = car_sales.drop("Price", axis=1)
y = car_sales["Price"]

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build machine learning model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# The above will throw ValueError: could not convert string to float: 'Toyota'
# We need to convert all Strings into numbers 

# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)],
                                   remainder="passthrough")

transformed_X = transformer.fit_transform(X)
transformed_X

pd.DataFrame(transformed_X)


# Another way to do the same thing with pd.dummies...
dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]],dtype=int)
dummies

# Let's refit the model
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
model.fit(X_train, y_train);

# Test the model 
model.score(X_test, y_test)


3. Disregarding missing values or Filling (also known as imputing) 
------------------------------------------------------------------
# Import car sales missing data
car_sales_missing = pd.read_csv("./resources/car-sales-extended-missing-data.csv")
car_sales_missing.head()

# Find the count of missing values in each column
car_sales_missing.isna().sum()

# Create X & y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# Let's try and convert our data to numbers
# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot,
                 categorical_features)],remainder="passthrough")

transformed_X = transformer.fit_transform(X)
transformed_X

# The above will throw ValueError: Input contains NaN (but in newer versions of Scikit Learn this is ignored)

3.1 Fill the missing values using Pandas
----------------------------------------
# Fill the "Make" column
car_sales_missing["Make"]=car_sales_missing["Make"].fillna("missing")

# Fill the "Colour" column
car_sales_missing["Colour"]=car_sales_missing["Colour"].fillna("missing")

# Fill the "Odometer (KM)" column
car_sales_missing["Odometer (KM)"]=car_sales_missing["Odometer (KM)"].fillna(car_sales_missing["Odometer (KM)"].mean())

# Fill the "Doors" column
car_sales_missing["Doors"]=car_sales_missing["Doors"].fillna(4)

# Check our dataframe again
car_sales_missing.isna().sum()

# Remove rows with missing Price value
# This is because we are trying to predict car sales and so cannot have any missing value here
car_sales_missing.dropna(inplace=True) 

# Check our dataframe again
car_sales_missing.isna().sum()

# To check how many missing values we have remaining after removing price = NA  
len(car_sales_missing)

# ReCreate X & y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# Let's try and convert our data to numbers
# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",one_hot,
                    categorical_features)],remainder="passthrough")

transformed_X = transformer.fit_transform(car_sales_missing)
transformed_X

3.2 Fill missing values with Scikit-Learn
-----------------------------------------

car_sales_missing = pd.read_csv("resources/car-sales-extended-missing-data.csv")
car_sales_missing.head()

car_sales_missing.isna().sum()

# Drop the rows with no labels
car_sales_missing.dropna(subset=["Price"], inplace=True)
car_sales_missing.isna().sum()

# Check missing values
X.isna().sum()

# Fill missing values with Scikit-Learn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill categorical values with 'missing' & numerical values with mean
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
door_imputer = SimpleImputer(strategy="constant", fill_value=4)
num_imputer = SimpleImputer(strategy="mean")

# Define columns
cat_features = ["Make", "Colour"]
door_feature = ["Doors"]
num_features = ["Odometer (KM)"]

# Create an imputer (something that fills missing data)
imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, cat_features),
    ("door_imputer", door_imputer, door_feature),
    ("num_imputer", num_imputer, num_features)
])

# Fill train and test values separately
filled_X_train = imputer.fit_transform(X_train)
filled_X_test = imputer.transform(X_test)

# Check filled X_train
filled_X_train

# Get our transformed data array's back into DataFrame's
car_sales_filled_train = pd.DataFrame(filled_X_train, 
                                      columns=["Make", "Colour", "Doors", "Odometer (KM)"])
car_sales_filled_test = pd.DataFrame(filled_X_test, 
                                     columns=["Make", "Colour", "Doors", "Odometer (KM)"])

# Check missing data in training set
car_sales_filled_train.isna().sum()

# Check to see the original... still missing values
car_sales_missing.isna().sum()

# Now let's one hot encode the features with the same code as before 
categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, 
                        categorical_features)], remainder="passthrough")

# Fill train and test values separately
transformed_X_train = transformer.fit_transform(car_sales_filled_train)
transformed_X_test = transformer.transform(car_sales_filled_test)

# Check transformed and filled X_train
transformed_X_train.toarray()

# Now we've transformed X, let's see if we can fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

# Make sure to use transformed (filled and one-hot encoded X data)
model.fit(transformed_X_train, y_train)
model.score(transformed_X_test, y_test)

# Check length of transformed data (filled and one-hot encoded)
# vs. length of original data 
# This model performs worst than the original model 
# because of the size of the data which is smaller than the original data 
len(transformed_X_train.toarray())+len(transformed_X_test.toarray()), len(car_sales)

```

## Feature Scaling  
```xml
Feature Scaling
Once your data is all in numerical format, there's one more transformation 
you'll probably want to do to it.

It's called Feature Scaling.

In other words, making sure all of your numerical data is on the same scale.

For example, say you were trying to predict the sale price of cars and 
the number of kilometres on their odometers varies from 6,000 to 345,000 
but the median previous repair cost varies from 100 to 1,700. 
A machine learning algorithm may have trouble finding patterns in 
these wide-ranging variables.

To fix this, there are two main types of feature scaling.

Normalization (also called min-max scaling):  
This rescales all the numerical values to between 0 and 1, 
with the lowest value being close to 0 and the highest previous value being 
close to 1. 
Scikit-Learn provides functionality for this in the MinMaxScalar class.

Standardization:  
This subtracts the mean value from all of the features 
(so the resulting features have 0 mean). 
It then scales the features to unit variance (by dividing the feature 
by the standard deviation). Scikit-Learn provides functionality for 
this in the StandardScalar class.

A couple of things to note.

Feature scaling usually isn't required for your target variable.

Feature scaling is usually not required with tree-based models 
(e.g. Random Forest) since they can handle varying features.

References: (Good reads)
https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
https://benalexkeen.com/feature-scaling-with-scikit-learn/
https://rahul-saini.medium.com/feature-scaling-why-it-is-required-8a93df1af310
```


# Choosing the righ estimator 
![alt text](https://github.com/balaji1974/python-and-machinelearning/blob/main/08%20-%20SciKit-Learn/images/choosing-the-right-estimator.png?raw=true)
https://scikit-learn.org/stable/machine_learning_map.html

## 2. Choosing the right estimator/algorithm for your problem
```xml
2.1 Picking a machine learning model for a regression problem

# Get California Housing dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
housing

# Getting it into a dataframe
housing_df = pd.DataFrame(housing["data"], columns=housing["feature_names"])
housing_df

# Adding the target variable into our dataset 
housing_df["target"] = housing["target"]
housing_df.head()

# Import algorithm/estimator
from sklearn.linear_model import Ridge

# Setup random seed
np.random.seed(42)

# Create the data
X = housing_df.drop("target", axis=1)
y = housing_df["target"] # median house price in $100,000s

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit the model (on the training set)
model = Ridge()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)
0.5758549611440126 <- result 

What if Ridge didn't work or the score didn't fit our needs?

Well, we could always try a different model...

How about we try an ensemble model (an ensemble is combination of smaller models to try and make better predictions than just a single model)?

Sklearn's ensemble models can be found here: https://scikit-learn.org/stable/modules/ensemble.html

# Import the RandomForestRegressor model class from the ensemble module
from sklearn.ensemble import RandomForestRegressor

# Setup random seed
np.random.seed(42)

# Create the data
X = housing_df.drop("target", axis=1)
y = housing_df["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create random forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)
0.8066196804802649 <- result 


2.2 Picking a machine learning model for a classification problem
Let's go to the map... https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# Get the data (be sure to click "raw") - https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/data/heart-disease.csv 
heart_disease = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
heart_disease.head()

len(heart_disease)

# Consulting the map and it says to try LinearSVC.

# Import the LinearSVC estimator class
from sklearn.svm import LinearSVC

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate LinearSVC
clf = LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)

# Evaluate the LinearSVC
clf.score(X_test, y_test)

# Result -> 0.8688524590163934

heart_disease["target"].value_counts()

# Import the RandomForestClassifier estimator class
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate the Random Forest Classifier
clf.score(X_test, y_test)

# Result -> 0.8524590163934426


# Tidbit:
# 1. If you have structured data, used ensemble methods
# 2. If you have unstructured data, use deep learning or transfer learning

heart_disease


```

## 3. Fit the model/algorithm on our data and use it to make predictions
```xml
# 3.1 Fitting the model to the data

Different names for:
X = features, features variables, data 
y = labels, targets, target variables


# Import the RandomForestClassifier estimator class
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)

# Fit the model to the data (training the machine learning model)
clf.fit(X_train, y_train)

# Evaluate the Random Forest Classifier (use the patterns the model has learned)
clf.score(X_test, y_test)

# Result -> 0.8524590163934426


```



### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery
https://scikit-learn.org/stable/user_guide.html

https://williamkoehrsen.medium.com/random-forest-simple-explanation-377895a60d2d

```
