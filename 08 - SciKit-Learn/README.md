
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

X.head()
y.tail()

# 3.2 Make predictions using a machine learning model
2 ways to make predictions:
predict()
predict_proba()

clf.predict(X_test)
np.array(y_test)

# Compare predictions to truth labels to evaluate the model
y_preds = clf.predict(X_test)
np.mean(y_preds == y_test) 
=> 0.8524590163934426

# The below will give the same results as above
clf.score(X_test, y_test)
=> 0.8524590163934426

# or one more way of checking this
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)
=> 0.8524590163934426

# Make predictions with predict_proba() - use this if someone asks you 
# "what's the probability your model is assigning to each prediction?"
# predict_proba() returns probabilities of a classification label 
clf.predict_proba(X_test[:5])
=> Result
array([[0.89, 0.11],
       [0.49, 0.51],
       [0.43, 0.57],
       [0.84, 0.16],
       [0.18, 0.82]])
# Here instead of returning 0 or 1 as from the below result of predict 
# it gives a probability of model being 0 or 1. Eg. for the first sample
# the probability of model returning 0 is 0.89 
# and the probability of model returing 1 is 0.11. 

# Let's predict() on the same data...
clf.predict(X_test[:5])
X_test[:5]

# `predict()` can also be used for regression models.
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# Create the data
X = housing_df.drop("target", axis=1)
y = housing_df["target"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model instance
model = RandomForestRegressor()

# Fit the model to the data
model.fit(X_train, y_train)

# Make predictions
y_preds = model.predict(X_test)

y_preds[:10]
np.array(y_test[:10])

# Compare the predictions to the truth
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_preds)
result => 0.3265721842781009 => This means our prediction is 
0.3265... different from the target value 

housing_df["target"]
```


## 4. Evaluating a machine learning model
```xml
Three ways to evaluate Scikit-Learn models/estimators:

1. Estimator's built-in score() method
2. The scoring parameter
3. Problem-specific metric functions
You can read more about these here: 
https://scikit-learn.org/stable/modules/model_evaluation.html

# 4.1 Evaluating a model with the score method
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate Random Forest Classifier
clf = RandomForestClassifier(n_estimators=1000)

# Fit the model to the data (training the machine learning model)
clf.fit(X_train, y_train)

# The highest value for the .score() method is 1.0, the lowest is 0.0
clf.score(X_train, y_train)
result=> 1.0

clf.score(X_test, y_test)
result => 0.8688524590163934

Let's use the score() on our regression problem...
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# Create the data
X = housing_df.drop("target", axis=1)
y = housing_df["target"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model instance
model = RandomForestRegressor(n_estimators=100)

# Fit the model to the data
model.fit(X_train, y_train)

# The default score() evaluation metric is r_squared for regression algorithms
# Highest = 1.0, lowest = 0.0
model.score(X_test, y_test)


# 4.2 Evaluating a model using the scoring parameter
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train);
clf.score(X_test, y_test)


# In cross validation score, the model is trained on different set 
# of training data (based on cv value that is set) and evaluated on 
# different set of test data (same number set by the cv value)
cross_val_score(clf, X, y, cv=5) # five different scores
cross_val_score(clf, X, y, cv=10) # ten different scores 


np.random.seed(42)

# Single training and test split score
clf_single_score = clf.score(X_test, y_test)

# Take the mean of 5-fold cross-validation score
clf_cross_val_score = np.mean(cross_val_score(clf, X, y, cv=5))

# Compare the two
clf_single_score, clf_cross_val_score

# Scoring parameter set to None by default
cross_val_score(clf, X, y, cv=5, scoring=None)


# 4.2.1 Classification model evaluation metrics
Accuracy
Area under ROC curve
Confusion matrix
Classification report

# Accuracy
heart_disease.head()

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

clf = RandomForestClassifier(n_estimators=100)
cross_val_score = cross_val_score(clf, X, y, cv=5)

np.mean(cross_val_score)
print(f"Heart Disease Classifier Cross-Validated Accuracy: {np.mean(cross_val_score) *100:.2f}%")
result => Heart Disease Classifier Cross-Validated Accuracy: 82.48%


# Area under the receiver operating characteristic curve (AUC/ROC)
Area under curve (AUC)
ROC curve

# ROC curves are a comparison of a model's true postive rate (tpr) 
# versus a models false positive rate (fpr).
True positive = model predicts 1 when truth is 1
False positive = model predicts 1 when truth is 0
True negative = model predicts 0 when truth is 0
False negative = model predicts 0 when truth is 1

# Create test,train split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.metrics import roc_curve

# Fit the classifier
clf.fit(X_train, y_train)

# Make predictions with probabilities
y_probs = clf.predict_proba(X_test)

y_probs[:10], len(y_probs)

y_probs_positive = y_probs[:, 1]
y_probs_positive[:10]

# Caculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)

# Check the false positive rates
fpr

# Create a function for plotting ROC curves
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    """
    Plots a ROC curve given the false positive rate (fpr)
    and true positive rate (tpr) of a model.
    """
    # Plot roc curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    # Plot line with no predictive power (baseline)
    #plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")
    
    # Customize the plot
    plt.xlabel("False positive rate (fpr)")
    plt.ylabel("True positive rate (tpr)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()

plot_roc_curve(fpr, tpr)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_probs_positive)

# Plot perfect ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_test)
plot_roc_curve(fpr, tpr)

# Perfect AUC score
roc_auc_score(y_test, y_test)

# ROC curves and AUC metrics are evaluation metrics 
# for binary classification models (a model which predicts 
# one thing or another, such as heart disease or not).

# The ROC curve compares the true positive rate (tpr) 
# versus the false positive rate (fpr) at different 
# classification thresholds.

# The AUC metric tells you how well your model is at choosing 
# between classes (for example, how well it is at deciding 
# whether someone has heart disease or not). 
# A perfect model will get an AUC score of 1.


# Confusion matrix
# The next way to evaluate a classification model is by using a confusion matrix.
# A confusion matrix is a quick way to compare the labels a model predicts and 
# the actual labels it was supposed to predict. 
# In essence, giving you an idea of where the model is getting confused.

from sklearn.metrics import confusion_matrix

preds = clf.predict(X_test)

confusion_matrix(y_test, y_preds)

# Again, this is probably easier visualized.
# One way to do it is with pd.crosstab().
pd.crosstab(y_test, 
            y_preds, 
            rownames=["Actual Label"], 
            colnames=["Predicted Label"])

# For the below to work install seabourn module
# Make our confusion matrix more visual with Seaborn's heatmap()
import seaborn as sns

# Set the font scale 
sns.set(font_scale=1.5)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)

# Plot it using Seaborn
sns.heatmap(conf_mat);

# Creating a confusion matrix using Scikit-Learn
# Scikit-Learn has multiple different implementations of plotting confusion matrices:

# 1. sklearn.metrics.ConfusionMatrixDisplay.from_estimator(estimator, X, y) - 
# this takes a fitted estimator (like our clf model), features (X) and labels (y), 
# it then uses the trained estimator to make predictions on X and 
# compares the predictions to y by displaying a confusion matrix.

# 2. sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred) - 
# this takes truth labels and predicted labels and 
# compares them by displaying a confusion matrix.

# Note: Both of these methods/classes require Scikit-Learn 1.0+. To check your version of Scikit-Learn run:

import sklearn
sklearn.__version__
# If you don't have 1.0+, you can upgrade at: https://scikit-learn.org/stable/install.html

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(estimator=clf, X=X, y=y);

# Plot confusion matrix from predictions
ConfusionMatrixDisplay.from_predictions(y_true=y_test, 
                                        y_pred=y_preds);

# Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_preds))

# Where precision and recall become valuable
disease_true = np.zeros(10000)
disease_true[0] = 1 # only one positive case

disease_preds = np.zeros(10000) # model predicts every case as 0

pd.DataFrame(classification_report(disease_true,
                                   disease_preds,
                                   output_dict=True,
                                   zero_division=0))

# To summarize classification metrics:
# Accuracy is a good measure to start with if all classes are balanced 
# (e.g. same amount of samples which are labelled with 0 or 1).
# Precision and recall become more important when classes are imbalanced.
# If false positive predictions are worse than false negatives, 
# aim for higher precision.
# If false negative predictions are worse than false positives, 
# aim for higher recall.
# F1-score is a combination of precision and recall.



```

## Additional Resources
```xml
# Random Forest model deep dive
## These resources will help you understand what's happening inside the Random Forest models we've been using.
https://en.wikipedia.org/wiki/Random_forest
https://builtin.com/data-science/random-forest-python-deep-dive
https://williamkoehrsen.medium.com/random-forest-simple-explanation-377895a60d2d
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
https://www.youtube.com/watch?v=4jRBRDbJemM

```

### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery
https://scikit-learn.org/stable/user_guide.html
```
