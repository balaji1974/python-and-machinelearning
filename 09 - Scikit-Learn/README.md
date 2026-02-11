
# scikit-learn (sklearn)

## Introduction
```xml 
scikit-learn, also referred to as sklearn, is an open-source Python machine learning library.

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

scikit-learn offers Python implementations for doing all of these kinds of tasks 
(from preparing data to modelling data). 
Saving you from having to build them from scratch.


```

## scikit-learn imports 
```xml 
Open Anaconda
Select jypter lab
Select your kernal inside jypter lab 
Choose your working folder
Create a new notebook and rename it. 


# Standard imports
# %matplotlib inline # No longer required in newer versions of Jupyter (2022+)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
print(f"Using scikit-learn version: {sklearn.__version__} (materials in this notebook require this version or newer).")
# Using scikit-learn version: 1.5.1 (materials in this notebook require this version or newer).

sklearn.show_versions() # This also shows the version information 

```

# scikit-learn Workflow 

![alt text](https://github.com/balaji1974/python-and-machinelearning/blob/main/09%20-%20scikit-learn/images/sklearn-workflow-title.png?raw=true)


## scikit-learn workflow - Steps
```xml 
1. Getting the data ready
2. Choosing the right maching learning estimator/aglorithm/model for your problem
3. Fitting your chosen machine learning model to data and using it to make a prediction
4. Evaluting a machine learning model
5. Improving predictions through experimentation (hyperparameter tuning)
6. Saving and loading a pretrained model
7. Putting it all together in a pipeline


In simple english & shortform:
Raw Data
   ↓
Train/Test Split
   ↓
Preprocessing (fit only on train)
   ↓
Model Training
   ↓
Evaluation
   ↓
Tuning
   ↓
Persist Model
   ↓
Deploy & Predict


```

## Sample machine learning end to end workflow  
```xml 
Refer to 01-End-To-End-Workflow-RandomForestClassifer.ipynb 

# The problem at hand is see if someone has heart disease of not.
# It is a classification problem 

1. Getting the data ready
-------------------------
# Import dataset
heart_disease = pd.read_csv("data/heart-disease.csv")

# View the data
heart_disease.head()

# Create X (all the feature columns except target column)
X = heart_disease.drop("target", axis=1)

# Create y (the target column - label)
y = heart_disease["target"] # This is a result that says if the person has heart disease or not


2. Choosing the right maching learning estimator/aglorithm/model for your problem
---------------------------------------------------------------------------------
# Choosing the right model and hyper parameters
# Random Forest Classifier (for classification problems)
from sklearn.ensemble import RandomForestClassifier

# Instantiating a Random Forest Classifier (clf short for classifier)
clf = RandomForestClassifier(n_estimators=100)

# We will keep the default hyperparameters 
# These are used for fine tuning the model 
clf.get_params() # checking the default hyperparameters

3. Fit the model to the training data 
-------------------------------------
# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# View the data shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# All models/estimators have the fit() function built-in
clf.fit(X_train, y_train);

# Once fit is called, you can make predictions using predict()
y_preds = clf.predict(X_test)

4. Evaluate the model 
---------------------
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

5. Improve a model 
------------------
# Try different numbers of estimators (n_estimators is a hyperparameter you can change)
np.random.seed(42)
for i in range(10,100,10):
    print(f"Trying model with {i} estimator....")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set : {clf.score(X_test, y_test) * 100:.2f}%")
    print("")

# The output of this will help us select the best estimator to use:
# select the one with the highest accuracy score

6. Save the model and load it
-----------------------------
# Saving a model with pickle
import pickle

# Save an existing model to file
pickle.dump(clf, open("rs_random_forest_model_1.pkl", "wb")) # wb- write binary

# Load a saved pickle model
loaded_pickle_model = pickle.load(open("rs_random_forest_model_1.pkl", "rb")) # read binary

# Evaluate loaded model
loaded_pickle_model.score(X_test, y_test)

```

# Scikit-learn Workflow In Detail  

## Getting the data ready
```xml 
Refer to 02-Getting-The-Data-Ready.ipynb

Clean -> Transform -> Reduce 
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

# Let's refit the model
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
model.fit(X_train, y_train);

# Test the model 
model.score(X_test, y_test)


# Another way to do the same thing with pd.dummies...
dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]],dtype=int)
dummies

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(dummies,
                                                    y, 
                                                    test_size=0.2)
model.fit(X_train, y_train)
model.score(X_test, y_test)


3. Disregarding missing values or Filling (also known as imputing) 
------------------------------------------------------------------
Note: The current version of OneHotEncoder (0.23+ versions) can handle 
missing values None & NaN. So can still use OneHotEncoder here as alternative.

# Import car sales missing data
car_sales_missing = pd.read_csv("./resources/car-sales-extended-missing-data.csv")
car_sales_missing.head()

# Find the count of missing values in each column
car_sales_missing.isna().sum()


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


3.2 Fill missing values with scikit-learn
-----------------------------------------

car_sales_missing = pd.read_csv("resources/car-sales-extended-missing-data.csv")
car_sales_missing.head()

car_sales_missing.isna().sum()

# Drop the rows with no labels
car_sales_missing.dropna(subset=["Price"], inplace=True)
car_sales_missing.isna().sum()

# Check missing values
X.isna().sum()

# Fill missing values with scikit-learn
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

```

# Feature Scaling 
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
scikit-learn provides functionality for this in the MinMaxScalar class.

Standardization:  
This subtracts the mean value from all of the features 
(so the resulting features have 0 mean). 
It then scales the features to unit variance (by dividing the feature 
by the standard deviation). scikit-learn provides functionality for 
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
![alt text](https://github.com/balaji1974/python-and-machinelearning/blob/main/09%20-%20scikit-learn/images/choosing-the-right-estimator.png?raw=true)

```xml
Sklearn refers to machine learning models, algorithms as estimators.
Classification problem - predicting a category (heart disease or not)
Sometimes you'll see clf (short for classifier) used as a classification estimator
Regression problem - predicting a number (selling price of a car)

Ref:
https://scikit-learn.org/stable/machine_learning_map.html

```

## Choosing the right estimator/algorithm for your problem
```xml
Refer to 03-Choosing-The-Model.ipynb

You can look into all the sample datasets that sklearn provides 
in the below link:
https://scikit-learn.org/stable/api/sklearn.datasets.html

1. Picking a machine learning model for a regression problem
------------------------------------------------------------

# Get California Housing dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
housing

# Getting it into a dataframe
housing_df = pd.DataFrame(housing["data"], columns=housing["feature_names"])
housing_df

# Add a target variable to the dataframe from our dataset
housing_df["MedHouseVal"] = housing["target"]
housing_df.head()

# Setup random seed
np.random.seed(42)

# Create the data
X = housing_df.drop("MedHouseVal", axis=1)
y = housing_df["MedHouseVal"] # median house price in $100,000s

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

1.1 Ridge 
---------
# Import algorithm/estimator
from sklearn.linear_model import Ridge

# Instantiate and fit the model (on the training set)
model = Ridge()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)
0.5758549611440126 <- result 


What if Ridge didn't work or the score didn't fit our needs?
Well, we could always try a different model...
How about we try an ensemble model 
(an ensemble is combination of smaller models to try and make better predictions than just a single model)?
Sklearn's ensemble models can be found here: https://scikit-learn.org/stable/modules/ensemble.html

1.2 Lasso
---------
# Import algorithm/estimator
from sklearn.linear_model import Lasso

# Instantiate and fit the model (on the training set)
model = Lasso()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


What if both Ridge & Lasso didn't work or the score didn't fit our needs?
Well, we could always try a different model...
How about we try an ensemble model?
An ensemble is combination of smaller models to try and make better predictions than just a single model
Sklearn's ensemble models can be found here: https://scikit-learn.org/stable/modules/ensemble.html

Sklearn's Random Forest Regressor can be found here:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

Random Forest Algorithm in Machine Learning
https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/

1.3 Random Foreset Regressor 
----------------------------
# Import the RandomForestRegressor model class from the ensemble module
from sklearn.ensemble import RandomForestRegressor

# Create random forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)
0.8066196804802649 <- result 


2. Picking a machine learning model for a classification problem
----------------------------------------------------------------
# Get the data (be sure to click "raw") - https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/data/heart-disease.csv 
heart_disease = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
heart_disease.head()

len(heart_disease)

Let's go to the map... 
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# Consulting the map and it says to try LinearSVC.
# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


2.1 LinearSVC 
-------------
# Import the LinearSVC estimator class
from sklearn.svm import LinearSVC

# Instantiate LinearSVC
clf = LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)

# Evaluate the LinearSVC
clf.score(X_test, y_test)

# Result -> 0.8688524590163934

heart_disease["target"].value_counts()


2.2 RandomForestClassifier 
--------------------------
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


Note:
-----
1. If you have structured data, used ensemble methods
2. If you have unstructured data, use deep learning or transfer learning

```

## Fit the model to our data and use it to make predictions
```xml
Refer to 04-Fit-The-Model-To-The-Data.ipynb

1. Classification Problem
-------------------------

1.1 Fitting the model to the data
---------------------------------
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

1.2. Make predictions using a machine learning model
--------------------------------------------------
2 ways to make predictions:
predict()
predict_proba()


1.2.1 Make predictions using predict function
---------------------------------------------
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


1.2.2 Make predictions using predict_proba function 
----------------------------------------------------
Make predictions with predict_proba(): 
use this if someone asks you 
"what's the probability your model is assigning to each prediction?"

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


2. Regression Problem
---------------------
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
result => 0.3265721842781009 => 
This means our prediction is 0.3265... different from the target value 

housing_df["target"]


Random Forest model deep dive
-----------------------------
These resources will help you understand what's happening inside 
the Random Forest models we've been using.

https://en.wikipedia.org/wiki/Random_forest
https://simple.wikipedia.org/wiki/Random_forest
https://www.kdnuggets.com/2016/12/random-forests-python.html
https://willkoehrsen.github.io/machine%20learning/tutorial/an-implementation-and-explanation-of-the-random-forest-in-python


Different names for:
--------------------
X = features, features variables, data 
y = labels, targets, target variables

```

## Evaluating a machine learning model
```xml
Refer to 05-Evaluating-The-Model.ipynb

Three ways to evaluate scikit-learn models/estimators:

1. Estimator's built-in score() method
2. The scoring parameter
3. Problem-specific metric functions
You can read more about these here: 
https://scikit-learn.org/stable/modules/model_evaluation.html

Evaluating a model with the score method
----------------------------------------
clf.score(X_test, y_test)
result => 0.8688524590163934

# The default score() evaluation metric is r_squared for regression algorithms
# Highest = 1.0, lowest = 0.0
model.score(X_test, y_test)

Evaluating a model using the scoring parameter
----------------------------------------------
The cross_val_score function in scikit-learn evaluates a model's performance 
using cross-validation. It repeatedly splits the data into training and 
testing sets, trains the model on the training data, and computes a score on 
the test data for each split (or "fold"). The function returns an array of 
these scores, providing a more robust estimate of how the model is expected 
to perform on unseen data compared to a single train-test split. 
```

![alt text](https://github.com/balaji1974/python-and-machinelearning/blob/main/08%20-%20SciKit-Learn/images/cross-val-score.png?raw=true)
```xml
# In cross validation score, the model is trained on different set 
# of training data (based on cv value that is set) and evaluated on 
# different set of test data (same number set by the cv value)
cross_val_score(clf, X, y, cv=5) # five different scores
cross_val_score(clf, X, y, cv=10) # ten different scores 

# Single training and test split score
clf_single_score = clf.score(X_test, y_test)

# Take the mean of 5-fold cross-validation score
clf_cross_val_score = np.mean(cross_val_score(clf, X, y, cv=5))

# Compare the two
clf_single_score, clf_cross_val_score

# Scoring parameter set to None by default
cross_val_score(clf, X, y, cv=5, scoring=None)


Classification model evaluation metrics:
----------------------------------------
Accuracy
Area under ROC curve
Confusion matrix
Classification report

Accuracy
--------
cross_val_score = cross_val_score(clf, X, y, cv=5)
np.mean(cross_val_score)
print(f"Heart Disease Classifier Cross-Validated Accuracy: {np.mean(cross_val_score) *100:.2f}%")
result => Heart Disease Classifier Cross-Validated Accuracy: 82.48%


Area under the receiver operating characteristic curve (AUC/ROC)
----------------------------------------------------------------
Area under curve (AUC)
ROC curve

# ROC curves are a comparison of a model's true postive rate (tpr) 
# versus a models false positive rate (fpr).
True positive = model predicts 1 when truth is 1
False positive = model predicts 1 when truth is 0
True negative = model predicts 0 when truth is 0
False negative = model predicts 0 when truth is 1

from sklearn.metrics import roc_curve
# Fit the classifier
clf.fit(X_train, y_train)
# Make predictions with probabilities
y_probs = clf.predict_proba(X_test)
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

ROC curves and AUC metrics are evaluation metrics for binary 
classification models (a model which predicts one thing or another, 
such as heart disease or not).

The ROC curve compares the true positive rate (tpr) versus the 
false positive rate (fpr) at different classification thresholds.

The AUC metric tells you how well your model is at choosing between classes 
(for example, how well it is at deciding whether someone has heart disease or not). 
A perfect model will get an AUC score of 1.

Confusion matrix
----------------
The next way to evaluate a classification model is by using a confusion matrix.
A confusion matrix is a quick way to compare the labels a model predicts and 
the actual labels it was supposed to predict. 
In essence, giving you an idea of where the model is getting confused.

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

# Creating a confusion matrix using scikit-learn
scikit-learn has multiple different implementations of plotting confusion matrices:

1. sklearn.metrics.ConfusionMatrixDisplay.from_estimator(estimator, X, y) - 
this takes a fitted estimator (like our clf model), features (X) and labels (y), 
it then uses the trained estimator to make predictions on X and 
compares the predictions to y by displaying a confusion matrix.

2. sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred) - 
this takes truth labels and predicted labels and compares them by displaying a confusion matrix.

Note: Both of these methods/classes require scikit-learn 1.0+.

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(estimator=clf, X=X, y=y);

# Plot confusion matrix from predictions
ConfusionMatrixDisplay.from_predictions(y_true=y_test, 
                                      y_pred=y_preds);


Classification Report
---------------------
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

Precision - Indicates the proportion of positive identifications (model predicted class 1) 
which were actually correct. A model which produces no false positives has a precision of 1.0.

Recall - Indicates the proportion of actual positives which were correctly classified. 
A model which produces no false negatives has a recall of 1.0.

F1 score - A combination of precision and recall. A perfect model achieves an F1 score of 1.0.

Support - The number of samples each metric was calculated on.

Accuracy - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0, 
in other words, getting the prediction right 100% of the time.

Macro avg - Short for macro average, the average precision, recall and F1 score between classes. 
Macro avg doesn't take class imbalance into effect. So if you do have class imbalances 
(more examples of one class than another), you should pay attention to this.

Weighted avg - Short for weighted average, the weighted average precision, recall and 
F1 score between classes. Weighted means each metric is calculated with respect to how many 
samples there are in each class. This metric will favour the majority class 
(e.g. it will give a high value when one class out performs another due to having more samples).


To summarize classification metrics:
------------------------------------
Accuracy is a good measure to start with if all classes are balanced 
(e.g. same amount of samples which are labelled with 0 or 1).

Precision and recall become more important when classes are imbalanced.
If false positive predictions are worse than false negatives, aim for higher precision.
If false negative predictions are worse than false positives, aim for higher recall.
F1-score is a combination of precision and recall.


Regression model evaluation metrics
-----------------------------------
Model evaluation metrics documentation - 
https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

The ones we're going to cover are:
1. R^2 (pronounced r-squared) or coefficient of determination
2. Mean absolute error (MAE)
3. Mean squared error (MSE)


R^2
---
What R-squared does: Compares your models predictions to the mean of the targets. 
Values can range from negative infinity (a very poor model) to 1. 
For example, if all your model does is predict the mean of the targets, 
it's R^2 value would be 0. 
And if your model perfectly predicts a range of numbers it's R^2 value would be 1.

model.score(X_test, y_test)
y_test
y_test.mean()

from sklearn.metrics import r2_score
# Fill an array with y_test mean
y_test_mean = np.full(len(y_test), y_test.mean())
y_test_mean[:10]
r2_score(y_true=y_test, y_pred=y_test_mean) -> 0.0

r2_score(y_true=y_test, y_pred=y_test) -> 1.0

Mean absolute error (MAE)
-------------------------
MAE is the average of the absolute differences between predictions and 
actual values. It gives you an idea of how wrong your models predictions are.

# MAE
from sklearn.metrics import mean_absolute_error
y_preds = model.predict(X_test)
mae = mean_absolute_error(y_test, y_preds)
mae

df = pd.DataFrame(data={"actual values": y_test, "predicted values": y_preds})
df["differences"] = df["predicted values"] - df["actual values"]
df.head(10)

Mean squared error (MSE)
------------------------
MSE is the mean of the square of the errors between actual and predicted values.

# Mean squared error
from sklearn.metrics import mean_squared_error
y_preds = model.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
mse

df["squared_differences"] = np.square(df["differences"])
df.head()

# Calculate MSE by hand
squared = np.square(df["differences"])
squared.mean()

df_large_error = df.copy()
df_large_error.iloc[0]["squared_differences"] = 16 # increase "squared_differences" for 1 sample
df_large_error.head()

# Calculate MSE with large error
df_large_error["squared_differences"].mean()

# Finally using the scoring parameter
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

clf = RandomForestClassifier(n_estimators=100)

np.random.seed(42)

# Cross-validation accuracy
cv_acc = cross_val_score(clf, X, y, cv=5, scoring=None) # if scoring=None, esitmator's default scoring evaulation metric is used (accuracy for classification models)
cv_acc

# Cross-validated accuracy
print(f"The cross-validated accuracy is: {np.mean(cv_acc)*100:.2f}%")

np.random.seed(42)

cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
cv_acc

# Cross-validated accuracy
print(f"The cross-validated accuracy is: {np.mean(cv_acc)*100:.2f}%")

# Precision
np.random.seed(42)
cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision")
cv_precision

# Cross-validated precision
print(f"The cross-validated precision is: {np.mean(cv_precision)}")

# Recall
np.random.seed(42)
cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall")
cv_recall

# Cross-validated recall
print(f"The cross-validated recall is: {np.mean(cv_recall)}")

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

X = housing_df.drop("target", axis=1)
y = housing_df["target"]

model = RandomForestRegressor(n_estimators=100)

np.random.seed(42)
cv_r2 = cross_val_score(model, X, y, cv=3, scoring=None)
np.mean(cv_r2)

cv_r2

# Mean squared error
cv_mse = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
np.mean(cv_mse)

cv_mse

# Mean absolute error
cv_mae = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
np.mean(cv_mae)

cv_mae

### 4.3 Using different evaluation metrics as scikit-learn functions
The 3rd way to evaluate scikit-learn machine learning models/estimators is to using the sklearn.metrics module - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Create X & y
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
clf = RandomForestClassifier()

# Fit model
clf.fit(X_train, y_train)

# Make predictions
y_preds = clf.predict(X_test)

# Evaluate model using evaluation functions
print("Classifier metrics on the test set")
print(f"Accurracy: {accuracy_score(y_test, y_preds)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_preds)}")
print(f"Recall: {recall_score(y_test, y_preds)}")
print(f"F1: {f1_score(y_test, y_preds)}")


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Create X & y
X = housing_df.drop("target", axis=1)
y = housing_df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = RandomForestRegressor()

# Fit model
model.fit(X_train, y_train)

# Make predictions
y_preds = model.predict(X_test)

# Evaluate model using evaluation functions
print("Regression metrics on the test set")
print(f"R2 score: {r2_score(y_test, y_preds)}")
print(f"MAE: {mean_absolute_error(y_test, y_preds)}")
print(f"MSE: {mean_squared_error(y_test, y_preds)}")


Machine Learning Model Evaluation - Consolidated 
------------------------------------------------

Evaluating the results of a machine learning model is as important as building one.

But just like how different problems have different machine learning models, 
different machine learning models have different evaluation metrics.

Below are some of the most important evaluation metrics you'll want 
to look into for classification and regression models.


Classification Model Evaluation Metrics/Techniques
--------------------------------------------------
1. Accuracy - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0.

2. Precision - Indicates the proportion of positive identifications (model predicted class 1) 
which were actually correct. A model which produces no false positives has a precision of 1.0.

3. Recall - Indicates the proportion of actual positives which were correctly classified. 
A model which produces no false negatives has a recall of 1.0.

4. F1 score - A combination of precision and recall. A perfect model achieves an F1 score of 1.0.

5. Confusion matrix - Compares the predicted values with the true values in a tabular way, 
if 100% correct, all values in the matrix will be top left to bottom right (diagonal line).

6. Cross-validation - Splits your dataset into multiple parts and train and 
tests your model on each part then evaluates performance as an average.

7. Classification report - Sklearn has a built-in function called classification_report() 
which returns some of the main classification metrics such as precision, recall and f1-score.

8. ROC Curve - Also known as receiver operating characteristic is a plot of true positive rate 
versus false-positive rate.

9. Area Under Curve (AUC) Score - The area underneath the ROC curve. 
A perfect model achieves an AUC score of 1.0.

Which classification metric should you use?
-------------------------------------------
1. Accuracy is a good measure to start with if all classes are balanced 
(e.g. same amount of samples which are labelled with 0 or 1).

2. Precision and recall become more important when classes are imbalanced.

3. If false-positive predictions are worse than false-negatives, 
aim for higher precision.

4. If false-negative predictions are worse than false-positives, aim for higher recall.

5. F1-score is a combination of precision and recall.

6. A confusion matrix is always a good way to visualize how a classification model is going.


Regression Model Evaluation Metrics/Techniques
----------------------------------------------
1. R^2 (pronounced r-squared) or the coefficient of determination - 
Compares your model's predictions to the mean of the targets. 
Values can range from negative infinity (a very poor model) to 1. 
For example, if all your model does is predict the mean of the targets, 
its R^2 value would be 0. 
And if your model perfectly predicts a range of numbers it's R^2 value would be 1.

2. Mean absolute error (MAE) - The average of the absolute differences between 
predictions and actual values. It gives you an idea of how wrong your predictions were.

3. Mean squared error (MSE) - The average squared differences between predictions and 
actual values. Squaring the errors removes negative errors. 
It also amplifies outliers (samples which have larger errors).

Which regression metric should you use?
---------------------------------------

1. R2 is similar to accuracy. 
It gives you a quick indication of how well your 
model might be doing. Generally, the closer your R2 value is to 1.0, 
the better the model. But it doesn't really tell exactly how wrong your 
model is in terms of how far off each prediction is.

2. MAE gives a better indication of how far off each of your model's 
predictions are on average.

3. As for MAE or MSE, because of the way MSE is calculated, 
squaring the differences between predicted values and actual values, 
it amplifies larger differences. 
Let's say we're predicting the value of houses (which we are).
-> Pay more attention to MAE: When being $10,000 off is twice as bad as being $5,000 off.
-> Pay more attention to MSE: When being $10,000 off is more than twice as bad as being $5,000 off.


----------------------------------------------------------
For more resources on evaluating a machine learning model, 
be sure to check out the following resources:
----------------------------------------------------------
scikit-learn documentation for metrics and scoring (quantifying the quality of predictions)
https://scikit-learn.org/stable/modules/model_evaluation.html

Beyond Accuracy: Precision and Recall by Will Koehrsen
https://medium.com/towards-data-science/beyond-accuracy-precision-and-recall-3da06bea9f6c

Stack Overflow answer describing MSE (mean squared error) and RSME (root mean squared error)
https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python/37861832#37861832


```


## Improving a model
```xml
Refer to 06-Improving-The-Model.ipynb

First predictions = baseline predictions. First model = baseline model.

From a data perspective:
1. Could we collect more data? (generally, the more data, the better)
2. Could we improve our data?

From a model perspective:
1. Is there a better model we could use?
2. Could we improve the current model?

Hyperparameters vs. Parameters
1. Parameters = model find these patterns in data
2. Hyperparameters = settings on a model you can adjust to (potentially) improve its ability to find patterns

Three ways to adjust hyperparameters:
1. By hand
2. Randomly with RandomSearchCV
3. Exhaustively with GridSearchCV


1 Tuning hyperparameters by hand
--------------------------------
# Let's make 3 sets, training, validation and test.

clf.get_params()

# We're going to try and adjust:
max_depth
max_features
min_samples_leaf
min_samples_split
n_estimators

def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    return metric_dict


# Shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split the data into train, validation & test sets
train_split = round(0.7 * len(heart_disease_shuffled)) # 70% of data
valid_split = round(train_split + 0.15 * len(heart_disease_shuffled)) # 15% of data
X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test, y_test = X[valid_split:], y[:valid_split]

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make baseline predictions
y_preds = clf.predict(X_valid)

# Evaluate the classifier on validation set
baseline_metrics = evaluate_preds(y_valid, y_preds)
baseline_metrics

np.random.seed(42)

# Create a second classifier with different hyperparameters
clf_2 = RandomForestClassifier(n_estimators=100)
clf_2.fit(X_train, y_train)

# Make predictions with different hyperparameters
y_preds_2 = clf_2.predict(X_valid)

# Evalute the 2nd classsifier
clf_2_metrics = evaluate_preds(y_valid, y_preds_2)

np.random.seed(42)

# Create a third classifier with different hyperparameters
clf_3 = RandomForestClassifier(n_estimators=100, max_depth=10)
clf_3.fit(X_train, y_train)

# Make predictions with different hyperparameters
y_preds_3 = clf_3.predict(X_valid)

# Evalute the 3rd classsifier
clf_3_metrics = evaluate_preds(y_valid, y_preds_3)


2. Hyperparameter tuning with RandomizedSearchCV
------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV

grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200],
        "max_depth": [None, 5, 10, 20, 30],
        "max_features": [None, "sqrt"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4]}

np.random.seed(42)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Setup RandomizedSearchCV
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid, 
                            n_iter=10, # number of models to try
                            cv=5,
                            verbose=2)

# Fit the RandomizedSearchCV version of clf
rs_clf.fit(X_train, y_train);


rs_clf.best_params_

# Make predictions with the best hyperparameters
rs_y_preds = rs_clf.predict(X_test)

# Evaluate the predictions
rs_metrics = evaluate_preds(y_test, rs_y_preds)


3. Hyperparameter tuning with GridSearchCV
------------------------------------------
grid

grid_2 = {'n_estimators': [100, 200, 500],
          'max_depth': [None],
          'max_features': [None, 'sqrt'],
          'min_samples_split': [6],
          'min_samples_leaf': [1, 2]}

from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed(42)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Setup GridSearchCV
gs_clf = GridSearchCV(estimator=clf,
                      param_grid=grid_2, 
                      cv=5,
                      verbose=2)

# Fit the GridSearchCV version of clf
gs_clf.fit(X_train, y_train);

gs_clf.best_params_

gs_y_preds = gs_clf.predict(X_test)

# evaluate the predictions
gs_metrics = evaluate_preds(y_test, gs_y_preds)

# Let's compare our different models metrics.

compare_metrics = pd.DataFrame({"baseline": baseline_metrics,
                                "clf_2": clf_2_metrics,
                                "random search": rs_metrics,
                                "grid search": gs_metrics})

compare_metrics.plot.bar(figsize=(10, 8));


Note:
-----
When comparing models, you should be careful to make sure they're 
compared on the same splits of data.
For example, let's say you have model_1 and model_2 which each differ slightly.
If you want to compare and evaluate their results, model_1 and 
model_2 should both be trained on the same data (e.g. X_train and y_train) and 
their predictions should each be made on the same data, for example:
model_1.fit(X_train, y_train) -> model_1.predict(X_test) -> model_1_preds
model_2.fit(X_train, y_train) -> model_2.predict(X_test) -> model_2_preds

Example: 
https://colab.research.google.com/drive/1ISey96a5Ag6z2CvVZKVqTKNWRwZbZl0m

```


## Saving and loading trained machine learning models
```xml
Two ways to save and load machine learning models:
    1. With Python's pickle module
    2. With the joblib module

1. Pickle
---------
import pickle

# Save an extisting model to file
pickle.dump(gs_clf, open("gs_random_random_forest_model_1.pkl", "wb"))

# Load a saved model
loaded_pickle_model = pickle.load(open("gs_random_random_forest_model_1.pkl", "rb"))

# Make some predictions
pickle_y_preds = loaded_pickle_model.predict(X_test)
evaluate_preds(y_test, pickle_y_preds)


2. Joblib
---------

from joblib import dump, load

# Save model to file
dump(gs_clf, filename="gs_random_forest_model_1.joblib")

# Import a saved joblib model
loaded_joblib_model = load(filename="gs_random_forest_model_1.joblib")

# Make and evaluate joblib predictions
joblib_y_preds = loaded_joblib_model.predict(X_test)
evaluate_preds(y_test, joblib_y_preds)

```

## Putting it all together!
```xml
data = pd.read_csv("resources/car-sales-extended-missing-data.csv")
data

data.dtypes

data.isna().sum()

# Steps we want to do (all in one cell):
# Fill missing data
# Convert data to numbers
# Build a model on the data


# Getting data ready
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Setup random seed
import numpy as np
np.random.seed(42)

# Import data and drop rows with missing labels
data = pd.read_csv("resources/car-sales-extended-missing-data.csv")
data.dropna(subset=["Price"], inplace=True)

# Define different features and transformer pipeline
categorical_features = ["Make", "Colour"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

door_feature = ["Doors"]
door_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=4))
])

numeric_features = ["Odometer (KM)"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

# Setup preprocessing steps (fill missing values, then convert to numbers)
preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", categorical_transformer, categorical_features),
                        ("door", door_transformer, door_feature),
                        ("num", numeric_transformer, numeric_features)
                    ])

# Creating a preprocessing and modelling pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("model", RandomForestRegressor())])

# Split data
X = data.drop("Price", axis=1)
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit and score the model
model.fit(X_train, y_train)
model.score(X_test, y_test)


# It's also possible to use GridSearchCV or RandomizedSesrchCV with our Pipeline.

# Use GridSearchCV with our regression Pipeline
from sklearn.model_selection import GridSearchCV

pipe_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "model__n_estimators": [100, 1000],
    "model__max_depth": [None, 5],
    "model__max_features": [None],
    "model__min_samples_split": [2, 4]    
}

gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)

gs_model.score(X_test, y_test)

# Reference:
https://colab.research.google.com/drive/1AX3Llawt0zdjtOxaYuTZX69dhxwinFDi?usp=sharing

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
