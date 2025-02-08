
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
# Import dataset
heart_disease = pd.read_csv("./heart-disease.csv")

# View the data
heart_disease.head()

# Create X (all the feature columns except target column)
X = heart_disease.drop("target", axis=1)

# Create y (the target column)
y = heart_disease["target"]

```

## 2. Choosing the right model and hyper parameters
```xml 
# Random Forest Classifier (for classification problems)
from sklearn.ensemble import RandomForestClassifier

# Instantiating a Random Forest Classifier (clf short for classifier)
clf = RandomForestClassifier()

# We will keep the default parameters 
clf.get_params() # checking the default parameters

```

## 3. Fit the model to the training data 
```xml 
# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

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
    clf = RandomForestClassifier(n_estimators=i).fit(X_test, y_test)
    print(f"Model accuracy on test set : {clf.score(X_test, y_test) * 100:.2f}%")
    print("")


```
## 4. Save the model and load it
```xml 

# Saving a model with pickle
import pickle

# Save an existing model to file
pickle.dump(clf, open("rs_random_forest_model_1.pkl", "wb"))

# Load a saved pickle model
loaded_pickle_model = pickle.load(open("rs_random_forest_model_1.pkl", "rb"))

# Evaluate loaded model
loaded_pickle_model.score(X_test, y_test)

```



### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery
https://scikit-learn.org/stable/user_guide.html

```
