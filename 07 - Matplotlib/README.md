
# Matplotlib

## Package requirements 
```xml
Make sure that the following packages are available in your Jupyter notebook environment
matplotlib
matplotlib-base
matplotlib-inline

run the following in a a new notebook to check that the empty chart loads up 
# Import matplotlib and setup the figures to display within the notebook
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

plt.plot()
plt.show()

```

## Different methods of plotting 
```xml

# Let's add some data on x-axix
plt.plot([1, 2, 3, 4])


# Create some data on x and y-axis
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]

# With a semi-colon and now a y value
plt.plot(x, y);


# Creating a plot with the Object Orientied verison
fig = plt.figure()
ax = fig.add_subplot()
plt.show()


# Second method 
fig = plt.figure()
ax = fig.add_axes([1, 1, 1, 1])
ax.plot(x, y)
plt.show()


# Easier and more robust going forward (what we're going to use)
fig, ax = plt.subplots()
ax.plot(x, y);

```

### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/learn/

https://matplotlib.org/stable/index.html

```