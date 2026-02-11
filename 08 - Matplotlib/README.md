
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

## Anatomy of Plotting & Workflow
![alt text](https://github.com/balaji1974/python-and-machinelearning/blob/main/08%20-%20Matplotlib/images/matplotlib-anatomy-of-a-plot-with-code.png?raw=true)
```xml
# This is where the object orientated name comes from 
type(fig), type(ax)


# A matplotlib workflow

# 0. Import and get matplotlib ready
%matplotlib inline
import matplotlib.pyplot as plt

# 1. Prepare data
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10,10))

# 3. Plot data
ax.plot(x, y)

# 4. Customize plot
ax.set(title="Sample Simple Plot", xlabel="x-axis", ylabel="y-axis")

# 5. Save & show
fig.savefig("./images/simple-plot.png")

```

## Create a line plot, scatter plot, bar plot, histogram
```xml
# Create an array
x = np.linspace(0, 10, 100) # From 0 to 10 it creates 100 numbers 

# The default plot is Line
fig, ax = plt.subplots()
ax.plot(x, x**2); # default is line plot

# Scatter plot
# Need to recreate our figure and axis instances when we want a new figure
fig, ax = plt.subplots()
ax.scatter(x, np.exp(x)); # scatter plot 

# Another scatter plot with different data in y-axis 
fig, ax = plt.subplots()
ax.scatter(x, np.sin(x)); # scatter plot 

# Bar plot
# You can make plots from a dictionary
nut_butter_prices = {"Almond butter": 10,
                     "Peanut butter": 8,
                     "Cashew butter": 12}
fig, ax = plt.subplots()
ax.bar(nut_butter_prices.keys(), nut_butter_prices.values()) # bar plot from dict
ax.set(title="Balaji's Nut Butter Store", ylabel="Price ($)");

# Bar plot - horizontal 
fig, ax = plt.subplots()
# note Dict cannot be passed and it has to be turned to a List
ax.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values())); 

# Histogram 
# Make some data from a normal distribution
x = np.random.randn(1000) # pulls data from a normal distribution
fig, ax = plt.subplots()
ax.hist(x); # Histogram 


```

## Create a subplot  
```xml
# Creating Figures with multiple Axes with Subplots
# Subplots allow you to create multiple Axes on the same Figure 
# (multiple plots within the same plot).

# Subplots are helpful because you start with one plot per Figure 
# but scale it up to more when necessary.

# For example, let's create a subplot that shows many of the 
# above datasets on the same Figure.

# We can do so by creating multiple Axes with plt.subplots() and 
# setting the nrows (number of rows) and ncols (number of columns) 
# parameters to reflect how many Axes we'd like.

# nrows and ncols parameters are multiplicative, 
# meaning plt.subplots(nrows=2, ncols=2) will create 2*2=4 total Axes.

# Option 1: 
# Create 4 subplots with each Axes having its own variable name
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, 
                                   ncols=2, figsize=(10, 5))

# Plot data to each axis
ax1.plot(x, x/2);
ax2.scatter(np.random.random(10), np.random.random(10));
ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax4.hist(np.random.randn(1000));


# Option 2: 
# Create 4 subplots with a single ax variable
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

# Index the ax variable to plot data
ax[0, 0].plot(x, x/2);
ax[0, 1].scatter(np.random.random(10), np.random.random(10));
ax[1, 0].bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax[1, 1].hist(np.random.randn(1000));

```

## Plotting from Pandas Dataframe
```xml
import pandas as pd

# Start with some dummy data
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2025', periods=1000))
ts

# Add up the values cumulatively
ts = ts.cumsum()

# Plot the values over time with a line plot 
# (note: both of these will return the same thing)
# ts.cumsum().plot() # kind="line" is set by default
ts.cumsum().plot(kind="line");

# Let's import the car_sales dataset 
# Note: The following two lines load the same data, one does it from a local file path, the other does it from a URL.
car_sales = pd.read_csv("./resource/car-sales.csv") # load data from local file
car_sales

# Remove price column symbols
car_sales["Price"] = car_sales["Price"].str.replace('[\\$\\,\\.]', '', 
                          regex=True) # Tell pandas to replace using regex
car_sales

# Remove last two zeros
car_sales["Price"] = car_sales["Price"].str[:-2]
car_sales

# Add a date column
car_sales["Sale Date"] = pd.date_range("1/1/2024", periods=len(car_sales))
car_sales

car_sales.plot(x='Sale Date', y='Total Sales');


# Note: In previous versions of matplotlib and pandas, 
# have the "Price" column as a string would return an error
car_sales["Price"] = car_sales["Price"].astype(str)

# Plot a scatter plot
car_sales.plot(x="Odometer (KM)", y="Price", kind="scatter");

# Convert Price to int
car_sales["Price"] = car_sales["Price"].astype(int)

# Plot a scatter plot
car_sales.plot(x="Odometer (KM)", y="Price", kind='scatter');

# Bar plots 
# Create 10 random samples across 4 columns
x = np.random.rand(10, 4)
x

# Turn the data into a DataFrame
df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
df

# Plot a bar chart
df.plot.bar();

# Plot a bar chart with the kind parameter
df.plot(kind='bar');

# Plot a bar chart from car_sales DataFrame
car_sales.plot(x="Make", 
               y="Odometer (KM)", kind="bar");

# Histogram plot from a pandas DataFrame
car_sales["Odometer (KM)"].plot.hist(bins=10); # default number of bins (or groups) is 10
# or 
car_sales["Odometer (KM)"].plot(kind="hist"); # default number of bins (or groups) is 10

# Create a histogram of the Price column
car_sales["Price"].plot.hist(bins=10);

# Import the heart disease dataset
# load from local file path 
heart_disease = pd.read_csv("./resources/heart-disease.csv") 
heart_disease.head()

# Create a histogram of the age column
heart_disease["age"].plot.hist(bins=50);

# Inspect the data
heart_disease.head()

# Since all of our columns are numeric in value, 
# let's try and create a histogram of each column.
heart_disease.plot.hist(figsize=(5, 20), 
                        subplots=True);

```

## Plotting more advanced plots from a pandas DataFrame 
```xml
# Perform data analysis on patients over 50
over_50 = heart_disease[heart_disease["age"] > 50]
over_50

# Create a scatter plot directly from the pandas DataFrame
over_50.plot(kind="scatter",
             x="age", y="chol", c="target", # colour the dots by target value
             figsize=(10, 6));



# We can recreate the same plot using plt.subplots() and 
# then passing the Axes variable (ax) to the pandas plot() method.
# Create a Figure and Axes instance
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data from the DataFrame to the ax object
over_50.plot(kind="scatter", 
             x="age", 
             y="chol", 
             c="target", 
             ax=ax); # set the target Axes

# Customize the x-axis limits (to be within our target age ranges)
ax.set_xlim([45, 100]);



# Now instead of plotting directly from the pandas DataFrame, 
# we can make a bit more of a comprehensive plot by plotting data 
# directly to a target Axes instance.
# Create Figure and Axes instance
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data directly to the Axes intance
scatter = ax.scatter(over_50["age"], 
                     over_50["chol"], 
                     c=over_50["target"]) # Colour the data with the "target" column

# Customize the plot parameters 
ax.set(title="Heart Disease and Cholesterol Levels",
       xlabel="Age",
       ylabel="Cholesterol");

# Setup the legend
ax.legend(*scatter.legend_elements(), 
          title="Target");



# What if we wanted a horizontal line going across with 
# the mean of heart_disease["chol"]?
# We do so with the Axes.axhline() method.
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
scatter = ax.scatter(over_50["age"], 
                     over_50["chol"], 
                     c=over_50["target"])

# Customize the plot
ax.set(title="Heart Disease and Cholesterol Levels",
       xlabel="Age",
       ylabel="Cholesterol");

# Add a legned
ax.legend(*scatter.legend_elements(), 
          title="Target")

# Add a meanline
ax.axhline(over_50["chol"].mean(),
           linestyle="--"); # style the line to make it look nice



# Plotting multiple plots on the same figure 
# (adding another plot to an existing one)
# Sometimes you'll want to visualize multiple features of 
# a dataset or results of a model in one Figure.
# You can achieve this by adding data to multiple Axes 
# on the same Figure.
# The plt.subplots() method helps you create Figures with 
# a desired number of Axes in a desired figuration.
# Using nrows (number of rows) and ncols (number of columns) 
# parameters you can control the number of Axes on the Figure.

# For example:
# nrows=2, ncols=1 = 2x1 = a Figure with 2 Axes
# nrows=5, ncols=5 = 5x5 = a Figure with 25 Axes
# Let's create a plot with 2 Axes.
# One the first Axes (Axes 0), 
# we'll plot heart disease against cholesterol levels (chol).
# On the second Axes (Axis 1), 
# we'll plot heart disease against max heart rate levels (thalach).

# Setup plot (2 rows, 1 column)
fig, (ax0, ax1) = plt.subplots(nrows=2, # 2 rows
                               ncols=1, # 1 column 
                               sharex=True, # both plots should use the same x-axis 
                               figsize=(10, 8))

# ---------- Axis 0: Heart Disease and Cholesterol Levels ----------

# Add data for ax0
scatter = ax0.scatter(over_50["age"], 
                      over_50["chol"], 
                      c=over_50["target"])
# Customize ax0
ax0.set(title="Heart Disease and Cholesterol Levels",
        ylabel="Cholesterol")
ax0.legend(*scatter.legend_elements(), title="Target")

# Setup a mean line
ax0.axhline(y=over_50["chol"].mean(), 
            color='b', 
            linestyle='--', 
            label="Average")

# ---------- Axis 1: Heart Disease and Max Heart Rate Levels ----------

# Add data for ax1
scatter = ax1.scatter(over_50["age"], 
                      over_50["thalach"], 
                      c=over_50["target"])

# Customize ax1
ax1.set(title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate")
ax1.legend(*scatter.legend_elements(), title="Target")

# Setup a mean line
ax1.axhline(y=over_50["thalach"].mean(), 
            color='b', 
            linestyle='--', 
            label="Average")

# Title the figure
fig.suptitle('Heart Disease Analysis', 
             fontsize=16, 
             fontweight='bold');

```

## Customizing your Plots
```xml
# Some of the things you can customize include:
# Axis limits - The range in which your data is displayed.
# Colors - That colors appear on the plot to represent different data.
# Overall style - Matplotlib has several different styles built-in which offer 
# different overall themes for your plots, you can see examples of these in 
# the matplotlib style sheets reference documentation.
# Legend - One of the most informative pieces of information on a Figure can be 
# the legend, you can modify the legend of an Axes with the plt.legend() method.

# Check the available styles
plt.style.available

# Plot before changing style
car_sales["Price"].plot();

# Change the style of our future plots
plt.style.use("seaborn-v0_8-whitegrid")

# Plot the same plot as before
car_sales["Price"].plot();

# Change the plot style
plt.style.use("fivethirtyeight")
car_sales["Price"].plot();

# Try scatter plot 
car_sales.plot(x="Odometer (KM)", 
               y="Price", kind="scatter");

# Change the plot style
plt.style.use("ggplot")
car_sales["Price"].plot.hist(bins=10);

# Change the plot style back to the default 
plt.style.use("default")
car_sales["Price"].plot.hist();

# Create random data
x = np.random.randn(10, 4)
# Turn data into DataFrame with simple column names
df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])

# Create a bar plot
ax = df.plot(kind="bar")

# Check the type of the ax variable
type(ax)

# Recreate the ax object
ax = df.plot(kind="bar")

# Set various attributes
ax.set(title="Random Number Bar Graph from DataFrame", 
       xlabel="Row number", ylabel="Random number");


# Recreate the ax object
ax = df.plot(kind="bar")

# Set various attributes
ax.set(title="Random Number Bar Graph from DataFrame", 
       xlabel="Row number", 
       ylabel="Random number")

# Change the legend position
ax.legend(loc="upper right");


# Customizing the colours of plots with colormaps (cmap)
# Setup the Figure and Axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot with no cmap change (use default colormap)
scatter = ax.scatter(over_50["age"], 
                     over_50["chol"], 
                     c=over_50["target"],
                     cmap="viridis") # default cmap value

# Add attributes to the plot
ax.set(title="Heart Disease and Cholesterol Levels",
       xlabel="Age",
       ylabel="Cholesterol");
ax.axhline(y=over_50["chol"].mean(), 
           c='b', 
           linestyle='--', 
           label="Average");
ax.legend(*scatter.legend_elements(), 
          title="Target");


# How about we try cmap="winter"
fig, ax = plt.subplots(figsize=(10, 6))

# Setup scatter plot with different cmap
scatter = ax.scatter(over_50["age"], 
                     over_50["chol"], 
                     c=over_50["target"], 
                     cmap="winter") # Change cmap value 

# Add attributes to the plot with different color line
ax.set(title="Heart Disease and Cholesterol Levels",
       xlabel="Age",
       ylabel="Cholesterol")
ax.axhline(y=over_50["chol"].mean(), 
           color="r", # Change color of line to "r" (for red)
           linestyle='--', 
           label="Average");
ax.legend(*scatter.legend_elements(), 
          title="Target");


# Customizing the xlim & ylim
# Recreate double Axes plot from above with colour updates 
fig, (ax0, ax1) = plt.subplots(nrows=2, 
                               ncols=1, 
                               sharex=True, 
                               figsize=(10, 7))

# ---------- Axis 0 ----------
scatter = ax0.scatter(over_50["age"], 
                      over_50["chol"], 
                      c=over_50["target"],
                      cmap="winter")
ax0.set(title="Heart Disease and Cholesterol Levels",
        ylabel="Cholesterol")

# Setup a mean line
ax0.axhline(y=over_50["chol"].mean(), 
            color="r", 
            linestyle="--", 
            label="Average");
ax0.legend(*scatter.legend_elements(), title="Target")

# ---------- Axis 1 ----------
scatter = ax1.scatter(over_50["age"], 
                      over_50["thalach"], 
                      c=over_50["target"],
                      cmap="winter")
ax1.set(title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate")

# Setup a mean line
ax1.axhline(y=over_50["thalach"].mean(), 
            color="r", 
            linestyle="--", 
            label="Average");
ax1.legend(*scatter.legend_elements(), 
           title="Target")

# Title the figure
fig.suptitle("Heart Disease Analysis", 
             fontsize=16, 
             fontweight="bold");


# Now let's recreate the plot from above but this time we'll change the axis limits.
# We can do so by using Axes.set(xlim=[50, 80]) or Axes.set(ylim=[60, 220]) 
# where the inputs to xlim and ylim are a list of integers defining a range of values.
# For example, xlim=[50, 80] will set the x-axis values to start at 50 and end at 80.
# Recreate the plot from above with custom x and y axis ranges
fig, (ax0, ax1) = plt.subplots(nrows=2, 
                               ncols=1, 
                               sharex=True, 
                               figsize=(10, 7))
scatter = ax0.scatter(over_50["age"], 
                      over_50["chol"], 
                      c=over_50["target"],
                      cmap='winter')
ax0.set(title="Heart Disease and Cholesterol Levels",
        ylabel="Cholesterol",
        xlim=[50, 80]) # set the x-axis ranges 

# Setup a mean line
ax0.axhline(y=over_50["chol"].mean(), 
            color="r", 
            linestyle="--", 
            label="Average");
ax0.legend(*scatter.legend_elements(), title="Target")

# Axis 1, 1 (row 1, column 1)
scatter = ax1.scatter(over_50["age"], 
                      over_50["thalach"], 
                      c=over_50["target"],
                      cmap='winter')
ax1.set(title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate",
        ylim=[60, 220]) # change the y-axis range

# Setup a mean line
ax1.axhline(y=over_50["thalach"].mean(), 
            color="r", 
            linestyle="--", 
            label="Average");
ax1.legend(*scatter.legend_elements(), 
           title="Target")

# Title the figure
fig.suptitle("Heart Disease Analysis", 
             fontsize=16, 
             fontweight="bold");

```


## Saving the  Plot
```xml
# You can save matplotlib Figures with plt.savefig(fname="your_plot_file_name") 
# where fname is the target filename you'd like to save the plot to.

# Check the supported filetypes
fig.canvas.get_supported_filetypes()

# Save the file
fig.savefig(fname="./images/heart-disease-analysis.png",
            dpi=100)


```

## Matplotlib - Extras 
```xml
# Resets figure
fig, ax = plt.subplots()


# Potential matplotlib workflow function
def plotting_workflow(data):
    # 1. Manipulate data
    # 2. Create plot
    # 3. Plot data
    # 4. Customize plot
    # 5. Save plot
    # 6. Return plot
    return plot

```


### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/learn/
https://matplotlib.org/stable/index.html
https://matplotlib.org/stable/users/explain/quick_start.html
https://matplotlib.org/stable/plot_types/index.html
https://matplotlib.org/stable/tutorials/lifecycle.html

https://regexone.com/

```