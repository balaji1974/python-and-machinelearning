
# Machine Learning Environment Setup 

## Basics Data Science Tools
```xml
Data Science Platform 
---------------------
Anaconda 
Jupyter Notebooks 


Data Analysis Tools
-------------------
Pandas
Matplotlibs
NumPy


Data Modelling Tools:
--------------------
TensorFlow
PyTorch
Sci-kit Learn
XGBoost 
CatBoost


```

## Anaconda, Miniconda, Conda - Introduction 
```xml
Anaconda (software distribution) -> 
	Size 3 GB and contains all things needed for Data Science 
MiniConda (software distribution) -> 
	Size 200 MB and contains just the basics needed for Data Science 
Conda (package manager) -> 
	An assistance tool that comes with MiniConda


Anaconda can be thought of the data scientists hardware store. 
It's got everything you need. From tools for exploring datasets, 
to tools for modelling them, to tools for visualising what you've found. 
Everyone can access the hardware store and all the tools inside.

Miniconda is the workbench of a data scientist. 
Every workbench starts clean with only the bare necessities. 
But as a project grows, so do the number of tools on the workbench. 
They get used, they get changed, they get swapped. 
Each workbench can be customised however a data scientist wants. 
One data scientists workbench may be completely different to another, 
even if they're on the same team.

Conda helps to organise all of these tools. 
Although Anaconda comes with many of them ready to go, 
sometimes they'll need changing. Conda is like the assistant 
who takes stock of all the tools. The same goes for Miniconda.


```


## Conda - Introduction 
```xml
Conda Documentation: 
https://docs.conda.io/en/latest/

Getting Started with Conda: 
https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html

Miniconda Mac Setup
-------------------
https://docs.anaconda.com/miniconda/

Download miniconda
------------------
https://www.anaconda.com/download/


Steps:
------
1. Download Miniconda 
For Mac installation download the .pkg installer 
For Windows installation download the graphical installer for windows 

2. Install Minicoda
Click the package manager that was downloaded and double click to install

3. Create a working directory
Go to command prompt
Create a dirctory called datascience -> mkdir datascience 
Change to this directory -> cd datascience 

4. Create a working environment - Download and install all the necessary packages 
Run the following commnand in this directory 
conda create --prefix ./env pandas numpy matplotlib scikit-learn

5. Activte the environment 
conda activate /<your base path>/datascience/env 

6. List the active environment 
conda env list

7. Install Jupyter 
note: could have installed in the first step itself along with other packages 
conda install jupyter 

8. Test Jupyter -> This will open the Jupyter environment 
jupyter notebook

9. Test the python interface using jupyter 
New -> (click) -> Python3 -> (click)

Here you can run any python command and test 

10. Test the packages that were downloaded on jupyter notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as ptl
import sklearn

and hit run  

11. Save the workspace 
Rename the workspace to your name by clicking the Untitled name 
on the top of the browser and save it. 

12. Close the workspace 
From the command line, enter cnt+c 
and confirm 

13. Deactivte the environment 
conda deactivate 


For other operating systems , please follow the below link
https://docs.anaconda.com/miniconda/install/#quick-command-line-install


```


## Sharing the Environment with others 
```xml
There a couple of ways to do this:
1. Share your entire project folder (including the environment folder 
containing all of your Conda packages).

The benefit of this is it's a very simple setup, share the folder, 
activate the environment, run the code. 
However, an environment folder can be quite a large file to share.

2. Share a .yml (pronounced YAM-L) file of your Conda environment.

A .yml is basically a text file with instructions to tell Conda 
how to set up an environment.

For example, to export the environment we created earlier at 
/<base-path>/project_1/env as a YAML file called environment.yml 
we can use the command:
conda env export --prefix /<base-path>/project_1/env > environment.yml

After running the export command, we can see our new .yml file stored as environment.yml.

A sample .yml file might look like the following:
name: my_ml_env
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - jupyter
  - matplotlib

Of course, your actual file will depend on the packages you've installed in your environment.
For more on sharing an environment, check out the Conda documentation on sharing environments.

Finally, to create an environment called env_from_file from a .yml file 
called environment.yml, you can run the command:
conda env create --file environment.yml --name env_from_file

For more on creating an environment from a .yml file, 
check out the Conda documentation on creating an environment from a .yml file.

```

## Jupyter notebook 
```xml
Documentation 
https://jupyter-notebook.readthedocs.io/en/stable/

Quick Tutorial
https://www.dataquest.io/blog/jupyter-notebook-tutorial/


Upload any file into the environment using the upload button of the Jupyter notebook

Running a command or program ->
Either click Run or press <shift+enter>


Mark the lines as markdown 
## Heart Disease Project
This is a smaple project


Create a code and run:
import pandas as pd
df = pd.read_csv("heart-disease.csv")
df.head(10)


Create a graph and run:
import matplotlib.pyplot as ptl 
df.target.value_counts().plot(kind="bar")


Markdown code to import image
![](6-step-ml-framework.png)


So we can run code, images and text within the same notebook environment

!ls or ls -> will list the files in the notebook environment 
Any terminal command can be run this way. 


ctl+s -> for saving the notebook 

From command line 
ctl+c -> For shutting down the notebook 

conda deactivate -> to come out of the project 


```

### Anaconda
```xml
Go to the Anaconda download page and download:
https://www.anaconda.com/download/

Download and install anaconda from the package manager

Next install the conda navigator with the following command
from the command line: 
conda install anaconda-navigator

Next run the following command from command line:
anaconda-navigator

Go to environments link from the left 
Create a new enviroment:
from the anaconda-navigator -> click "+ button" at the bottom of the 
navigator

For the top filter -> select not installed packages and 
select the following 3 packages (use search for easy package selection):
pandas
bottleneck
numexpr
click the apply button -> This will install the 3 packages selected


Now from the Home menu, click and install the Jupiter Lab and 
if already installed, then launch it. 

```

## Importing Libraries into Jupyter notebook 
```xml
Importing pandas into Jupyter notebook
import pandas as pd 



```


### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/
https://www.mrdbourke.com/get-your-computer-ready-for-machine-learning-using-anaconda-miniconda-and-conda/
https://docs.conda.io/en/latest/
https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
https://docs.anaconda.com/miniconda/
https://jupyter-notebook.readthedocs.io/en/stable/
https://www.dataquest.io/blog/jupyter-notebook-tutorial/

https://www.udemy.com/course/data-analysis-with-pandas
https://jupyter.org/

```