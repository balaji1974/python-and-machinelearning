
# Complete Projects

## Creating an environment from exisitng environment
```xml 
1. Look at the list of environments available
conda env list 

2. Activate the environment that you want to copy 
conda activate <full-path-to-environment>/env
Eg. 
conda activate  /opt/miniconda3/envs/datascience/env

3. Export everything from the active environment
conda env export > environment.yml

4. View the content of the yml file
vim environment.yml
(to quit) <esc>:q

5. Deactivate the current environment
conda deactivate

6. Create a new project folder and change to it
mkdir <full-path-to-project>
cd <full-path-to-project>
Eg.
mkdir /Users/balaji/python/10-CompleteProject
cd /Users/balaji/python/10-CompleteProject

7. Import packages into the new environment
conda env create --prefix ./env -f <path-to-the-yml-file>/<yml-file-name>
Eg. 
conda env create --prefix ./env -f /Users/balaji/datascience/scikit-learn/environment.yml

Another way to create this will be to spell out the exact packages 
that we want in this environment:
Eg. 
conda env create --prefix ./env numpy pandas matplotlib sklearn

8. Active the new environment created
conda activate <full-path-to-environment>
eg. 
conda activate /Users/balaji/datascience/scikit-learn/env

9. Copy our data files into a data folder inside this environment 
mkdir data
copy <path-to-source-folder>/*.csv <path-to-destination-folder>/data/*.csv 


```
## End to end classification problem
```xml 
Check all the details in the self explanatory notebook file:
01-End-To-End-Classification-Problem.ipynb

```


## To do 
#### Look into other classifcation datasets at the below links and practise:
https://www.kaggle.com/datasets?tags=13302-Classification
https://archive.ics.uci.edu/datasets


## End to end regression problem
```xml 
Check all the details in the self explanatory notebook file:
02-End-To-End-Regression-Problem.ipynb

```

## XGBoost and CatBoost samples
```xml 
Check the sample in the following notebook file:
03-XGBoost-And-CatBoost.ipynb

```



### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/learn/lecture/17684274
```
