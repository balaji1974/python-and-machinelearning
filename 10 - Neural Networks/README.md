
# Neural Networks

## Introduction
```xml 
What is TensorFlow: 
------------------
TensorFlow is a software library for machine learning and artificial intelligence. 
It can be used across a range of tasks, but is used mainly for training and 
inference of neural networks. It is one of the most popular deep learning frameworks, 
alongside others such as PyTorch and PaddlePaddle - Wikipedia 

Why do we use TensorFlow:
------------------------
We use TensorFlow primarily for building and deploying machine learning (ML) 
and deep learning models. Here’s why it's widely used:

1. Scalability & Performance: 
Efficient computation: Uses optimized operations with 
support for GPUs, TPUs, and distributed computing.
Scalable deployment: Works from mobile devices (TensorFlow Lite) 
to large-scale distributed systems (TensorFlow Serving).

2. Flexibility & Ease of Use: 
Multi-language support: Primarily Python, 
but also supports C++, Java, and Go.
Multiple APIs: High-level APIs like Keras for quick prototyping 
and low-level APIs for custom model development.

3. Strong Ecosystem & Tooling: 
TensorFlow Extended (TFX): A full ML pipeline framework for 
production-ready applications.
TensorBoard: Visualization tools for debugging and optimizing models.
TF Lite & TF.js: Deploy models on mobile, edge devices, or in-browser.

4. Industry Adoption & Community Support: 
Developed by Google and widely adopted across industries, 
making it well-documented with strong community support.

5. Support for Advanced ML & DL Features: 
Pre-trained models & transfer learning.
Reinforcement learning, generative models, and graph-based computations.
Auto-differentiation and gradient-based optimization.

What is neural network:
----------------------
A neural network is a method in artificial intelligence (AI) that teaches 
computers to process data in a way that is inspired by the human brain. 
It is a type of machine learning (ML) process, called deep learning, 
that uses interconnected nodes or neurons in a layered structure that 
resembles the human brain. - Oxford Dictionary

What is deep learning:
----------------------
Deep learning is a type of machine learning that uses artificial neural networks 
to learn from data. Artificial neural networks are inspired by the human brain, 
and they can be used to solve a wide variety of problems, 
including image recognition, natural language processing, and speech recognition.
- Oxford Dictionary

What kind of object detection problems are there:
-------------------------------------------------
1. Classification
Predicting a category or class label from given input data.
Example Use Cases:
Image classification (e.g., detecting cats vs. dogs in images).
Spam detection (classifying emails as spam or not).
Sentiment analysis (classifying text as positive, negative, or neutral).
Common Models: CNNs for images, RNNs/Transformers for text, and MLPs for tabular data.

2. Sequence-to-Sequence (Seq2Seq)
Mapping an input sequence to an output sequence of different or the same length.
Example Use Cases:
Machine translation (e.g., English to French translation).
Speech-to-text (converting audio into transcribed text).
Chatbots and conversational AI (generating responses based on input text).
Common Models: LSTMs, GRUs, Transformer-based models (like T5, GPT, BERT).

3. Object Detection
Identifying and localizing multiple objects within an image or video frame.
Example Use Cases:
Autonomous driving (detecting pedestrians, vehicles, and traffic signs).
Security surveillance (detecting suspicious activities or intruders).
Medical imaging (detecting tumors in X-ray/MRI scans).
Common Models: Faster R-CNN, YOLO, SSD, EfficientDet

What is Transfer Learning
-------------------------
Transfer learning is a deep learning technique where a pre-trained model 
(trained on a large dataset) is fine-tuned or adapted for a different but 
related task. Instead of training a model from scratch, 
we reuse learned features from a previously trained network.

Why Use Transfer Learning
-------------------------
Faster Training – Pre-trained models already have learned general features, 
reducing the need for extensive training.
Less Data Requirement – Since the model has prior knowledge, 
we can achieve good results even with a small dataset.
Improved Performance – Leveraging knowledge from large datasets helps in 
achieving better accuracy, especially in domains with limited data.
Efficient Use of Resources – Saves computational power and time compared 
to training deep models from scratch.

Example Use Cases
Image classification using pre-trained CNNs (e.g., ResNet, VGG, MobileNet).
NLP tasks like sentiment analysis using transformer models (e.g., BERT, GPT).
Medical imaging, where pre-trained models on general images are fine-tuned for specific diseases.


```
## Data to work on and environment setup on Google Colab 
```xml 

Go to the below website (after registering in it)
https://www.kaggle.com/c/dog-breed-identification/overview

Go to Data (from the below link) -> Download all the data 
https://www.kaggle.com/c/dog-breed-identification/data
(dog-breed-identification.zip)

With a google account go to below link:
https://colab.research.google.com/

And start a new notebook
-> This is quite similar to Juypter notebook 
-> This environment is connected to Google Compute Engine at the backend

You can check this by hovering over the right side of the colab on the ram/disk dropdown
Connected to
Python 3 Google Compute Engine backend
RAM: 0.87 GB/12.67 GB
Disk: 37.09 GB/107.72 GB


Google colab FAQ: (please read this to understand colab)
https://research.google.com/colaboratory/faq.html

Next Mount Google Drive in your Colab environment, create a new folder. 
Right click on this folder, upload, point to the file from the local pc 
(dog-breed-identification.zip) and upload it. 

Once the file has been upload, from colab notebook issue the following 
command to unzip the file: 
!unzip "<full-path-to-directory>/dog-breed-identification.zip" -d "<full-path-to-directory>"
(Note: you can also right click on the file and copy its path)

Once the file is unzipped comment the code, to accidently prevent it from running again 



```



## Working with TensorFlow on Google Colab 
```xml 
# import tensorflow
import tensorflow as tf
# Print version of TensorFlow
print ("TensorFlow version : " + tf.__version__)


import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)

# Check for GPU
print("GPU", "available (YESS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")

# This will output result as: (may vary from time to time)
TF version: 2.18.0
Hub version: 0.16.1
GPU not available :(

# Adding a GPU to TensorFlow
By default Colab runs on a computer located on Google's servers which doesn't have a GPU attached to it.
But we can fix this going to runtime and then changing the runtime type:

Go to Runtime.
Click "Change runtime type".
Where it says "Hardware accelerator", choose "GPU" (don't worry about TPU for now but feel free to research them).
Click save.
The runtime will be restarted to activate the new hardware, so you'll have to rerun the above cells.
If the steps have worked you should see a print out saying "GPU available".

# Google Colab Tutorial - This is a notebook by itself which can be copied 
and run in our environment or directly connected to a runtime by clicking the runtime button
on the right side (top)
https://colab.research.google.com/

# TensorFlow with GPU enabled (comparision with CPU)
https://colab.research.google.com/notebooks/gpu.ipynb
Run the TensorFlow speedup on GPU relative to CPU section
but before running, connect it to a hosted runtime and add GPU. 


# To turn any cell from command to markdown 
<cmd> + MM

# To show shortcuts
<cmd> + MH

# To show help of any function
# inside the function press
<cmd> + shift + space

```

## Getting data ready to work inside TensorFlow 
```xml 
# Running this cell will provide you with a token to link your drive to this notebook
from google.colab import drive
drive.mount('/content/drive')


# Checkout the labels of our data 
import pandas as pd
labels_csv = pd.read_csv("/<path-to-file>/labels.csv")
print(labels_csv.describe())
print(labels_csv.head())

# Display the breed column 
labels_csv["breed"]
# How many images are there of each breed?
labels_csv["breed"].value_counts();
# Plot it to a graph
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10));
# Calculate mean 
print(labels_csv["breed"].value_counts().mean())
# Calculate median
print(labels_csv["breed"].value_counts().median())

```

## Preparing the images
```xml 
# Let's check out one of the images.
from IPython.display import display, Image
Image("<path-to-files>/000bec180eb18c7604dcecc8fe0dba07.jpg")

# Create pathnames from image ID's
filenames = ["/content/drive/MyDrive/AI Project/train/" + fname + ".jpg" for fname in labels_csv["id"]]
# Check the first 10 filenames
filenames[:10]

# Now we've got a list of all the filenames from the ID column of labels_csv, 
# we can compare it to the number of files in our training data directory 
# to see if they line up.
# Check whether number of filenames matches number of actual image files
import os
if len(os.listdir("<path-to-files>")) == len(filenames):
  print("Filenames match actual amount of files!")
else:
  print("Filenames do not match actual amount of files, check the target directory.")

# Visualizing directly from a filepath.
# Check an image directly from a filepath
Image(filenames[9000])

```

## Convert labels into numbers
```xml 
# We'll take them from labels_csv and turn them into a NumPy array.
import numpy as np
labels = labels_csv["breed"].to_numpy() # convert labels column to NumPy array
labels[:10]

# See if number of labels matches the number of filenames
if len(labels) == len(filenames):
  print("Number of labels matches number of filenames!")
else:
  print("Number of labels does not match number of filenames, check data directories.")

# Find the unique label values
unique_breeds = np.unique(labels)
len(unique_breeds)

# Example: Turn one label into an array of booleans
print(labels[0])
labels[0] == unique_breeds # use comparison operator to create boolean array

# Turn every label into a boolean array
boolean_labels = [label == np.array(unique_breeds) for label in labels]
boolean_labels[:2]

len(boolean_labels)

# Example: Turning a boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds == labels[0])[0][0]) # index where label occurs
print(boolean_labels[0].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs

# List one more time to check
boolean_labels[:2]
filenames[:10]

```

##  Creating our own validation set
```xml 
Since the dataset from Kaggle doesn't come with a validation set 
(a split of the data we can test our model on before making final predicitons on the test set), 
let's make one.
We could use Scikit-Learn's train_test_split function or 
we could simply make manual splits of the data.

For accessibility later, let's save our filenames variable to X (data) and 
our labels to y.
# Setup X & y variables
X = filenames
y = boolean_labels

Since we're working with 10,000+ images, it's a good idea to work with a portion of them 
to make sure things are working before training on them all.

This is because computing with 10,000+ images could take a fairly long time. 
And our goal when working through machine learning projects is to reduce the time between experiments.

Let's start experimenting with 1000 and increase it as we need.

# Set number of images to use for experimenting
NUM_IMAGES = 1000 #@param {type:"slider", min:1000, max:10000, step:1000}
NUM_IMAGES

Now let's split our data into training and validation sets. 
We'll use and 80/20 split (80% training data, 20% validation data).

# Import train_test_split from Scikit-Learn
from sklearn.model_selection import train_test_split

# Split them into training and validation using NUM_IMAGES 
X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES], 
                                                  test_size=0.2,
                                                  random_state=42)

len(X_train), len(y_train), len(X_val), len(y_val)

# Check out the training data (image file paths and labels)
X_train[:5], y_train[:2]

```

## Preprocessing images (turning images into Tensors)
```xml


```

### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery
https://colab.research.google.com/
https://cloud.google.com/automl

```
