
# Neural Networks

## Introduction

### Neural Network
A neural network is a machine learning model inspired by the human brain, featuring 
interconnected nodes (neurons) arranged in layers to process data and recognize complex 
patterns. By adjusting weights between neurons during training, they map inputs to outputs 
for tasks like image recognition, NLP, and predictive modeling.


### What is TensorFlow
TensorFlow is a software library for machine learning and artificial intelligence. 
It can be used across a range of tasks, but is used mainly for training and 
inference of neural networks. It is one of the most popular deep learning frameworks, 
alongside others such as PyTorch and PaddlePaddle


### TensorFlow Hub (TF Hub)
TensorFlow Hub is an "App Store" or package manager specifically for pre-trained 
machine learning model components (called modules or SavedModels). 
Its primary goal is to promote transfer learning and model reuse. 
Instead of training a model from scratch (which can require massive computational resources 
and data), developers can use tensorflow_hub library functions like hub.KerasLayer or hub.load() 
to download and incorporate robust, pre-trained models (e.g., text embeddings, 
image classification models like Inception) into their own projects with minimal code.


#### Why do we use TensorFlow
We use TensorFlow primarily for building and deploying machine learning (ML) 
and deep learning models. Here’s why it's widely used:

1. **Scalability & Performance**
Efficient computation: Uses optimized operations with 
support for GPUs (Graphical Processing Units), TPUs (Tensor Processing Units), 
and distributed computing.
Scalable deployment: Works from mobile devices (TensorFlow Lite) 
to large-scale distributed systems (TensorFlow Serving).

2. **Flexibility & Ease of Use**
Multi-language support: Primarily Python, 
but also supports C++, Java, and Go.
Multiple APIs: High-level APIs like Keras for quick prototyping 
and low-level APIs for custom model development.

3. **Strong Ecosystem & Tooling**
TensorFlow Extended (TFX): A full ML pipeline framework for 
production-ready applications.
TensorBoard: Visualization tools for debugging and optimizing models.
TF Lite & TF.js: Deploy models on mobile, edge devices, or in-browser.

4. **Industry Adoption & Community Support**
Developed by Google and widely adopted across industries, 
making it well-documented with strong community support.

5. **Support for Advanced ML & DL Features**
Pre-trained models & transfer learning.
Reinforcement learning, generative models, and graph-based computations.
Auto-differentiation and gradient-based optimization.


### What is deep learning:
Deep learning is a type of machine learning that uses artificial neural networks 
to learn from data. Artificial neural networks are inspired by the human brain, 
and they can be used to solve a wide variety of problems, 
including image recognition, natural language processing, and speech recognition.

#### What kind of object detection problems are there:
1. **Classification**
Predicting a category or class label from given input data.
Example Use Cases:
Image classification (e.g., detecting cats vs. dogs in images).
Spam detection (classifying emails as spam or not).
Sentiment analysis (classifying text as positive, negative, or neutral).
Common Models: CNNs for images, RNNs/Transformers for text, and MLPs for tabular data.

2. **Sequence-to-Sequence (Seq2Seq)** 
Mapping an input sequence to an output sequence of different or the same length.
Example Use Cases:
Machine translation (e.g., English to French translation).
Speech-to-text (converting audio into transcribed text).
Chatbots and conversational AI (generating responses based on input text).
Common Models: LSTMs, GRUs, Transformer-based models (like T5, GPT, BERT).

3. **Object Detection** 
Identifying and localizing multiple objects within an image or video frame.
Example Use Cases:
Autonomous driving (detecting pedestrians, vehicles, and traffic signs).
Security surveillance (detecting suspicious activities or intruders).
Medical imaging (detecting tumors in X-ray/MRI scans).
Common Models: Faster R-CNN, YOLO, SSD, EfficientDet


### What is Transfer Learning
Transfer learning is a deep learning technique where a pre-trained model 
(trained on a large dataset) is fine-tuned or adapted for a different but 
related task. Instead of training a model from scratch, 
we reuse learned features from a previously trained network.

#### Why Use Transfer Learning

**Faster Training**  – Pre-trained models already have learned general features, 
reducing the need for extensive training.
**Less Data Requirement** – Since the model has prior knowledge, 
we can achieve good results even with a small dataset.
** Improved Performance ** – Leveraging knowledge from large datasets helps in 
achieving better accuracy, especially in domains with limited data.
** Efficient Use of Resources ** – Saves computational power and time compared 
to training deep models from scratch.

**Example Use Cases**
Image classification using pre-trained CNNs (e.g., ResNet, VGG, MobileNet).
NLP tasks like sentiment analysis using transformer models (e.g., BERT, GPT).
Medical imaging, where pre-trained models on general images are fine-tuned for 
specific diseases.


## Data to work on and environment setup on Google Colab  
Go to the below website (after registering in it)
https://www.kaggle.com/c/dog-breed-identification/overview

Go to Data (from the below link) -> Download all the data 
https://www.kaggle.com/c/dog-breed-identification/data
(dog-breed-identification.zip)

With a google account go to below link:
https://colab.research.google.com/

And start a new notebook
>This is quite similar to Juypter notebook 
>This environment is connected to Google Compute Engine at the backend

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


## Intro to TensorFlow on Google Colab 
```xml 
Follow the below files for a quick into to Tensorflow:
01-tensorflow_test.ipynb
```


## Working with TensorFlow on Google Colab 
```xml 
Follow the below files which are self explanatory:
02-dog-vision.ipynb

```

## Interesting vidoes to watch

### How Machines Learn by GCP Grey
https://www.youtube.com/watch?v=R9OHn5ZF4Uo

### Deep Learning series by 3Blue1Brown
https://www.youtube.com/watch?v=aircAruvnKk


## Important Resources

Build TensorFlow input pipelines:  
https://www.tensorflow.org/guide/data


Load and preprocess images:  
https://www.tensorflow.org/tutorials/load_data/images



### Reference
```xml
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery
https://colab.research.google.com/
https://cloud.google.com/automl

```
