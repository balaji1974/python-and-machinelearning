
# Hugging Face

## Introduction & Steps
```xml 
HuggingFace is a large open-source community that builds tools to enable users 
to build, train, and deploy machine learning models based on open-source code 
and technologies.

Step 1:
------
Create an account in hugging face
https://huggingface.co/join

and login using 
https://huggingface.co/login

Step 2: (Prerequisite)
-------
Install the dependencies:

# Transformers acts as the model-definition framework for state-of-the-art machine learning 
models in text, computer vision, audio, video, and multimodal model, for both inference and training.
Transformer -> pip install transformers

# LangChain is an open-source framework that simplifies building applications powered by 
large language models (LLMs). It provides tools and components to connect LLMs with various data 
sources and computation, enabling developers to create sophisticated AI applications like chatbots 
and question-answering systems
Langchain -> pip install langchain

# This package contains the LangChain integrations for huggingface related classes.
Langchain HuggingFace -> pip install langchain-huggingface


Step 3:
-------
Create Token -> From your Profile -> Access Token -> Create New Token 
Go to Read Tab -> <Enter Token Name> -> Create Token 
Copy the token 

Step 4: 
-------
Adding the Token 
In command line run the following command: 
huggingface-cli login 
It will ask for the token, paste it. 
Add to git central (y/N): y

This will add the token successfully and you will see:
The current active token is: 'your token name'

Step 5: 
-------
To continue with installing CUDA and activate GPU (later)
pip install torch torchvision torchaudio



```


### Reference
```xml
https://huggingface.co
https://www.youtube.com/watch?v=1h6lfzJ0wZw
https://github.com/techwithtim/Langchain-Transformers-Python/tree/main

```
