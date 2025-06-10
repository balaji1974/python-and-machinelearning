# This script demonstrates a text summarization and question-answering pipeline using LangChain and Hugging Face Transformers.
from transformers import pipeline

# Import the necessary pipeline from Hugging Face Transformers
model = pipeline("summarization", model="facebook/bart-large-cnn", max_length=10)
# Use the model to summarize a given text
response = model("LangChain is a framework for developing applications powered by language models. It provides modular components that can be combined to create complex workflows, such as chatbots, question-answering systems, and more. LangChain supports various language models and allows developers to easily integrate them into their applications.")
# Print the response from the summarization model
print(response)
