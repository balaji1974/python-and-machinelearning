from ollama import embeddings

text = "Ollama lets you run LLMs locally."
vec = embeddings(model='embeddinggemma', prompt=text)
print(vec['embedding']) # dimension