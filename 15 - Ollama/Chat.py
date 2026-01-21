from ollama import chat

resp = chat(model='llama3.1:8b', messages=[
{'role': 'user', 'content': 'Give me 3 beginner Python tips.'}])
print(resp['message']['content'])