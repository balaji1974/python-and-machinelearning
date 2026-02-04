from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("What is the capital of {country}?")
prompt = template.format(country="India")
print(prompt)


from langchain.llms import OpenAI
llm = OpenAI(api_key="Your API Key")
response = llm(prompt)
print(response)