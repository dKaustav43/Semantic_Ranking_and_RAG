I had a bit of reading on which kind of language models to use and what would be my LLM specific tools to work with text summarization tasks with the available harware. 

My task is zero shot text summarization task. 
I have a few things I need to look into 

1. LLM kind and size - Ollama 3.1 8b parameter is set to be good with my mac m4 and HP z book. 
Qwen - 14b is quite large with my hardware. Need to look into these models a bit closer. 
FLANT5 Large is better with fine tuning and not as much with zero shot. These are encoder-decoder type models with less context window than the above two.


2. Ollama with Python is said to be easier to run than with Hugging face.
The use of 4-bit quantisation makes it more memory efficient, which comes inherently with Ollama. 

Look at the real python article to learn more about Ollama and Python 
https://realpython.com/ollama-python/

# Pulling the Ollama model 

```bash
ollama pull llama3.2:latest
ollama pull codellama:latest
```

# Directly work from the shell 

```bash
ollama run llama3.2:latest 
>>> Explain what python is in one sentence
```

# Install Ollama Python SDK 
```bash
(venv) python -m pip install Ollama
```

# Generate text and code from Python

ollama.chat() - role based, multi-turn conversation.
ollama.generate() - for one shot prompts. Suitable for drafting, rewriting, summarizing and code generation. 

```python
from ollama import chat 

messages = [
    {
    "role" : "user",
    "content" : "Explain what Python is in one sentence",
    },
]

response = chat(model="llama3.2:latest", messages=messages)
print(response.message.content)

# to keep the context 
messages.apend(response.message)
messages.append(
    {
        "role" : "user",
        "content" : "Provide a simple code in Python"
    }
)
response = chat(model="llama3.2:latest", messages = messages)
print(response.message.content)
```
# Text generation interface

```python
from ollama import generate 

prompt = """
"""

response = generate(model, prompt)
print(response.response)
```

I downloaded two models from Ollama. One is Granite and the other is Llama. I feel the summarization done by Granite was much better that that done by Llama. 
Granite also promises to use less RAM and be faster as it uses only 1B active weights rather than the whole of 7B as opposed to 8B by Llama.

Feature - Integrating Chroma into applications

1. Issue is that the prompt table has prompts which ask it to summarise the data. These are not prompts too, they have case study entries plus user query tailored to task summarisation. 
    One of the fix, is to delete the table Prompts. Rename it to case study text entry or something and adjust the main.py code base. 
    
2. Add chroma with augmented prompt.

3. No need to store user query or augmented prompt in sqlite database as these need to be updated soon and I don't need a log the user prompts in a new table as of now. I only need to be able to see the outputs and how they compare in this example project. I might need to store this in my other projects. 

Feature - Sqlite FTS5 search
1. Implementing this would be interesting ofcourse but I am not sure how useful it will be for this or my Catapult casestudy project.
Firstly, I will have 500 case studies, all of which could in principle be fed to the LLM directly as my model has 128K token window. 
Secondly, I have about 18-19 identifiers to sort the relevant case study for more specific queries. That itself is a huge bost before applying my search. 
Thirdly, keyword search would not be as accurate in my opinion as it will be heavily baised towards the keyword I choose to input for a 
particular query. 
2. So implement this on an experiment basis only for trial and to get familiar. Use this in my real projects only if needed. 
