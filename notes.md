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

