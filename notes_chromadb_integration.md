Model:
```bash
ollama pull bge-m3
```
I have used this as it comes recommended by several online resources. 
MRR is higest amongst many open source models. 
Provides dense, sparse and multivector retrieval.

max tokens: 8192 tokens (large context length)
parameters: 567M 
embedding dimension: 1024
architecture: BGE-M3 is XLM-RoBERTa-large(some sort of BPE), multilingual encoder with 24 attention layers and 1024 hidden dimensions.
pre-training: pre-trained on 184 million text samples covering 105 languages. 
fine-tuning - dense+sparse+CoLbert retrival

syntax

```python
import ollama 

response = ollama.embed(model="bge-m3", input = "Your text to embed here")
embeddings = reponse["embeddings"] #shape(1,1024) for bge-m3.
vector        = response["embeddings"][0]          # first (only) vector
dims          = len(vector)                        # 1024 for bge-m3
total_ms      = response["total_duration"] / 1e6  # ns → ms
tokens        = response["prompt_eval_count"]
```
additional syntax from claude: need to check the docs
For sparse vectors to enable BM25 etc keyword search.

pip install FlagEmbedding

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    ["Your text here"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False
)

dense_vector   = output["dense_vecs"][0]        # shape: (1024,)
sparse_weights = output["lexical_weights"][0]   # dict: {token_id: weight}
```
