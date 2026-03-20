import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from sqlmodel import Session, select
from modules.database import engine
from modules.models import CaseStudyTexts

embedding_model_name = "bge-m3"

ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name=embedding_model_name,
)

# Creating the vectordatabase. This is already created here
def create_vector_db_and_collection():
    client = chromadb.PersistentClient()
    try:
        collection = client.get_or_create_collection(name="chroma_vector_db", 
                                                     embedding_function = ollama_ef, # pyright: ignore[reportArgumentType]
                                                     configuration={
                                                         "hnsw": {
                                                             "space": "cosine",
                                                             "ef_construction":150,
                                                             "ef_search":50
                                                         }}) 
        entry_counts = collection.count()
        print(f"collection created. Number of entries: {entry_counts}")
    except Exception as e:
        print(f"Error in creating database. Error message: {e}")

def create_id_textsembeddings_metadata():

    with Session(engine) as session:
        texts_data_list = session.exec(select(CaseStudyTexts).where(CaseStudyTexts.id == 4)) # this will be decoupled from sqlite in the future

        id_entry:list[str] = []
        input_text_embeddings:list[list[float]] = []
        token_count_per_input_text:list[dict] = []

        for text_object in texts_data_list:
            id = str(text_object.id)
            id_entry.append(id)
            
            response = ollama.embed(model=embedding_model_name, input=text_object.text)
            vector_embedding  = response["embeddings"][0]
            input_text_embeddings.append(vector_embedding)
            
            tokens = response["prompt_eval_count"]
            tokens_dict = {'total_tokens': tokens}
            token_count_per_input_text.append(tokens_dict)
        
        return id_entry, input_text_embeddings, token_count_per_input_text
   # needs a return statement here

# Add embedding along with other information for the vector database. 
def add_to_vector_db():
    id_entry, input_texts_embeddings, token_count_per_input_text = create_id_textsembeddings_metadata()
    client = chromadb.PersistentClient()
    collection = client.get_collection(name="experiment_chroma_vector_db",
                                       embedding_function = ollama_ef) # type: ignore
    try:
        collection.add(ids=id_entry,
                       embeddings = input_texts_embeddings, # type: ignore
                       metadatas= token_count_per_input_text # type: ignore
                       )
        get_the_added_entry = collection.get(ids = id_entry, include=["metadatas"])
        print(f"added the entry: {get_the_added_entry}")
    except Exception as e:
        print(f"Couldn't add to the database. error {e}")

#connect to collection and get data
def connect_to_vector_db_and_get_data():
    client = chromadb.PersistentClient()
    collection = client.get_collection(name="experiment_chroma_vector_db")
    total_entries = collection.count()
    metadata = collection.get(include=["metadatas"])
    print(f" total_entries: {total_entries}")
    print(f"their metadatas {metadata}")

# Semantic Ranking for a query

# LLM augmented prompt generation

def main():
    # create_id_textsembeddings_metadata()
    # add_to_vector_db()
    connect_to_vector_db_and_get_data()

if __name__ == "__main__":
    main()
