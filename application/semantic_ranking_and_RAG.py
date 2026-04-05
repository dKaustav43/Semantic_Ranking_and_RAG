import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from sqlmodel import Session, select
from modules.database import engine
from modules.models import CaseStudyTexts
import ollama 
from typing import Any
import time

embedding_model_name = "bge-m3"

ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name=embedding_model_name,
)

RAG_llm_model = 'llama3.1:8b'

def semantic_ranking(user_query:list[str],number_of_results:int):

    client = chromadb.PersistentClient()
    collection = client.get_collection(name="experiment_chroma_vector_db", embedding_function = ollama_ef) # type: ignore
    result = collection.query(query_texts=user_query,
                              n_results=number_of_results,
                              include=["metadatas","distances"])
    
    # Printing results to CLI
    for i,query_text in enumerate(user_query):
        print(f"Your query: {query_text}\nSemantic ranking: \n")

        for ids, distances, metadatas in zip(result["ids"][i], result["distances"][i], result["metadatas"][i]): # type: ignore
            print(f"id:{ids} | distances:{distances} | metadatas:{metadatas}"+'\n')

def LLM_augment_prompt_generation(user_query_for_augment_prompt:str, LLM_model:str = RAG_llm_model):
    #since I am using only four case studies I am putting in all as a part of my augement prompt.
    #For larger database I will be inputing top_k results from a Hybrid search retrieval or from the Chromadb retrival above.
    
    #Read the texts from sqlite database
    with Session(engine) as session:
        texts_data_list = session.exec(select(CaseStudyTexts)).all() # this will be decoupled from sqlite in the future

    augment_prompt = f"""
    Answer using the case studies ONLY from below. 
    Cite which case study supports each point and include the case study id. 
    If there's no suitable case study then just say so without any further explanation. 

    Here are the case study texts against there corresponding id's - 

    [Case studies] case_id = {texts_data_list[0].id} : text = {texts_data_list[0].text} 
    \n\n + case_id = {texts_data_list[1].id} : text = {texts_data_list[1].text} + 
    \n\n + case_id =  {texts_data_list[2].id} : text = {texts_data_list[2].text}
    \n\n + case_id_3 {texts_data_list[3].id} : text = {texts_data_list[3].text}

    Prompt:{user_query_for_augment_prompt}

    """
    response = ollama.chat(
        model= LLM_model,
        messages=[{'role': 'user', 'content': augment_prompt}],
        options = {
            'num_ctx': 12288,
            'num_predict': 256,
            'num_gpu' : 99,
            'num_thread': 8
        }
        )
    result = response['message']['content']
    print(f"\n query : {user_query_for_augment_prompt} \n\n Answer: {result}")

def main():
    #   query = ["Is there a case study where the Catapult has worked with the NHS?", "Circular Economy"]
    #   semantic_ranking(user_query=[query[1]],number_of_results=4)
    start_time = time.perf_counter()
    rag_query = "Is there case study where atleast one UK government department is named other than UKRI?"
    LLM_augment_prompt_generation(rag_query)
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n--- Response received in {duration:.2f} seconds ---")


if __name__ == "__main__":
    main()
