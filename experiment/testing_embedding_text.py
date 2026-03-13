import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

text_1 = """
Accelerating AI innovation in the West of England creative industries

Artificial intelligence is transforming the UK’s creative economy, providing a powerful boost to productivity, creativity and market reach. AI is being used to unlock new production methods, enhance workflows, and to create new ways to personalise content.

The UK creative sector contributes £124 billion to the national economy and employs 2.4 million people. Integrating AI with creative skills is a cornerstone of the UK’s Modern Industrial Strategy, which marks a new era of collaboration between government and high growth industries. The comprehensive Sector Plan for Creative Industries sets out the ambition for the UK to be recognised as the best place in the world to make and invest in film and TV, video games, music, performing and visual arts, and advertising and marketing. 
Challenge

Led by the University of Bristol and backed by UKRI, MyWorld explores the future of creative technology innovation by pioneering new ideas, products and processes. The programme focuses on supporting the creative industries in the West of England region by providing access to world-class facilities, funding, research, tools, and training. It brings together prominent partners from academia and industry, and invests in creative technology innovation to stimulate growth, drive societal change, and strengthen the region’s position as a centre for cutting-edge development.

The MyWorld programme is a consortium of 13 partners – including universities and relevant creative organisations from the region (including Aardman and the BBC), and Digital Catapult.

As part of MyWorld, Digital Catapult delivered four open calls – two challenge led accelerators that enabled early stage prototype development in response to industry needs, and two collaborative research and development programmes, that enabled teams made up of SMEs and an academic counterpart to experiment with new applications of advanced technology in the creative industries. While only one of the four calls was designed to specifically address AI, its use played a part in all four calls, allowing the programme to build on the previous results and ramp up the impact.
"""

text_2 = """
Intelligent handling system design and integration
MTC developed an AI driven sorting system for separating intermediate and low-level nuclear decommissioning waste.

Challenge

This project was funded by the SBRI (Small Business Research Initiative) to develop intelligent automated solutions for the sorting and segregation of nuclear waste. Currently waste from the decommissioning of nuclear facilities has to be sorted manually with the hazards associated and inefficient sorting can result in costly storage solutions with inefficiently packaged waste.

The aim was to develop a prototype system for autonomously sorting radioactive waste with a robot handling system using AI (Artificial Intelligence) vision systems to identify waste objects, and a measurement and sorting system to stream the objects to a set of output waste containers. This will enable the sorting of waste more safely with no/minimal human intervention and create a more efficient packaging process, reducing waste storage capacity requirements.

Solution

MTC developed an AI based multi-camera computer vision software package with a custom user interface.
Grasp planning algorithms for vacuum and parallel gripper types were developed and integrated with the robot to pick the detected waste items.
Object measurement was also performed through the vision system, generating important information for the waste sorting decision model.
A bin packing algorithm was developed for optimising use of output bin space, reducing the number of container swaps required.
Outcome

An intelligent system that identifies individual waste items using its vision system. Multiple views are fused into a 3D model for input to the grasping algorithms.
Waste objects are picked up with an appropriate gripper for radiological and chemical analysis which drives output stream sorting.
The vision system provides a traceable data record with every sorted object.
The efficient packing algorithm maximises use of the output containers for safe processing or storage.
Impact

Reduction or elimination of human manual sorting tasks which must currently be conducted in a high risk, dangerous and dirty environment.
Trainable AI system will enhance performance though machine learning during use.
Each object receives traceable record including images, measurement, radiological and chemical data, building a digital data set for life.
The developed solution will allow improved sorting and segregation of nuclear waste ensuring waste streams are appropriately classified and managed in a safe manner, and reduce waste storage costs.
"""
text_3 = """

Rethinking the bottle – sustainable design for toddlers
CPI and SLN Sustainability redesigned a toddler water bottle using circular, data-driven design to balance safety, functionality and sustainability from the start.

Challenge

The traditional design process for consumer goods is often inefficient in its material use, prioritising form, features and market appeal, often at the expense of environmental impact. This approach has led to widespread overengineering and material inefficiency. 

Products often exceed functional requirements, which can cause unnecessary consumption, and create complex recycling challenges and increase waste. 

The Design for Sustainability and Circularity (DfSC) framework, which is being developed by the High Value Manufacturing Catapult challenges manufacturers to position environmental impact at the heart of the early stages of product development, rather than as a constraint or an afterthought. 

Children’s water bottles, for example, often exceed functional requirements, generating unnecessary waste and rarely considering end-of-life (EOL) impacts. High safety standards in childcare products add further complexity, making the integration of sustainability difficult. The sector lacks a consistent approach to embedding circularity and sustainable design from the start. 

Solution

CPI, in collaboration with SLN Sustainability Ltd, applied the Design for Sustainability and Circularity (DfSC) framework  to redesign a toddler’s water bottle. The aim was to create a product that is safe, functional and inherently sustainable. 

The team took a data-driven approach to material selection, considering drinking water safety, impact resistance, manufacturability and environmental performance. Life Cycle Assessment (LCA) was integrated early to evaluate environmental impacts against planetary boundaries. Using an adapted double diamond design model, three material concepts were explored: bio-based polymer, metal and recycled plastic. To improve circularity, the final concept minimised material variety, using just two components – a rigid polymer and an elastomer – reducing production waste and simplifying EOL processes. 

Outcome

The iterative design process resulted in a proof-of-concept bottle that balanced safety, functionality and sustainability. LCA comparisons between bio-based polymer and recycled plastic options showed that the recycled plastic concept had a lower overall environmental impact, challenging common assumptions about “eco-friendly” materials. The evidence-based approach highlighted trade-offs and key hotspots, giving stakeholders actionable insights for decision-making. 

Impact

This project demonstrates how sustainability can be embedded from the outset. 

By aligning materials, manufacturing processes and EOL pathways with planetary boundaries, CPI and SLN Sustainability Ltd delivered a solution grounded in science. It demonstrates that sustainability does not need to be a trade-off, but rather a powerful enabler of function, form and future resilience. 

This approach offers a blueprint for how designers and engineers can both reduce harm and reimagine how products are conceived.

Funded by Innovate UK Business Growth and HVM Catapult. 
"""
id_entry:list[int] = []
input_texts:list[str] = [text_1,text_2,text_3]
input_texts_embeddings:list[float] = []
token_count_per_input_text:list[int] = []

embedding_model_name = "bge-m3"

def create_id_texts_embeddings_tokenlength_lists():
    for index, text in enumerate(input_texts):
        id_entry.append(index+1)
        reponse = ollama.embed(model=embedding_model_name, input=text)
        vector  = reponse["embeddings"][0]
        input_texts_embeddings.append(vector)
        tokens = reponse["prompt_eval_count"]
        token_count_per_input_text.append(tokens)
    return id_entry, input_texts_embeddings, token_count_per_input_text

ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name=embedding_model_name,
)

def create_vector_db_and_collection():
    client = chromadb.PersistentClient()
    try:
        collection = client.get_or_create_collection(name="experiment_chroma_vector_db", 
                                                     embedding_function = ollama_ef, # pyright: ignore[reportArgumentType]
                                                     configuration={
                                                         "hnsw": {
                                                             "space": "cosine",
                                                             "ef_construction":50,
                                                             "ef_search":50
                                                         }}) 
        entry_counts = collection.count()
        print(f"collection created. Number of entries: {entry_counts}")
    except Exception as e:
        print(f"Error in creating database. Error message: {e}")

#add to the vector database
def add_to_vector_db():
    id_entry, input_texts_embeddings, token_count_per_input_text = create_id_texts_embeddings_tokenlength_lists()
    id_entry_str = [str(num) for num in id_entry]
    metadata:list[dict] = [{"Title":"Accelerating AI innovation in the West of England creative industries", "Catapult":"Digital Catapult", "token_count_per":token_count_per_input_text[0]},
                            {"Title": "Intelligent handling system design and integration", "Catapult":"HVMC", "token_count_per":token_count_per_input_text[1]},
                            {"Title":"Rethinking the bottle – sustainable design for toddlers", "Catapult":"HVMC", "token_count_per":token_count_per_input_text[2]}]
    
    client = chromadb.PersistentClient()
    collection = client.get_collection(name="experiment_chroma_vector_db", embedding_function = ollama_ef) # type: ignore
    try:
        collection.add(ids=id_entry_str,
                   embeddings=input_texts_embeddings,
                    metadatas=metadata) # pyright: ignore[reportArgumentType]
        entry_counts = collection.count()
        print(entry_counts)
        get_first_entry = collection.get(ids = ["1"], include=["metadatas"])
        print(get_first_entry)
    except Exception as e:
        print(f"couldn't add to the database. error {e}")

def query_collection(query:list[str]):
    client = chromadb.PersistentClient()
    collection = client.get_collection(name="experiment_chroma_vector_db", embedding_function = ollama_ef) # type: ignore
    result = collection.query(query_texts=query,
                              n_results=3,
                              include=["metadatas","distances"])
   
    with open("experiment/semantic_outputs_chroma.txt","a") as file:
        for i, query_text in enumerate(query):
            print(query_text)
            file.write('\n' + query_text+ '\n\n')
            for ids, distances, metadatas in zip(result["ids"][i], result["distances"][i], result["metadatas"][i]): # type: ignore
                    print(ids,distances)
                    file.write(f"id:{ids} | distances:{distances} | metadatas:{metadatas}"+'\n')
    print("results writen to file experiment/semantic_outputs_chroma.txt")               


def main():
    query:str = "Any case study which has UK government department"
    query_collection([query])

if __name__ == "__main__":
    main()




