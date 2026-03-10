from sqlmodel import Session, select
import json
from ollama import generate
from modules.database import engine, create_db_and_tables
from modules.models import Models, Prompts, Outputs

#Json_load's
with open("data/llm_models.json", mode="r", encoding="utf-8") as read_file:
        llm_models_dict = json.load(read_file)
with open("data/text_and_prompts.json", mode="r", encoding="utf-8") as read_file:
        prompts_data_dict = json.load(read_file)

#add data to the tables
def create_model_and_prompts():

    #Creating session and adding data
    with Session(engine) as session:
        #add llm_model data
        for llm_model_name in llm_models_dict.values():
            llm_entry = Models(model=llm_model_name)
            session.add(llm_entry)
        session.commit()

        #add prompt_model data
        for prompt_text in prompts_data_dict.values():
            entry_prompt_text = Prompts(prompt=prompt_text)
            session.add(entry_prompt_text)
            #output_entry_prompt_id = Outputs(prompt_id=entry.id)
            #session.add(output_entry_prompt_id)
        session.commit()

#generate summaries and add them to the table
def create_outputs():
    with Session(engine) as session:
        
        #creating lookup table's to link the tables
        models_table_data = session.exec(select(Models)).all()
        #lookup table with {model_name:model_object} for using relationship attribute
        models_table_data_lookup = {m.model : m for m in models_table_data}

        prompts_table_data = session.exec(select(Prompts)).all()
        #lookup table with {prompt: prompt_object}
        prompts_table_data_lookup = {p.prompt : p for p in prompts_table_data}
        
        # add outputs data
        for prompt in prompts_data_dict.values():
            for llm_model in llm_models_dict.values(): 
                response = generate(llm_model,prompt)
                text_summarisation_entry = Outputs(output=str(response),
                                                   models=models_table_data_lookup[llm_model],
                                                   prompts=prompts_table_data_lookup[prompt])
                session.add(text_summarisation_entry)
        session.commit()

def select_model():
     with Session(engine) as session:
          models = session.exec(select(Models)).all()
          model_names = []
          for m in models:
               model_names.append(m.model)
     return model_names  
               
                    
def main():
    # create_db_and_tables()
    # create_model_and_prompts()
    # create_outputs()
    model_names = select_model()
    #now i can open a file and write to it. Or even to external applications.

if __name__ == "__main__":
    main()