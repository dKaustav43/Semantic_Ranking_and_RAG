from sqlmodel import Session, select
import json
from ollama import generate
from modules.database import engine, create_db_and_tables
from modules.models import Models, Prompts, Outputs

#Json_load
with open("data/llm_models.json", mode="r", encoding="utf-8") as read_file:
        llm_models_list = json.load(read_file)
with open("data/text_and_prompts.json", mode="r", encoding="utf-8") as read_file:
        prompts_data_list = json.load(read_file)

#add data to the tables
def create_model_and_prompts():

    #Creating session and adding data
    with Session(engine) as session:
        for llm_model in llm_models_list:   
            existing = session.exec(select(Models).where(Models.model == llm_model["model"])).one_or_none()
            if not existing:
                new_model_entry = Models(model=llm_model["model"],use=llm_model["use"])
                session.add(new_model_entry)
        session.commit()

        #add prompt_model data
        for prompt_text in prompts_data_list:
            existing = session.exec(select(Prompts).where(Prompts.prompt == prompt_text["prompt"])).one_or_none()
            if not existing:
                entry_prompt_text = Prompts(prompt=prompt_text["prompt"])
                session.add(entry_prompt_text)
        session.commit()

#generate summaries using models and prompts in Ollama and add them to the table
def create_outputs():
    with Session(engine) as session:
        models_table_data = session.exec(select(Models)).all()
        prompts_table_data = session.exec(select(Prompts)).all()

        # add outputs data
        for prompt in prompts_table_data:
            for llm_model in models_table_data: 
                response = generate(llm_model.model,prompt.prompt)
                text_summarisation_entry = Outputs(output=str(response),
                                                   models=llm_model,
                                                   prompts=prompt)
                session.add(text_summarisation_entry)
                print(f"Done: {llm_model.model}")
        session.commit()

def select_model():
     with Session(engine) as session:
          models = session.exec(select(Models)).all()
          model_table = []
          for m in models:
               model_table.append(m)
     return model_table
               
                    
def main():
    create_model_and_prompts()
    create_outputs()

if __name__ == "__main__":
    main()