from sqlmodel import Session, select
import json
from ollama import generate
from modules.database import engine, create_db_and_tables
from modules.models import Models, Outputs, CaseStudyTexts
import re

#Json_load
with open("data/llm_models.json", mode="r", encoding="utf-8") as read_file:
        llm_models_list = json.load(read_file)
with open("data/case_Study_texts.json", mode="r", encoding="utf-8") as read_file:
        texts_data_list = json.load(read_file)

#text summarisation user query
text_summarisation_query = """
Summarise the following case study from one of the Catapults based in the UK in 150 words. 

- No need to include the title.
- Include Catapult innovation support, the outcome and impact of the project if present. 
- Include UK government departments if any the case study.
- Include company/business names if mentioned.
- Include region, international information if mentioned.

"""

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

def create_casestudy_texts():

    #creating session and adding data
    with Session(engine) as session:
        for text in texts_data_list:
            existing = session.exec(select(CaseStudyTexts).where(CaseStudyTexts.text == text["text"])).one_or_none()
            if not existing:
                entry_text = CaseStudyTexts(text=text["text"])
                session.add(entry_text)
        session.commit() 

#generate summaries using models and prompts in Ollama and add them to the table
def create_outputs(user_query:str = text_summarisation_query):
    with Session(engine) as session:
        models_table_data = session.exec(select(Models)).all()
        texts_data_list = session.exec(select(CaseStudyTexts)).all()

        # add outputs data
        for t in texts_data_list:
            for llm_model in models_table_data: 
                existing = session.exec(select(Outputs).where(Outputs.model_id == llm_model.id).where(Outputs.text_id == t.id)).one_or_none()
                if not existing:
                    response = generate(llm_model.model,user_query + str(t.text))
                    text_summarisation_entry = Outputs(output=str(response.response),
                                                    models=llm_model,
                                                    texts=t)
                    session.add(text_summarisation_entry)
                    print(f"Done: {llm_model.model}")
        session.commit()

#Reading model table
def select_model():
     with Session(engine) as session:
          models = session.exec(select(Models)).all()
          model_table = []
          for m in models:
               model_table.append(m)
     return model_table

#Reading texts table
def select_texts():
    with Session(engine) as session:
        texts_first_entry = session.exec(select(CaseStudyTexts)).first()
        print(text_summarisation_query + str(texts_first_entry.text)) # type: ignore
    

#Read outputs and write to a .txt file
def select_outputs():
     with Session(engine) as session:
        outputs = session.exec(select(Outputs)).all()
        def add_newlines_after_period(text):
            # Replace '. ' with '.\n'
            return re.sub(r'\. ', '.\n', text)
        
        for o in outputs:
            with open("data/outputs.txt",'a') as file:
                first_line = f"Reponse from {o.models.model} for casestudy_{o.text_id}" # pyright: ignore[reportOptionalMemberAccess]
                output_edited = add_newlines_after_period(str(o.output))
                file.write(first_line + '\n' + output_edited + '\n\n')
        print("outputs appended")

#updating the outputs table with only the response object and nothing else
def update_outputs(user_query:str = text_summarisation_query):
     with Session(engine) as session:
        models_table_data = session.exec(select(Models)).all()
        case_study_texts = session.exec(select(CaseStudyTexts)).all()
        
        for t in case_study_texts:
            for llm_model in models_table_data: 
                response = generate(llm_model.model, user_query + str(t.text))
                outputs_table = session.exec(select(Outputs).where(Outputs.model_id == llm_model.id).where(Outputs.text_id == t.id)).one()
                outputs_table.output = str(response.response) 
                session.add(outputs_table)
                print(f"Done: {llm_model.model}")
        session.commit()

                    
def main():
    create_db_and_tables()
    #create_casestudy_texts()
    #create_outputs()
    update_outputs()
    #select_outputs()
    #select_texts()

if __name__ == "__main__":
    main()