# Script to create a sqlite database with models, prompts and outputs. 
from sqlmodel import SQLModel, create_engine

sqlite_file_name = "summarisation_outputs.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)