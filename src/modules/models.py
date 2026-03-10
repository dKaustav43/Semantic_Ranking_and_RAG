from sqlmodel import Field, SQLModel, Relationship

#Define the tables
class Models(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model: str
    use: str | None = Field(default=None)

    outputs: list["Outputs"] = Relationship(back_populates="models")

class Prompts(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    prompt: str

    outputs: list["Outputs"] = Relationship(back_populates="prompts")

class Outputs(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    output: str | None = Field(default=None)
    model_id: int | None = Field(default=None, foreign_key="models.id")
    prompt_id: int | None = Field(default=None, foreign_key="prompts.id")

    models: Models | None = Relationship(back_populates="outputs")
    prompts: Prompts | None = Relationship(back_populates="outputs")