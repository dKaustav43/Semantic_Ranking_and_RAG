from sqlmodel import Field, SQLModel, Relationship

#Define the tables
class Models(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model: str
    use: str | None = Field(default=None)

    outputs: list["Outputs"] = Relationship(back_populates="models")

class CaseStudyTexts(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str

    outputs: list["Outputs"] = Relationship(back_populates="texts")

class Outputs(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    output: str | None = Field(default=None)
    model_id: int | None = Field(default=None, foreign_key="models.id")
    text_id: int |  None = Field(default=None, foreign_key="casestudytexts.id")

    models: Models | None = Relationship(back_populates="outputs")
    texts: CaseStudyTexts | None = Relationship(back_populates="outputs")