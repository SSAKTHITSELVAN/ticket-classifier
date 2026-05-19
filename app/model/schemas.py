from pydantic import BaseModel, Field

class Ticket(BaseModel):
    description: str = Field(..., example="The application crashes when I try to upload a file.")

class ClassificationResult(BaseModel):
    category: str = Field(..., example="bug")
    score: float = Field(..., example=0.95)