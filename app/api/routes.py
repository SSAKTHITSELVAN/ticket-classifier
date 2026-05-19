from fastapi import FastAPI
from pydantic import BaseModel
from app.services.classifier import Classifier
app = FastAPI()

