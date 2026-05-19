from fastapi import FastAPI
from app.services.classifier import Classifier
from app.model.schemas import Ticket, ClassificationResult

app = FastAPI()
classifier = Classifier()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Ticket Classification API!"}

@app.post("/classify", response_model=ClassificationResult)
def classify_ticket(request: Ticket):
    category, probabilities = classifier.classify(request.description)
    return ClassificationResult(category=category[0], score=float(probabilities[0].max()))