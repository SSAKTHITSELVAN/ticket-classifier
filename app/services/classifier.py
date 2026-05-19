import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from app.services.preprocessor import preprocess_text

MODEL_PATH = os.path.join(PROJECT_ROOT, 'app', 'model', 'ticket_classifier_model.pkl')
MODEL = joblib.load(MODEL_PATH)

class Classifier:
    def __init__(self):
        self.model = MODEL

    def classify(self, data):
        data = preprocess_text(data)
        return [self.model.predict([data]), self.model.predict_proba([data])]

if __name__ == "__main__":
    classifier = Classifier()
    sample_text = "I need help with issue."
    category = classifier.classify(sample_text)
    print(f"Predicted category: {category}")