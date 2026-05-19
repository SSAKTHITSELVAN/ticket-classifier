import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.preprocessor import preprocess_text
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_model(df):
    df['embedding'] = df['text'].apply(preprocess_text)
    df['category'] = df['category'].str.lower()
    MODEL_DIR = os.path.join('app', 'model')
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(df['embedding'], df['category'], test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(list(df['embedding']), df['category'])
    y_pred = model.predict(list(df['embedding']))
    print("Model trained successfully!")
    print("Classification Report:")
    print(classification_report(df['category'], y_pred))
    print("Accuracy:", accuracy_score(df['category'], y_pred))
    joblib.dump(model, os.path.join(MODEL_DIR, 'ticket_classifier_model.pkl'))

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/sample_tickets.csv')
    train_model(df)