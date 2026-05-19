import pandas
import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
st = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text: str):
    text = text.lower()
    text = " ".join([token.lemma_ for token in nlp(text) if not token.is_stop and not token.is_punct and token.is_alpha])
    text = st.encode(text)
    return text