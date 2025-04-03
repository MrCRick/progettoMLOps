import sys
import os

# Aggiungi la cartella principale del progetto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import load_model, predict_sentiment
from src.data import load_and_preprocess_data

# Test per verificare il caricamento del modello e le previsioni
def test_model():
    # Carica i dati (usiamo solo una parte per testare velocemente)
    texts, labels = load_and_preprocess_data()
    texts = texts[:10]  # Usa solo i primi 10 testi per velocità
    
    # Carica il modello
    tokenizer, model = load_model()
    
    # Predici il sentiment
    predicted_labels, probabilities = predict_sentiment(texts, tokenizer, model)
    
    # Mostra i risultati
    print(f"Predicted labels: {predicted_labels}")
    print(f"Probabilities: {probabilities}")
    
test_model()
