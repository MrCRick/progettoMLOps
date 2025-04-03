import sys
import os

# Aggiungi la cartella principale del progetto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import plot_confusion_matrix, print_classification_report
from sklearn.metrics import accuracy_score
from src.model import predict_sentiment, load_model
from src.data import load_and_preprocess_data

# Test per visualizzare la matrice di confusione e il classification report
def test_utils():
    # Carica i dati
    texts, labels = load_and_preprocess_data()
    texts = texts[:100]  # Usa solo i primi 100 testi per velocità
    labels = labels[:100]
    
    # Predici il sentiment
    tokenizer, model = load_model()
    predicted_labels, _ = predict_sentiment(texts, tokenizer, model)
    
    # Calcola l'accuratezza
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Visualizza la matrice di confusione e il classification report
    plot_confusion_matrix(labels, predicted_labels)
    print_classification_report(labels, predicted_labels)
    
test_utils()
