import sys
import os

# Aggiungi la cartella principale del progetto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ora importa il modulo 'data' da 'src'
from src import data

# Test per verificare il caricamento dei dati
def test_load_and_preprocess_data():
    texts, labels = data.load_and_preprocess_data()
    print(f"Sample of texts: {texts[:5]}")  # Mostra i primi 5 testi
    print(f"Sample of labels: {labels[:5]}")  # Mostra le prime 5 etichette

test_load_and_preprocess_data()
