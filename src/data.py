import pandas as pd
from datasets import load_dataset

def load_and_preprocess_data(dataset_name="sentiment140", split="train"):
    """
    Carica il dataset sentiment140 o un altro dataset simile e lo pre-processa.
    """
    # Caricamento del dataset utilizzando la libreria 'datasets' (Hugging Face)
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    # Pre-processing dei dati: pulizia, tokenizzazione, rimozione di stopword, ecc.
    # Qui potresti aggiungere il codice per pre-processare i tuoi dati.
    
    # Se il dataset è già pronto per l'analisi del sentiment, non serve molta pulizia.
    # Ad esempio, ci concentreremo solo sul testo e sull'etichetta (sentiment)
    texts = dataset['text']
    labels = dataset['sentiment']
    
    return texts, labels
