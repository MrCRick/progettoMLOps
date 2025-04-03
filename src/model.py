from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Carica il modello pre-addestrato per l'analisi del sentiment.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return tokenizer, model

def predict_sentiment(texts, tokenizer, model):
    """
    Predice il sentiment per una lista di testi utilizzando il modello caricato.
    """
    # Tokenizzazione dei testi
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Eseguiamo il modello sui dati
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Otteniamo le probabilità per ogni classe (positivo, neutro, negativo)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    
    # Indici della classe con probabilità più alta
    predicted_labels = torch.argmax(probabilities, dim=-1)
    
    return predicted_labels, probabilities
