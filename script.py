from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Caricamento del modello e del tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    sentiment = torch.argmax(logits, dim=1).item()

    if sentiment == 0:
        return "Negative"
    elif sentiment == 1:
        return "Neutral"
    else:
        return "Positive"

# Test della funzione con un esempio di tweet
example_text = "I love using Hugging Face models for sentiment analysis!"
print(analyze_sentiment(example_text))
