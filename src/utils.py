import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, labels=["Negative", "Neutral", "Positive"]):
    """
    Visualizza una matrice di confusione per il modello.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])  # 0=Negative, 1=Neutral, 2=Positive
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    Stampa il classification report (precision, recall, F1-score).
    """
    print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))
