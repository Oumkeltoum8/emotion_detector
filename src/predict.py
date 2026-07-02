import torch
from src.preprocessing import encode_text
from src.model import BertEmotionClassifier

EMOTIONS = ['sadness','joy','love','anger','fear','surprise']

# Charger le modèle sauvegardé
model = BertEmotionClassifier(n_classes=6)
model.load_state_dict(torch.load('model_bert.pth'))
model.eval()

def predict_emotion(text):
  encoded = encode_text(text)
  with torch.no_grad():
    probs = model(
      encoded['input_ids'],
      encoded['attention_mask']
    )
  emotion = EMOTIONS[probs.argmax()]
  confidence = probs.max().item() * 100
  return emotion, confidence

# Test rapide
if __name__ == '__main__':
  e, c = predict_emotion("I can't stop smiling today!")
  print(f"→ {e} ({c:.0f}%)")