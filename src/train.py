from src.model import BertEmotionClassifier
from src.preprocessing import load_data, encode_text
from torch.optim import AdamW
import torch

# Config
EPOCHS = 3
LR = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modèle
model = BertEmotionClassifier().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()
train_loader, val_loader, test_loader = load_data()

# Boucle d'entraînement
for epoch in range(EPOCHS):
  model.train()
  for batch in train_loader:
    optimizer.zero_grad()
    probs = model(batch['input_ids'].to(DEVICE),
                batch['attention_mask'].to(DEVICE))
    loss = loss_fn(probs, batch['label'].to(DEVICE))
    loss.backward()
    optimizer.step()
  print(f"Epoch {epoch+1} — Loss: {loss.item():.4f}")

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), 'model_bert.pth')