from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch

EMOTIONS = ['sadness','joy','love','anger','fear','surprise']

def evaluate(model, test_loader, device):
  all_preds, all_labels = [], []
  model.eval()
  with torch.no_grad():
    for batch in test_loader:
      probs = model(batch['input_ids'].to(device),
                  batch['attention_mask'].to(device))
      preds = probs.argmax(dim=1).cpu().tolist()
      all_preds.extend(preds)
      all_labels.extend(batch['labels'].tolist())

  # Rapport complet
  print(classification_report(all_labels, all_preds,
      target_names=EMOTIONS))

  # Matrice de confusion
  cm = confusion_matrix(all_labels, all_preds)
  sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
  plt.savefig('confusion_matrix.png')
  plt.show()