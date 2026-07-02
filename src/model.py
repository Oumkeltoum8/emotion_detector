import torch
import torch.nn as nn
from transformers import BertModel

class BertEmotionClassifier(nn.Module):
  def __init__(self, n_classes=6):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Linear(768, 64)
    self.out = nn.Linear(64, n_classes)

  def forward(self, input_ids, attention_mask):
    output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    cls = output.last_hidden_state[:, 0, :]
    x = self.dropout(torch.relu(self.fc(cls)))
    return torch.softmax(self.out(x), dim=1)