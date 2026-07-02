import re, nltk, spacy
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# Chargement outils
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128  # ajuste selon la longueur de tes textes
    )

dataset = load_dataset("dair-ai/emotion")

# Applique la tokenisation à tout le DatasetDict d'un coup
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Indique à PyTorch quelles colonnes utiliser, et sous quel format
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Nettoyage d'une phrase
def clean_text(text):
  text = text.lower()
  text = re.sub(r'http\S+|@\w+|[^a-z\s]', '', text)
  words = [w for w in text.split() if w not in stop_words]
  doc = nlp(' '.join(words))
  return ' '.join([t.lemma_ for t in doc])


# Tokenisation BERT d'une phrase
def encode_text(text):
  return tokenizer.encode_plus(
    clean_text(text),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
  )

# Charger le dataset complet et retourner des DataLoaders PyTorch
def load_data(batch_size=16):
  train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(tokenized_datasets['validation'], batch_size=batch_size)
  test_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size)
  return train_loader, val_loader, test_loader