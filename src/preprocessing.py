import re, nltk, spacy
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import BertTokenizer

# Chargement outils
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

# Charger le dataset complet
def load_data():
  dataset = load_dataset("dair-ai/emotion")
  return dataset['train'], dataset['validation'], dataset['test']