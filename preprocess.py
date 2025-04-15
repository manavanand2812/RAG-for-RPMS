import json
import re
import nltk
from nltk.tokenize import word_tokenize

# # Download the punkt tokenizer (if not already downloaded)
# nltk.download('punkt_tab')

# Load the documents from the saved JSON file
with open('documents.json', 'r') as doc_file:
    documents = json.load(doc_file)

# Preprocess function: clean and tokenize text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (keeping only alphabets and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    return words

# Preprocess all documents
preprocessed_documents = [preprocess_text(doc) for doc in documents]


with open('documents.json', 'w') as f:
    json.dump(preprocessed_documents, f)

# # Show the preprocessed documents
# for i, doc in enumerate(preprocessed_documents):
#     print(f"Document {i+1} Preprocessed: {doc}\n")


