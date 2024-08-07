import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load BERT model and tokenizer
print("Loading BERT model and tokenizer for quality metric measure...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
print("BERT model and tokenizer loaded for quality metric measure.")

def get_embedding(word):
    """Get BERT embedding for a word."""
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    return last_hidden_state.mean(dim=1).detach().numpy()

def similarity(word1, word2):
    """Calculate cosine similarity between two words using BERT embeddings."""
    embedding1 = get_embedding(word1)
    embedding2 = get_embedding(word2)
    return cosine_similarity(embedding1, embedding2)[0][0]

def round_quality_metric(target_word, words):
    """Measure the quality of a round."""
    associations = [similarity(target_word, word) for word in words]
    diversity = [similarity(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))]
    quality = np.mean(associations) - np.mean(diversity)
    return quality

#Sanity Check
#Growing phase to increase (approx 100 adjectives, 200-300 nouns)
#Choose a card from the deck (500 cards)
