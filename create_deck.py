# import random
# import torch
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Set a random seed for reproducibility
# random_seed = 42
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(random_seed)

# # Load BERT model and tokenizer
# print("Loading BERT model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
# print("BERT model and tokenizer loaded.")

# def get_embedding(word):
#     """Get BERT embedding for a word."""
#     inputs = tokenizer(word, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Extract the hidden states from the model output
#     hidden_states = outputs.hidden_states
#     # We are using the hidden states from the last layer
#     last_hidden_state = hidden_states[-1]
#     return last_hidden_state.mean(dim=1).detach().numpy()

# def similarity(word1, word2):
#     """Calculate cosine similarity between two words using BERT embeddings."""
#     embedding1 = get_embedding(word1)
#     embedding2 = get_embedding(word2)
#     return cosine_similarity(embedding1, embedding2)[0][0]

# def generate_associations(adjective, num_responses=3):
#     """Generate associated nouns for a given adjective using BERT embeddings."""
#     print(f"Generating associations for adjective: {adjective}")
#     # Here you can use a predefined list of nouns and select those that are most similar to the adjective
#     nouns = ['firetruck', 'cherry', 'anger', 'apple', 'rose', 'sky', 'sun', 'ocean', 'tree', 'flower']
#     associations = [(noun, similarity(adjective, noun)) for noun in nouns]
#     associations = sorted(associations, key=lambda x: x[1], reverse=True)
#     print(f"Associations for {adjective}: {associations}")
#     return [assoc[0] for assoc in associations[:num_responses]]

# def growing_phase(initial_adjectives, deck_size=30):
#     """Generate a deck by growing it with nouns and adjectives iteratively."""
#     print(f"Starting growing phase with initial adjectives: {initial_adjectives}")
#     deck = []
#     adjectives = initial_adjectives
#     while len(deck) < deck_size:
#         print(f"Current deck size: {len(deck)}")
#         for adjective in adjectives:
#             nouns = generate_associations(adjective)
#             deck.extend(nouns)
#             if len(deck) >= deck_size:
#                 break
#         new_adjectives = [noun for noun in nouns if noun not in deck]
#         if not new_adjectives:
#             print("No new adjectives to continue growing the deck. Breaking out.")
#             break
#         adjectives = new_adjectives
#     print(f"Final deck: {deck[:deck_size]}")
#     return deck[:deck_size]

# def round_quality_metric(target_word, words):
#     """Measure the quality of a round."""
#     associations = [similarity(target_word, word) for word in words]
#     diversity = [similarity(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))]
#     quality = np.mean(associations) - np.mean(diversity)
#     return quality

# # Example usage of deck creation
# initial_adjectives = ['red', 'beautiful']
# print("Creating deck...")
# deck1 = growing_phase(initial_adjectives, deck_size=30)
# print(f"Generated Deck 1: {deck1}")

# # Example usage of quality metric
# target_word = 'red'
# words = ['firetruck', 'cherry', 'anger']
# quality = round_quality_metric(target_word, words)
# print(f"Quality of the round: {quality}")

import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set a random seed for reproducibility
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Load BERT model and tokenizer
print("Loading BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
print("BERT model and tokenizer loaded.")

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

def generate_associations(adjective, num_responses=3):
    """Generate associated nouns for a given adjective using BERT's masked language model."""
    print(f"Generating associations for adjective: {adjective}")
    prompt = f"The {adjective} [MASK]."
    inputs = tokenizer(prompt, return_tensors='pt')
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, num_responses, dim=1).indices[0].tolist()
    associations = [(tokenizer.decode([token]).strip(), similarity(adjective, tokenizer.decode([token]).strip())) for token in top_k_tokens]
    associations = sorted(associations, key=lambda x: x[1], reverse=True)
    print(f"Associations for {adjective}: {associations}")
    return [assoc[0] for assoc in associations[:num_responses]]

def growing_phase(initial_adjectives, deck_size=30):
    """Generate a deck by growing it with nouns and adjectives iteratively."""
    print(f"Starting growing phase with initial adjectives: {initial_adjectives}")
    deck = set()
    adjectives = initial_adjectives
    seen_adjectives = set(adjectives)

    while len(deck) < deck_size:
        print(f"Current deck size: {len(deck)}")
        new_nouns = []

        for adjective in adjectives:
            nouns = generate_associations(adjective)
            new_nouns.extend(nouns)

        for noun in new_nouns:
            if len(deck) < deck_size:
                deck.add(noun)

        new_adjectives = [noun for noun in new_nouns if noun not in seen_adjectives]
        seen_adjectives.update(new_adjectives)

        if not new_adjectives:
            print("No new adjectives to continue growing the deck. Breaking out.")
            break

        adjectives = new_adjectives

    final_deck = list(deck)[:deck_size]
    print(f"Final deck: {final_deck}")
    return final_deck

def round_quality_metric(target_word, words):
    """Measure the quality of a round."""
    associations = [similarity(target_word, word) for word in words]
    diversity = [similarity(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))]
    quality = np.mean(associations) - np.mean(diversity)
    return quality

# Example usage of deck creation
initial_adjectives = ['red', 'beautiful']
print("Creating deck...")
deck1 = growing_phase(initial_adjectives, deck_size=30)
print(f"Generated Deck 1: {deck1}")

# Example usage of quality metric
target_word = 'red'
words = ['firetruck', 'cherry', 'anger']
quality = round_quality_metric(target_word, words)
print(f"Quality of the round: {quality}")
