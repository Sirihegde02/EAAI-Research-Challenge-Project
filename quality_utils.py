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
    diversity = [similarity(words[i], words[j]) for i in range(len(words)) for j in range(i + 1, len(words))]
    quality = np.mean(associations) - np.mean(diversity)
    return quality

def distribute_cards(deck, players=6, cards_per_player=7):
    """Distribute cards among players."""
    hands = {}
    for player in range(1, players + 1):
        hands[f"Player-{player}"] = random.sample(deck, cards_per_player)
        deck = [card for card in deck if card not in hands[f"Player-{player}"]]
    return hands

def find_best_association(player_hand, target_word):
    """Find the best association from a player's hand for a given word."""
    return max(player_hand, key=lambda word: similarity(target_word, word))

def simulate_rounds(deck, num_rounds=10):
    """Simulate multiple rounds using the generated deck and log results."""
    with open("round_simulation_results.txt", "w") as file:
        for i in range(num_rounds):
            target_word = random.choice(deck)
            hands = distribute_cards(deck, players=6, cards_per_player=7)
            file.write(f"\n--- Round {i + 1} ---\n")
            file.write(f"Target word: {target_word}\n")
            
            player_choices = {}
            for player, hand in hands.items():
                best_choice = find_best_association(hand, target_word)
                player_choices[player] = best_choice
                file.write(f"{player} puts down: {best_choice}\n")
            
            quality_score = round_quality_metric(target_word, list(player_choices.values()))
            file.write(f"Quality of the round: {quality_score}\n\n")
    print("Simulation complete. Results saved to 'round_simulation_results.txt'.")

if __name__ == "__main__":
    try:
        with open("generated_deck.txt", "r") as file:
            deck = file.read().splitlines()
            print(f"Loaded Deck of {len(deck)} cards: {deck}")
            simulate_rounds(deck, num_rounds=10)
    except FileNotFoundError:
        print("Error: 'generated_deck.txt' not found. Please generate a deck first.")
