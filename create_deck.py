import random
import torch
import threading
from transformers import AutoTokenizer, AutoModelForMaskedLM
from quality_metric_measure import round_quality_metric, similarity

# Setting a random seed for reproducibility:
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Loading BERT model and tokenizer:
print("Loading BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
print("BERT model and tokenizer loaded.")

# Function to generate nouns or adjectives based on a prompt with a timeout:
def generate_associations_with_timeout(prompt, num_responses=3, timeout=10):
    """Generate associated words for a given prompt using BERT's masked language model with a timeout."""
    def target_function(result):
        inputs = tokenizer(prompt, return_tensors='pt')
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_k_tokens = torch.topk(mask_token_logits, num_responses, dim=1).indices[0].tolist()
        result.extend([tokenizer.decode([token]).strip() for token in top_k_tokens if "##" not in tokenizer.decode([token]).strip()])

    result = []
    thread = threading.Thread(target=target_function, args=(result,))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("Operation timed out while generating associations.")
        return []
    
    return result

# Function to generate a large deck
def growing_phase(adjectives, deck_size=500):
    """Generate a deck by growing it with nouns and adjectives iteratively."""
    deck = set()  # Use a set to ensure unique words
    while len(deck) < deck_size:
        print(f"Current deck size: {len(deck)}")
        for adjective in adjectives:
            prompt = f"The {adjective} [MASK]."
            nouns = generate_associations_with_timeout(prompt)
            filtered_nouns = [noun for noun in nouns if not noun.startswith("##")]  # Filter out common suffixes
            deck.update(filtered_nouns)
            if len(deck) >= deck_size:
                break
        if len(deck) >= deck_size:
            break
        if not deck:  # If no words generated, avoid an infinite loop
            print("No words generated. Exiting the growing phase.")
            break
        random_noun = random.choice(list(deck))
        adjectives = generate_associations_with_timeout(f"The {random_noun} is [MASK].", num_responses=2)
    print(f"Final deck size: {len(deck)}")
    return list(deck)[:deck_size]

# Save the deck to a text file
def save_deck_to_file(deck, filename="generated_deck.txt"):
    """Save the generated deck to a text file."""
    with open(filename, "w") as file:
        for card in deck:
            file.write(card + "\n")
    print(f"Deck saved to {filename}")

# Load the deck from a text file
def load_deck_from_file(filename="generated_deck.txt"):
    """Load a deck from a text file."""
    with open(filename, "r") as file:
        deck = [line.strip() for line in file.readlines()]
    return deck

# Function to distribute cards among players:
def distribute_cards(deck, players=6, cards_per_player=7):
    """Distribute cards among players."""
    hands = {}
    for player in range(1, players + 1):
        hands[f"Player-{player}"] = random.sample(deck, cards_per_player)
        deck = [card for card in deck if card not in hands[f"Player-{player}"]]
    return hands

# Function to find the best association from a player's hand:
def find_best_association(player_hand, target_word):
    """Find the best association from a player's hand for a given word."""
    return max(player_hand, key=lambda word: similarity(target_word, word))

# Step 1: Generate and save the deck
adjective1 = input("Enter the first adjective: ")
adjective2 = input("Enter the second adjective: ")
deck = growing_phase([adjective1, adjective2], deck_size=500)
save_deck_to_file(deck)

# Step 2: Load the deck for sanity checks
deck = load_deck_from_file()
print("\nLoaded Deck of 500 cards:", deck)

# Step 3: Distribute 7 cards to each player:
hands = distribute_cards(deck)
print("\nDistributed hands to players:")
for player, hand in hands.items():
    print(f"{player}: {hand}")

# Step 4: Perform multiple rounds for sanity check
rounds = int(input("\nEnter the number of rounds for sanity check: "))
for i in range(rounds):
    target_word = random.choice(deck)
    print(f"\nRound {i+1}, Target word: {target_word}")

    player_choices = {}
    for player, hand in hands.items():
        best_choice = find_best_association(hand, target_word)
        player_choices[player] = best_choice
        print(f"{player} puts down: {best_choice}")

    # Step 5: Calculating the quality metric for the round:
    quality_score = round_quality_metric(target_word, list(player_choices.values()))
    print(f"Quality of round {i+1}: {quality_score}")
