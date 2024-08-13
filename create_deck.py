import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from quality_metric_measure import round_quality_metric, similarity

#Setting a random seed for reproducibility:
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

#Loading BERT model and tokenizer:
print("Loading BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
print("BERT model and tokenizer loaded.")

#This function generated nouns that fit the context of the adjectives:
def generate_associations(prompt, num_responses=3):
    """Generate associated words for a given prompt using BERT's masked language model."""
    inputs = tokenizer(prompt, return_tensors='pt')
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, num_responses, dim=1).indices[0].tolist()
    return [tokenizer.decode([token]).strip() for token in top_k_tokens if "##" not in tokenizer.decode([token]).strip()]

#This function generates nouns or adjectives until deck size == 42
def growing_phase(adjectives, deck_size=500):
    """Generate a larger deck by growing it with nouns and adjectives iteratively."""
    deck = set()  # Use a set to ensure unique words
    while len(deck) < deck_size:
        for adjective in adjectives:
            prompt = f"The {adjective} [MASK]."
            nouns = generate_associations(prompt)
            filtered_nouns = [noun for noun in nouns if not noun.startswith("##")]  # Filter out common suffixes
            deck.update(filtered_nouns)
            if len(deck) >= deck_size:
                break
        if len(deck) >= deck_size:
            break
        random_noun = random.choice(list(deck))
        adjectives = generate_associations(f"The {random_noun} is [MASK].", num_responses=2)
    return list(deck)[:deck_size]

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

#Step 1: Getting user input for adjectives:
adjective1 = input("Enter the first adjective: ")
adjective2 = input("Enter the second adjective: ")

#Step 2: Creating the deck:
deck = growing_phase([adjective1, adjective2], deck_size=42)
print("\nGenerated Deck of 42 cards:", deck)

#Step 3: Distribute 7 cards to each player:
hands = distribute_cards(deck)
print("\nDistributed hands to players:")
for player, hand in hands.items():
    print(f"{player}: {hand}")

#Step 4: Ask the user to enter a word:
target_word = input("\nEnter a target word for the round: ")

#Showing what each player puts down:
player_choices = {}
for player, hand in hands.items():
    best_choice = find_best_association(hand, target_word)
    player_choices[player] = best_choice
    print(f"{player} puts down: {best_choice}")

#Step 5: Calculating the quality metric for the round:
quality_score = round_quality_metric(target_word, list(player_choices.values()))
print(f"\nQuality of the round: {quality_score}")
