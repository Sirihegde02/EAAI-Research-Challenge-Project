import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from quality_metric_measure import round_quality_metric, similarity
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Function to generate associations with a timeout mechanism
def generate_associations_with_timeout(prompt, model, tokenizer, num_responses=3, timeout_duration=10):
    def generate():
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
        mask_token_logits = logits[0, mask_token_index, :]
        top_k_tokens = torch.topk(mask_token_logits, num_responses, dim=1).indices[0].tolist()
        return [tokenizer.decode([token]).strip() for token in top_k_tokens if "##" not in tokenizer.decode([token]).strip()]

    with ThreadPoolExecutor() as executor:
        future = executor.submit(generate)
        try:
            return future.result(timeout=timeout_duration)
        except TimeoutError:
            print(f"Timeout occurred while processing: {prompt}")
            return []

# Growing phase function with retry mechanism and deck size verification
def growing_phase(adjectives, model, tokenizer, deck_size=50):
    """Generate a deck by growing it with nouns and adjectives iteratively."""
    deck = set()  # Use a set to ensure unique words
    max_retries = 5  # Maximum number of retries if BERT fails to generate enough words

    while len(deck) < deck_size and max_retries > 0:
        for adjective in adjectives:
            prompt = f"The {adjective} [MASK]."
            nouns = generate_associations_with_timeout(prompt, model, tokenizer)
            if not nouns:  # If no nouns are generated, retry
                print("No words generated. Retrying...")
                max_retries -= 1
                continue

            filtered_nouns = [noun for noun in nouns if not noun.startswith("##")]  # Filter out common suffixes
            deck.update(filtered_nouns)
            print(f"Current deck size: {len(deck)}")
            if len(deck) >= deck_size:
                break

        if len(deck) < deck_size:
            # If the deck is still too small, try generating more adjectives
            random_noun = random.choice(list(deck))
            adjectives = generate_associations_with_timeout(f"The {random_noun} is [MASK].", model, tokenizer, num_responses=2)

    if len(deck) < deck_size:
        print(f"No more words could be generated. Final deck size: {len(deck)}")
    else:
        print(f"Final deck size: {len(deck)}")

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

# Example usage of the growing_phase function
if __name__ == "__main__":
    # Load the model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
    print("BERT model and tokenizer loaded.")

    # Step 1: Get user input for adjectives
    adjective1 = input("Enter the first adjective: ")
    adjective2 = input("Enter the second adjective: ")

    # Step 2: Create the deck
    deck = growing_phase([adjective1, adjective2], model, tokenizer, deck_size=50)
    print("\nGenerated Deck of cards:", deck)

    # Step 3: Distribute 7 cards to each player
    hands = distribute_cards(deck)
    print("\nDistributed hands to players:")
    for player, hand in hands.items():
        print(f"{player}: {hand}")

    # Step 4: Ask the user to enter a word
    target_word = input("\nEnter a target word for the round: ")

    # Showing what each player puts down
    player_choices = {}
    for player, hand in hands.items():
        best_choice = find_best_association(hand, target_word)
        player_choices[player] = best_choice
        print(f"{player} puts down: {best_choice}")

    # Step 5: Calculating the quality metric for the round
    quality_score = round_quality_metric(target_word, list(player_choices.values()))
    print(f"\nQuality of the round: {quality_score}")


#Further changes to make:
#Discussed Aug 15th