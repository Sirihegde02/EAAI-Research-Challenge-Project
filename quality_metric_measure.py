import numpy as np
from gensim.models import KeyedVectors

# Loading pre-trained word vectors:
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def similarity(word1, word2):
    """Calculate cosine similarity between two words using word2vec embeddings."""
    if word1 in word_vectors and word2 in word_vectors:
        return word_vectors.similarity(word1, word2)
    else:
        return 0.0

def round_quality_metric(target_word, words):
    """Measure the quality of a round."""
    associations = [similarity(target_word, word) for word in words]
    diversity = [similarity(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))]
    quality = np.mean(associations) - np.mean(diversity)
    return quality

# Example:
target_word = 'red'
words = ['firetruck', 'cherry', 'anger']
quality = round_quality_metric(target_word, words)
print(f"Quality of the round: {quality}")
