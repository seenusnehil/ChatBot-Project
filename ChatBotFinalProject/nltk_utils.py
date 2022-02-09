import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()  # A type of Stemmer


# Function to tokenize all the words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Function to Stem all the words given after lowering their case
def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
   sentence = ["hello", "how", "are", "you"]
   words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
   bag = [0, 1, 0, 1, 0, 0, 0]
   """

    # calling stemmer for stemming each word
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Iterate over all the words, and set 1 and 0 at their position
    # According to their presence
    bag = np.zeros(len(all_words), dtype=np.float32)  # Making initially all the position values 0 in bag
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag
