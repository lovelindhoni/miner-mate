import nltk
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
# there is two possible stemming techniques predicted for this chatbot, one is the wordnetlemmatizer and another is the porter stemmer
from string import \
    punctuation  # to get the punctuation marks which is to be ignored

from nltk.stem import PorterStemmer

stemmer = PorterStemmer() # use this or one below
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() # i mean this

from nltk.corpus import \
    stopwords  # stop word contains the set of words which can be ignored, like "this", "am", "is" etc. which doesnt have muh context to itself

nltk.download('stopwords')

to_be_ignored = set(stopwords.words('english')) | set(punctuation) # this set contains both the stop words and the punctuations which to be ignored while processing

def stem_word(word):
    return stemmer.stem(word.lower()) # uses the porter stemmer for stemming the passed word

def tokenize_sentence(s):
    return nltk.word_tokenize(s) # tokenizes the given query

def bow(tokened_sentence, all_words):
    # the bag of words function (bow) checks if the words in tokened sentence present in all_words, if it is present, then it is 1 at the index of the returned array, if it is not present then it is 0.

    word_bag = np.zeros(len(all_words), dtype=np.float32) # generates a array with zeroes with respective of no. of elements in all_words
    tokened_sentence = set([stem_word(x) for x in tokened_sentence]) # stemming the tokened_sentence
    for i, word in enumerate(all_words):
        if word in tokened_sentence:
            word_bag[i] = 1.0
    return word_bag
        
