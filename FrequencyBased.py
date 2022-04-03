""" FREQUENCY-BASED ALGORITHM
1. Process clean the sentences from punctuations, stopwords, and capital letters (Other methods can be applied)
2. Count how many important words exist in the document (Normalize it)
3. From the word_frequency, read every sentence and add up all the values of the word exist in the sentence
4. Choose the best sentences
"""
import re  # regular expression
from typing import Dict
import nltk  # nl tool kit
import string  # string
import heapq  # for sorting
import math

nltk.download("punkt")  # to access word/sentence tokenizer
nltk.download("stopwords")  # to get all stop words


def preprocess(text: str):
    """ Step 1: Pre-process sentences"""

    # lowercase && Clean all spaces
    formatted_text = text.lower()
    formatted_text = re.sub(r'\s+', ' ', formatted_text)
    # remove all stopwords and punctuations
    stopwords = nltk.corpus.stopwords.words('english')
    punctuations = string.punctuation
    tokens = [word for word in nltk.word_tokenize(formatted_text)
              if word not in stopwords and word not in punctuations]
    return ' '.join(tokens)


def create_word_freq_dic(formatted_text: str):
    """ Step 2: Find Word Frequency """

    word_freq = nltk.FreqDist(nltk.word_tokenize(formatted_text))
    highest_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= highest_freq

    return word_freq


def get_sentences_score(original_text: str, word_freq: Dict[str, float]):
    """Step 3: Get all sentence and calculate the total of the words appear"""

    sentence_result = {}
    for sentence in nltk.sent_tokenize(original_text):
        for word in nltk.word_tokenize(sentence.lower()):
            if sentence not in sentence_result:
                sentence_result[sentence] = word_freq[word]
            else:
                sentence_result[sentence] += word_freq[word]
    return sentence_result


def get_best_sentences(score_sentences: Dict[str, float], percentage: float = .1):
    """Step 4 : Get the best sentences"""
    if percentage < 0 or percentage > 1:
        raise ValueError('percentage must be between 0 and 1')

    num_sent = math.ceil(percentage * len(score_sentences))
    best_sentences = heapq.nlargest(
        num_sent, score_sentences, key=score_sentences.get)
    return best_sentences


def frequency_based_algorithm(original_text, percentage=.1, formatted_text=""):
    if formatted_text == "":
        formatted_text = preprocess(original_text)

    word_freq = create_word_freq_dic(formatted_text)
    score_sentences = get_sentences_score(original_text, word_freq)
    best_sentences = get_best_sentences(score_sentences, percentage)
    return best_sentences


if __name__ == "__main__":
    test_text = """Artificial intelligence is human like intelligence. 
                    It is the study of intelligent artificial agents. 
                    Science and engineering to produce intelligent machines. 
                    Solve problems and have intelligence. 
                    Related to intelligent behavior. 
                    Developing of reasoning machines. 
                    Learn from mistakes and successes. 
                    Artificial intelligence is related to reasoning in everyday situations."""

    a = frequency_based_algorithm(test_text, .5)
    print(a)
