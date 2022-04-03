""" Cosine Similarity
Link: https://en.wikipedia.org/wiki/Cosine_similarity
Step by step calculations: https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
"""
import re
import nltk
import string
import numpy as np
import networkx as nx
from nltk.cluster.util import cosine_distance

nltk.download('punkt')
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')


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


def calculate_sentence_similarity(sent1, sent2):
    words1 = [word for word in nltk.word_tokenize(sent1)]
    words2 = [word for word in nltk.word_tokenize(sent2)]

    all_words = list(set(words1 + words2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)


def calculate_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = calculate_sentence_similarity(
                sentences[i], sentences[j])
    return similarity_matrix


def cosine_similarity_algo(original_sentences, number_of_sentences=10, percentage=0):
    original_sentences = [
        sentence for sentence in nltk.sent_tokenize(original_sentences)]
    formatted_sentences = [preprocess(sentence)
                           for sentence in original_sentences]
    similarity_matrix = calculate_similarity_matrix(formatted_sentences)

    # From the matrix, create a graph (with nodes and edges)
    similarity_graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(similarity_graph)
    # print(scores)
    ordered_scores = sorted(
        ((scores[i], score) for i, score in enumerate(original_sentences)), reverse=True)
    # print(ordered_scores)

    if percentage > 0:
        number_of_sentences = int(len(formatted_sentences) * percentage)

    best_sentences = []
    for sentence in range(number_of_sentences):
        best_sentences.append(ordered_scores[sentence][1])

    return original_sentences, best_sentences, ordered_scores


if __name__ == "__main__":
    print("Cosine Similarity")
    test_text = """Artificial intelligence is human like intelligence.
                    It is the study of intelligent artificial agents.
                    Science and engineering to produce intelligent machines.
                    Solve problems and have intelligence.
                    Related to intelligent behavior.
                    Developing of reasoning machines.
                    Learn from mistakes and successes.
                    Artificial intelligence is related to reasoning in everyday situations."""

    original_sentences, best_sentences, scores = cosine_similarity_algo(
        test_text, 3, 1)
    print(best_sentences)
