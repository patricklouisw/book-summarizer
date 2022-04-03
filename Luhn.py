""" Luhn Algorithm: https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf
1. Find a list of important words in the article (e.g. top 5 most common words used in the article)
2. Luhn algorithm tries to group every sentences based on the `important words` and the `distance` between important words
3. For each group in a sentence, max((num_of_important_words)**2 / num_of_words_in_group)
4. Repeat for each sentence and get the best sentence

e.g. 
IMPORTANT_WORDS = [fishing, salmon, cultivating, corn]
DISTANCE = 3

If the distance between important words is less than DISTANCE, group them in the same group, otherwise different group

Case 1:
I love [ fishing salmon and cultivating corn ]
            *       *    _      *      *
Score = 4**2 / 5 = 3.2

Case 2:
I love [ fishing fresh salmon ] but my sister loves [ cultivating corn ]
            *      _      *                                *        *
Score = max( [2**2/3] , [2**2/2]) = max(1.33, 2) = 2
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


def calculate_sentences_score(sentences, important_words, distance):
    scores = []
    sentence_index = 0

    for sentence in [nltk.word_tokenize(sentence) for sentence in sentences]:
        # print('------------')
        # print(sentence)

        word_index = []
        for word in important_words:
            # print(word)
            try:
                word_index.append(sentence.index(word))
            except ValueError:
                pass

        word_index.sort()
        # print(word_index)

        if len(word_index) == 0:
            continue

        # [0, 1, 5]
        groups_list = []
        group = [word_index[0]]
        i = 1  # 3
        while i < len(word_index):  # 3
            # first execution: 1 - 0 = 1 ; 1 < 2
            # second execution: 5 - 1 = 4 ; 4 > 2
            if word_index[i] - word_index[i - 1] < distance:
                group.append(word_index[i])
                #print('group', group)
            else:
                groups_list.append(group[:])
                group = [word_index[i]]
                #print('group', group)
            i += 1
        groups_list.append(group)
        # print('all groups', groups_list)

        max_group_score = 0
        for g in groups_list:
            # print(g)
            important_words_in_group = len(g)
            total_words_in_group = g[-1] - g[0] + 1
            score = 1.0 * important_words_in_group**2 / total_words_in_group
            #print('group score', score)

            if score > max_group_score:
                max_group_score = score

        scores.append((max_group_score, sentence_index))
        sentence_index += 1

    # print('final scores', scores)
    return scores


def get_top_n_words(formatted_sentences, top_n_words):
    words = [
        word for sentence in formatted_sentences for word in nltk.word_tokenize(sentence)]
    # print(words)
    frequency = nltk.FreqDist(words)
    top_n_words = [word[0] for word in frequency.most_common(top_n_words)]
    # print(top_n_words)
    return top_n_words


def luhn_algorithm(text, top_n_words, distance, number_of_sentences, percentage=0):
    original_sentences = [sentence for sentence in nltk.sent_tokenize(text)]
    # print(original_sentences)
    formatted_sentences = [preprocess(original_sentence)
                           for original_sentence in original_sentences]
    # print(formatted_sentences)

    top_n_words = get_top_n_words(formatted_sentences, top_n_words)

    sentences_score = calculate_sentences_score(
        formatted_sentences, top_n_words, distance)
    # print(sentences_score)
    if percentage > 0:
        best_sentences = heapq.nlargest(
            int(len(formatted_sentences) * percentage), sentences_score)
    else:
        best_sentences = heapq.nlargest(number_of_sentences, sentences_score)
    # print(best_sentences)
    best_sentences = [original_sentences[i] for (score, i) in best_sentences]
    # print(best_sentences)
    return original_sentences, best_sentences, sentences_score


if __name__ == "__main__":
    test_text = """Artificial intelligence is human like intelligence. 
                    It is the study of intelligent artificial agents. 
                    Science and engineering to produce intelligent machines. 
                    Solve problems and have intelligence. 
                    Related to intelligent behavior. 
                    Developing of reasoning machines. 
                    Learn from mistakes and successes. 
                    Artificial intelligence is related to reasoning in everyday situations."""

    # a = luhn_algorithm(test_text, .5)
    # print(a)
    original_sentences, best_sentences, sentences_score = luhn_algorithm(
        test_text, 5, 2, 3, 1)

    # print(original_sentences)
    # print(sentences_score)
    # print(best_sentences)
