from typing import List
import nltk
import os

# Importing NL Summarization methods
from FrequencyBased import frequency_based_algorithm
from Luhn import luhn_algorithm
from CosineSimilarity import cosine_similarity_algo


def save_summary(title: str, original_sentences: str, best_sentences: List[str]):
    HTML_TEMPLATE = """<html>
    <head>
        <title>{0}</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    </head>
    <body>{1}</body>

    </html>"""
    text = ''
    for sentence in nltk.sent_tokenize(original_sentences):
        if sentence in best_sentences:
            text += str(sentence).replace(sentence, f"<mark>{sentence}</mark>")
        else:
            text += sentence

    save_file = open(os.path.join(title + '.html'), 'wb')
    html_file = HTML_TEMPLATE.format(title, text)
    save_file.write(html_file.encode('utf-8'))
    save_file.close()


if __name__ == "__main__":
    # test_text = """Artificial intelligence is human like intelligence.
    #                 It is the study of intelligent artificial agents.
    #                 Science and engineering to produce intelligent machines.
    #                 Solve problems and have intelligence.
    #                 Related to intelligent behavior.
    #                 Developing of reasoning machines.
    #                 Learn from mistakes and successes.
    #                 Artificial intelligence is related to reasoning in everyday situations."""
    with open("Infrastructure-summary.txt", "r") as f:

        all_text = f.read()

        # a = frequency_based_algorithm(all_text, .5)
        # print(a)
        # save_summary("Test", test_text, a)

        original_sentences, best_sentences, scores = cosine_similarity_algo(
            all_text, 3, .2)
        save_summary("Summary Infrastructure", all_text, best_sentences)
