import nltk
from nltk.corpus import wordnet as wn
import random
import statistics
import spacy
import pandas as pd
import ast


def get_sense_keys_from_lemma(lemma):
    # Obtenir un dictionnaire de sens pour chaque lemma à partir de wordnet
    senses = []
    for synset in wn.synsets(lemma, lang='fra'):
        senses.extend([str(i._key) for i in synset.lemmas()])
    return senses


def get_lemma_from_sentence(sentence, spacy_model):
    doc = spacy_model(sentence)
    return [token.lemma_ for token in doc]


def get_score_per_words(sentences, spacy_model):
    scores_per_words = []
    for s in sentences:
        lemmas = get_lemma_from_sentence(s, spacy_model)
        for lem in lemmas:
            senses = get_sense_keys_from_lemma(lem)
            if len(senses) != 0:
                score = 1 / len(senses)
            else:
                score = 0
            scores_per_words.append(score)
    return scores_per_words


def read_data_from_csv(file):
    return pd.read_csv(file, sep='\t')


def get_sentences_from_corpus(file):
    data = read_data_from_csv(file)
    sentences = data["sentence"].tolist()
    return sentences


def random_assignment(corpus_file):
    spacy_model = spacy.load("fr_dep_news_trf")
    sentences = get_sentences_from_corpus(corpus_file)
    senses_per_words = get_score_per_words(sentences, spacy_model)
    random_assignment = (1 / len(senses_per_words)) * sum(senses_per_words)
    print("Random assignment : ", random_assignment)
    # print("Random assignment: ", round(statistics.mean(random_assignment), 3))


# print("corpus A medical : ")
# random_assignment("/home/getalp/macairec/data/corpus_evaluation_s2p/A_medical/a_medical.csv")
# print("corpus B stories :")
# random_assignment("/home/getalp/macairec/data/corpus_evaluation_s2p/B_stories/b_stories.csv")
# print("corpus C emails :")
# random_assignment("/home/getalp/macairec/data/corpus_evaluation_s2p/C_emails/c_emails_test.csv")
# print("corpus D stories2 : ")
# random_assignment("/home/getalp/macairec/data/corpus_evaluation_s2p/D_stories2/d_stories2.csv")
# print("corpus E polysemous : ")
# random_assignment("/home/getalp/macairec/data/corpus_evaluation_s2p/E_polysemy/e_polysemous.csv")
random_assignment("/home/getalp/macairec/data/corpus_evaluation_s2p/all/all.csv")
