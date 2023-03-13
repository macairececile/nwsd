from embeddings.VectorOperation import *
import numpy as np


class WordVectors:
    def __init__(self, vectors, vector_count=None, vector_size=None, words=None, words_indexes=None):
        self.vector_count = len(vectors)
        self.vector_size = len(vectors[0][1])
        self.vectors = [[0.0] * self.vector_size for _ in range(self.vector_count)]
        self.words = [''] * self.vector_count
        self.words_indexes = {}
        for i in range(self.vector_count):
            self.vectors[i] = vectors[i][1]
            self.words[i] = vectors[i][0]
            self.words_indexes[self.words[i]] = i

        if vector_count and vector_size and words and words_indexes:
            self.vector_count = vector_count
            self.vector_size = vector_size
            self.vectors = vectors
            self.words = words
            self.words_indexes = words_indexes

    def get_vector_size(self):
        return self.vector_size

    def get_vocabulary(self):
        return self.words

    def has_word_vector(self, word):
        return word in self.words_indexes

    def get_word_vector(self, word):
        return self.vectors[self.words_indexes[word]]

    def get_word_vector_index(self, word):
        return self.words_indexes[word]

    def get_most_similar_words(self, word, threshold):
        if not word not in self.words_indexes:
            return []
        return self.get_most_similar_words_5(self.vectors[self.words_indexes[word]], threshold)

    def get_most_similar_words2(self, word, top_n):
        if word not in self.words_indexes:
            return []
        return self.get_most_similar_words3(self.vectors[self.words_indexes[word]], top_n)

    def get_most_similar_words3(self, word, top_n):
        nearests = [(-float('inf'), 0) for _ in range(top_n)]
        for j in range(self.vector_count):
            sim = VectorOperation.dot_product(word, self.vectors[j])
            if sim > nearests[0][0]:
                nearests[0] = (sim, j)
                nearests.sort()
        return [self.words[i[1]] for i in nearests[::-1]]

    def get_most_similar_words4(self, word):
        indexes = self.get_most_similar_word_indexes(word)
        return [self.words[i] for i in indexes]

    def get_most_similar_words_5(self, word, threshold):
        similar_words = []
        for i in range(self.vector_count):
            sim = VectorOperation.dot_product(word, self.vectors[i])
            if sim > threshold:
                similar_words.append(self.words[i])
        return similar_words

    def get_most_similar_word_indexes(self, word):
        indexes = np.arange(self.vector_count)
        sims = np.array([VectorOperation.dot_product(word, self.vectors[i]) for i in range(self.vector_count)])
        indexes = indexes[np.argsort(-sims)]
        return indexes

    def get_similar_words_and_similarity(self, word):
        if type(word) == str:
            return self.get_similar_words_and_similarity(self.vectors[self.words_indexes[word]])
        return [(self.words[i], VectorOperation.dot_product(word, self.vectors[i])) for i in range(self.vector_count)]

    def get_most_similar_words_and_similarity(self, word, threshold):
        similar_words = []
        for i in range(self.vector_count):
            sim = VectorOperation.dot_product(word, self.vectors[i])
            if sim > threshold:
                similar_words.append((self.words[i], sim))
        return similar_words


class Stuff:
    def __init__(self, sim, index):
        self.sim = sim
        self.index = index

    def compare_to(self, o):
        return self.sim.compare_to(o.sim)
