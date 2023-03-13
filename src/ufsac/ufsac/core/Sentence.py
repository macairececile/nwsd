from ufsac.ufsac.core.LexicalEntity import *
from ufsac.ufsac.core.Word import Word


class Sentence(LexicalEntity):
    def __init__(self, sentence_to_copy=None):
        self.words = []
        if sentence_to_copy is None:
            super().__init__()
        elif type(sentence_to_copy) == Sentence():
            super(sentence_to_copy)
            for w in sentence_to_copy.get_words():
                self.add_word(Word(w))
        elif type(sentence_to_copy) == str:
            self.add_words_from_string(sentence_to_copy)
        elif type(sentence_to_copy) == list():
            for w in sentence_to_copy:
                self.add_word(w)

    def add_word(self, word):
        self.words.append(word)

    def remove_word(self, word):
        self.words.remove(word)

    def remove_all_words(self):
        self.words = []

    def get_words(self):
        return self.words

    def limit_sentence_length(self, max_length: int):
        if len(self.words) > max_length:
            self.words = self.words[:max_length]

    def add_words_from_string(self, value):
        words_array = value.split(' ')
        for word_in_array in words_array:
            self.add_word(Word(word_in_array))

    def to_string(self):
        ret = ""
        for word in self.get_words():
            ret += word.to_string() + " "
        return ret.strip()
