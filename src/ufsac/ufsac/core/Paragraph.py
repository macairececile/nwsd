from ufsac.ufsac.core.ParentLexicalEntity import ParentLexicalEntity


class Paragraph(ParentLexicalEntity):
    def __init__(self, parent_document=None):
        super().__init__()
        if parent_document:
            pass

    def add_sentence(self, sentence):
        self.addChild(sentence)

    def add_sentences(self, sentences):
        self.addChildren(sentences)

    def remove_sentence(self, sentence):
        self.removeChild(sentence)

    def remove_all_sentences(self):
        self.removeAllChildren()

    def get_sentences(self):
        return self.getChildren()

    def get_words(self):
        words = []
        for s in self.get_sentences():
            words.extend(s.get_words())
        return words
