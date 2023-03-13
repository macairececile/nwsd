from ufsac.ufsac.core.ParentLexicalEntity import ParentLexicalEntity


class Document(ParentLexicalEntity):
    def __init__(self, parent_corpus=None):
        super().__init__()
        if parent_corpus:
            pass

    def add_paragraph(self, paragraph):
        self.addChild(paragraph)

    def get_paragraphs(self):
        return self.getChildren()

    def get_sentences(self):
        sentences = []
        for p in self.get_paragraphs():
            sentences.extend(p.get_sentences())
        return sentences

    def get_words(self):
        words = []
        for p in self.get_paragraphs():
            words.extend(p.get_words())
        return words
