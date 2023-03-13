from ufsac.common.POSConverter import POSConverter
from ufsac.common.WordnetHelper import WordnetHelper


class CorpusLemmatizer:
    def __init__(self, lemma_annotation_name="lemma", wn_version=30):
        self.lemma_annotation_name = lemma_annotation_name
        self.wn = WordnetHelper.wn(wn_version)

    def tag(self, words):
        self.add_wn_morphy_lemma_annotations(words)

    def add_wn_morphy_lemma_annotations(self, words):
        for word in words:
            if word.has_annotation(self.lemma_annotation_name):
                continue
            pos = POSConverter.to_wn_pos(word.get_annotation_value("pos"))
            if pos == "x":
                continue
            word.set_annotation(self.lemma_annotation_name, self.wn.morphy(word.get_value(), pos))
