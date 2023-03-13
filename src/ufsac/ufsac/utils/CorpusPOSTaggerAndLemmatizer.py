from ufsac.ufsac.utils.CorpusPOSTagger import CorpusPOSTagger
from ufsac.ufsac.utils.CorpusLemmatizer import CorpusLemmatizer


class CorpusPOSTaggerAndLemmatizer:
    def __init__(self, privilegiate_speed_over_quality=False):
        self.pos_tagger = CorpusPOSTagger(privilegiate_speed_over_quality)
        self.lemmatizer = CorpusLemmatizer()

    def tag(self, words):
        self.pos_tagger.tag(words)
        self.lemmatizer.tag(words)
