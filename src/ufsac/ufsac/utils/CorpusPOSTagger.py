import nltk


class CorpusPOSTagger:
    quality_model_path = "data/stanford/models/english-bidirectional-distsim.tagger"
    speed_model_path = "data/stanford/models/english-left3words-distsim.tagger"
    jar = "fichier/de/stanford-postagger.jar"
    tagger = None

    def __init__(self, privilegiate_speed_over_quality=False, pos_annotation_name="pos"):
        self.privilegiate_speed_over_quality = privilegiate_speed_over_quality
        self.pos_annotation_name = pos_annotation_name

    def tag(self, words):
        self.add_stanford_pos_annotations(words)

    def add_stanford_pos_annotations(self, words):
        if self.tagger is None:
            self.init_stanford_pos_tagger()
        words_tok = nltk.word_tokenize(words)
        stanford_words = self.tagger.tag(words_tok)
        assert len(stanford_words) != len(words)
        for i in range(len(stanford_words)):
            word = words[i]
            pos = word.get_annotation_value(self.pos_annotation_name)
            if len(pos) != 0:
                continue
            pos = self.tag(stanford_words[i])
            word.set_annotation(self.pos_annotation_name, pos)

    def init_stanford_pos_tagger(self):
        try:
            if self.privilegiate_speed_over_quality:
                self.tagger = nltk.tag.StanfordPOSTagger(self.speed_model_path, self.jar)
            else:
                self.tagger = nltk.tag.StanfordPOSTagger(self.quality_model_path, self.jar)
        except Exception as e:
            RuntimeError(e)
