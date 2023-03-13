from ufsac.common.POSConverter import POSConverter


class FirstSenseDisambiguator:
    def __init__(self, wn_or_mfs_path=None):
        if type(wn_or_mfs_path) == str:
            try:
                with open(wn_or_mfs_path, 'r') as f:
                    self.mfs = {}
                    for line in f:
                        parts = line.split(' ')
                        self.mfs[parts[0]] = parts[1]
                self.wn = None
            except Exception as e:
                raise RuntimeError(e)
        else:
            self.wn = wn_or_mfs_path
            self.mfs = None

    def disambiguate(self, corpus, new_sense_tags=None):
        words = corpus.get_words()
        for w in words:
            if not w.has_annotation(new_sense_tags):
                pos = POSConverter.to_wn_pos(w.get_annotation_value("pos"))
                lemmas = w.get_annotation_values("lemma", ";")
                for lemma in lemmas:
                    word_key = lemma + "%" + pos
                    if self.wn.is_word_key_exists(word_key):
                        if self.wn is not None:
                            sense_key = self.wn.get_first_sense_key_from_word_key(word_key)
                        else:
                            sense_key = self.mfs[word_key]
                        w.set_annotation(new_sense_tags, sense_key)
                        break
