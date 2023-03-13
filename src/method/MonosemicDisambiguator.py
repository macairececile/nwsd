from ufsac.common.POSConverter import POSConverter


class MonosemicDisambiguator:
    def __init__(self, wn):
        self.wn = wn

    def disambiguate(self, document, new_sense_tags=None):
        words = document.get_words()
        for w in words:
            if w.has_annotation(new_sense_tags):
                continue
            pos = POSConverter.to_wn_pos(w.get_annotation_value("pos"))
            for lemma in w.get_annotation_values("lemma", ";"):
                word_key = lemma + "%" + pos
                if self.wn.is_word_key_exists(word_key):
                    sense_keys = self.wn.get_sense_key_list_from_word_key(word_key)
                    if len(sense_keys) == 1:
                        w.set_annotation(new_sense_tags, sense_keys[0])
                        break
