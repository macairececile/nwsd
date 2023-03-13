import jpype
from ufsac.ufsac.core.Sentence import Sentence
import os

# PYTHONPATH = "/home/cecilemacaire/Bureau/demo/nwsd/src/"
PYTHONPATH = os.getcwd() + "/../nwsd/src/"

class WordnetHelper:
    wordnetDirectoryPath = PYTHONPATH + "ufsac/data/wordnet/"

    def __init__(self, wordnet_directory_path, version):
        self.wordnet = None
        self.version = version
        self.synset_to_hypernyms_synsets = {}
        self.sense_to_synset = {}
        self.synset_to_sense_list = {}
        self.synset_to_gloss = {}
        self.sense_to_related_synsets = {}
        self.synset_to_related_synsets = {}
        self.synset_to_instance_hypernyms_synsets = {}
        self.synset_to_hyponyms_synsets = {}
        self.synset_to_instance_hyponyms_synsets = {}
        self.synset_to_antonyms_synsets = {}
        self.synset_to_similar_to_synsets = {}
        self.word_key_to_sense_list = {}
        self.sense_key_to_sense_number = {}
        self.sense_number_to_sense_key = {}
        self.word_key_to_first_sense_key = {}
        self.morphyy = None
        self.load_glosses = True
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=PYTHONPATH + "../lib/edu.mit.jwi_2.4.0.jar")
        self.load(wordnet_directory_path)

    @staticmethod
    def wn16():
        return WordnetHelper.wn(16)

    @staticmethod
    def wn21():
        return WordnetHelper.wn(21)

    @staticmethod
    def wn30():
        return WordnetHelper.wn(30)

    @staticmethod
    def wn(version=None):
        if version is None:
            return WordnetHelper.wn(30)
        return WordnetHelper(f"{WordnetHelper.wordnetDirectoryPath}/{version}/dict", version)

    def get_synset_key_from_sense_key(self, sense_key):
        return self.sense_to_synset[sense_key]

    def get_sense_key_list_from_synset_key(self, synset_key):
        return self.synset_to_sense_list[synset_key]

    def get_sense_key_list_from_sense_key(self, sense_key):
        return self.synset_to_sense_list[self.sense_to_synset][sense_key]

    def get_sense_key_from_sense_number(self, sense_number):
        return self.sense_number_to_sense_key[sense_number]

    def get_sense_number_from_sense_key(self, sense_key):
        return self.sense_key_to_sense_number[sense_key]

    def get_gloss_from_synset_key(self, synset_key):
        return self.synset_to_gloss[synset_key]

    def get_gloss_from_sense_key(self, sense_key):
        return self.synset_to_gloss[self.sense_to_synset][sense_key]

    def get_related_synsets_key_from_sense_key(self, sense_key):
        return self.sense_to_related_synsets[sense_key]

    def get_related_synsets_key_from_synset_key(self, synset_key):
        return self.synset_to_related_synsets[synset_key]

    def get_hypernym_synset_keys_from_synset_key(self, synset_key):
        return self.synset_to_hypernyms_synsets[synset_key]

    def get_instance_hypernym_synset_keys_from_synset_key(self, synset_key):
        return self.synset_to_instance_hypernyms_synsets[synset_key]

    def get_hyponym_synset_keys_from_synset_key(self, synset_key):
        return self.synset_to_hyponyms_synsets[synset_key]

    def get_instance_hyponym_synset_keys_from_synset_key(self, synset_key):
        return self.synset_to_instance_hyponyms_synsets[synset_key]

    def get_antonym_synset_keys_from_synset_key(self, synset_key):
        return self.synset_to_antonyms_synsets[synset_key]

    def get_similar_to_synset_keys_from_synset_key(self, synset_key):
        return self.synset_to_similar_to_synsets[synset_key]

    def get_vocabulary(self):
        return self.word_key_to_sense_list.keys()

    def get_all_sense_keys(self):
        return self.sense_to_synset.keys()

    def get_sense_key_list_from_word_key(self, word_key):
        return self.word_key_to_sense_list[word_key]

    def get_first_sense_key_from_word_key(self, word_key):
        return self.word_key_to_first_sense_key[word_key]

    def get_first_sense_key_index_from_word_key(self, word_key):
        first_sense_key = self.get_first_sense_key_from_word_key(word_key)
        sense_keys = self.get_sense_key_list_from_word_key(word_key)

        for i in range(len(sense_keys)):
            if sense_keys[i] == first_sense_key:
                return i
        return -1

    def is_sense_key_exists(self, sense_key):
        return sense_key in self.sense_to_synset

    def is_synset_key_exists(self, synset_key):
        return synset_key in self.synset_to_sense_list

    def is_lemma_exists(self, lemma):
        for word_key in self.word_key_to_sense_list.keys():
            if lemma == word_key[:word_key.indexOf("%")]:
                return True
        return False

    def is_word_key_exists(self, word_key):
        return word_key in self.word_key_to_sense_list

    def get_version(self):
        return self.version

    def morphy(self, surface_form, pos_tag=None, pos=None):
        if pos_tag is None or len(pos_tag) == 0:
            return self.morphy(surface_form)
        if pos is None and pos_tag is None:
            return self.morphy(surface_form, None)
        if pos:
            try:
                return self.morphyy.findStems(surface_form, pos)[0]
            except Exception as e:
                raise RuntimeError(e)
        pos_java = jpype.JClass('edu.mit.jwi.item.POS')
        return self.morphy(surface_form, pos_java().getPartOfSpeech(pos_tag[0]))

    def load(self, wordnet_dict_path):
        self.sense_key_to_sense_number = {}
        self.sense_number_to_sense_key = {}
        self.word_key_to_first_sense_key = {}
        dictionary_java = jpype.JClass('edu.mit.jwi.Dictionary')
        file = jpype.JClass('java.io.File')
        self.wordnet = dictionary_java(file(wordnet_dict_path))
        self.wordnet.open()
        wordnet_stemmer = jpype.JClass('edu.mit.jwi.morph.WordnetStemmer')
        self.morphyy = wordnet_stemmer(self.wordnet)
        iise = self.wordnet.getSenseEntryIterator()

        while iise.hasNext():
            ise = iise.next()
            sense_key = str(ise.getSenseKey()).lower()
            if self.version == 30:
                sense_key = sense_key.replace("%5", "%3")
            sense_lemma = str(ise.getSenseKey().getLemma()).lower()
            sense_pos = ise.getSenseKey().getPOS().getTag()
            sense_number = sense_lemma + "%" + sense_pos + "#" + str(ise.getSenseNumber())
            self.sense_key_to_sense_number[sense_key] = sense_number
            self.sense_number_to_sense_key[sense_number] = sense_key

            if ise.getSenseNumber() == 1:
                self.word_key_to_first_sense_key[sense_lemma + "%" + sense_pos] = sense_key

        pos_java = jpype.JClass('edu.mit.jwi.item.POS')
        for pos in pos_java.values():
            iis = self.wordnet.getSynsetIterator(pos)
            while iis.hasNext():
                isss = iis.next()
                self.add_synset(isss, pos)

    def add_synset(self, isss, pos):
        sense_key_list = []
        synset_key = "" + pos.getTag() + str(isss.getOffset())
        self.synset_to_hypernyms_synsets[synset_key] = self.load_hypernyms(isss)
        self.synset_to_instance_hypernyms_synsets[synset_key] = self.load_instance_hypernyms(isss)
        self.synset_to_hyponyms_synsets[synset_key] = self.load_hyponyms(isss)
        self.synset_to_instance_hyponyms_synsets[synset_key] = self.load_instance_hyponyms(isss)
        self.synset_to_related_synsets[synset_key] = self.load_semantic_relations(isss)
        antonyms_synsets = set()
        self.synset_to_similar_to_synsets[synset_key] = self.load_similar_to(isss)

        for iw in isss.getWords():
            lemma = str(iw.getLemma()).lower()
            word_key = lemma + "%" + iw.getPOS().getTag()
            sense_key = str(iw.getSenseKey()).lower()
            if self.version == 30:
                sense_key = sense_key.replace("%5", "%3")
            if sense_key not in self.sense_key_to_sense_number.keys():
                continue
            if sense_key in sense_key_list:
                continue
            sense_key_list.append(sense_key)
            self.sense_to_synset[sense_key] = synset_key
            self.sense_to_related_synsets[sense_key] = self.load_relations(isss, iw)
            antonyms_synsets.update(self.load_antonyms(iw))
            if word_key in self.word_key_to_sense_list:
                self.word_key_to_sense_list[word_key].append(sense_key)
            else:
                self.word_key_to_sense_list[word_key] = [sense_key]
        self.synset_to_antonyms_synsets[synset_key] = [antonyms_synsets]
        self.synset_to_sense_list[synset_key] = sense_key_list
        if self.load_glosses:
            self.synset_to_gloss[synset_key] = Sentence(isss.getGloss())

    def load_relations(self, synset, word):
        related_synsets = []
        for iPointer, iSynsetIDList in synset.getRelatedMap().items():
            for iwd in iSynsetIDList:
                related_synset = self.wordnet.getSynset(iwd)
                related_synset_key = "{}{}".format(related_synset.getPOS().getTag(), related_synset.getOffset())
                related_synsets.append(related_synset_key)

        for iPointer, iWordIDList in word.getRelatedMap().items():
            for iwd in iWordIDList:
                related_word = self.wordnet.getWord(iwd)
                related_synset = related_word.getSynset()
                related_synset_key = "{}{}".format(related_synset.getPOS().getTag(), related_synset.getOffset())
                related_synsets.append(related_synset_key)
        return related_synsets

    def load_semantic_relations(self, synset):
        related_synsets = []
        for iPointer, ISynsetID in synset.getRelatedMap().items():
            for iwd in ISynsetID:
                related_synset = self.wordnet.getSynset(iwd)
                related_synset_key = "{}{}".format(related_synset.getPOS().getTag(), related_synset.getOffset())
                related_synsets.append(related_synset_key)
        return related_synsets

    def load_hypernyms(self, synset):
        return self.load_semantic_relations_by_symbol(synset, "@")

    def load_instance_hypernyms(self, synset):
        return self.load_semantic_relations_by_symbol(synset, "@i")

    def load_hyponyms(self, synset):
        return self.load_semantic_relations_by_symbol(synset, "~")

    def load_instance_hyponyms(self, synset):
        return self.load_semantic_relations_by_symbol(synset, "~i")

    def load_similar_to(self, synset):
        return self.load_semantic_relations_by_symbol(synset, "&")

    def load_antonyms(self, word):
        return self.load_lexical_relations_by_symbol(word, "!")

    def load_semantic_relations_by_symbol(self, synset, relation_symbol):
        related_synset_keys = []
        for IPointer, ISynsetID in synset.getRelatedMap().items():
            if IPointer.getSymbol() == relation_symbol:
                for iwd in ISynsetID:
                    related_synset = self.wordnet.getSynset(iwd)
                    related_synset_key = "" + str(related_synset.getPOS().getTag()) + str(related_synset.getOffset())
                    related_synset_keys.append(related_synset_key)
        return related_synset_keys

    def load_lexical_relations_by_symbol(self, word, relation_symbol):
        related_synset_keys = []
        for IPointer, IWordID in word.getRelatedMap().items():
            if IPointer.getSymbol() == relation_symbol:
                for iwd in IWordID:
                    related_word = self.wordnet.getWord(iwd)
                    related_synset = related_word.getSynset()
                    related_synset_key = "" + str(related_synset.getPOS().getTag()) + str(related_synset.getOffset())
                    related_synset_keys.append(related_synset_key)
        return related_synset_keys
