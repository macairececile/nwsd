import os
import json
import random
import xml.etree.ElementTree as ET
from embeddings.TextualModelLoader import TextualModelLoader
from ufsac.ufsac.core.Sentence import Sentence
from ufsac.ufsac.core.Word import Word
from utils.WordnetUtils import WordnetUtils
from ufsac.common.POSConverter import POSConverter
from ufsac.common.WordnetHelper import WordnetHelper
from ufsac.common.XMLHelper import XMLHelper
import statistics


class NeuralDataPreparator:
    padding_token = "<pad>"
    unknown_token = "<unk>"
    beginning_of_sentence_token = "<bos>"
    end_of_sentence_token = "<eos>"
    skip_token = "<skip>"
    input_vocabulary_filename = "/input_vocabulary"
    output_vocabulary_filename = "/output_vocabulary"
    output_translation_vocabulary_filename_1 = "/output_translation"
    output_translation_vocabulary_filename_2 = "_vocabulary"
    train_filename = "/train"
    dev_filename = "/dev"
    config_filename = "/config.json"

    wn = WordnetHelper.wn30()
    sense_tag = "wn" + str(wn.get_version()) + "_key"
    input_features = 0
    txt_corpus_features = []
    input_annotation_name = []
    input_embeddings_path = []
    input_vocabulary_path = []
    input_vocabulary = []
    output_features = 0
    output_annotation_name = []
    output_fixed_vocabulary_path = []
    output_vocabulary = []
    output_translations = 0
    output_translation_name = []
    output_translation_features = 0
    output_translation_annotation_name = []
    output_translation_fixed_vocabulary_path = []
    output_translation_vocabulary = []
    output_directory_path = "data/neural/wsd/"
    original_train_paths = []
    original_dev_paths = []
    output_feature_sense_index = -1
    corpus_format = ""
    input_vocabulary_limit = 0
    input_clear_text = []
    output_feature_vocabulary_limit = 0
    output_translation_vocabulary_limit = 0
    output_translation_clear_text = False
    share_translation_vocabulary = False
    extra_word_keys = None
    max_line_length = 80
    lowercase_words = True
    filter_lemma = True
    add_word_key_from_sense_key = False
    uniform_dash = False
    multi_senses = False
    remove_all_coarse_grained = True
    remove_monosemics = False
    add_monosemics = False
    remove_duplicate_sentences = True
    reduced_output_vocabulary = {}
    additional_dev_from_train_size = 0

    def __init__(self):
        self.vocabulary_frequencies = None

    @staticmethod
    def read_sentences_from_xml_file(file, all_sentences):
        tree = ET.parse(file)
        for s in tree.findall('.//sentence'):
            words = s.findall('word')
            sentence = Sentence()
            for w in words:
                attributes = w.attrib
                w = ''
                for k, v in attributes.items():
                    if k == 'surface_form':
                        w = Word(XMLHelper.from_valid_xml_entity(v))
                    else:
                        w.set_annotation(XMLHelper.from_valid_xml_entity(k), XMLHelper.from_valid_xml_entity(v))
                sentence.add_word(w)
            all_sentences.append(sentence)

    def set_output_directory_path(self, path):
        self.output_directory_path = path

    def add_training_corpus(self, corpus_path):
        self.original_train_paths.append(corpus_path)

    def add_development_corpus(self, corpus_path):
        self.original_dev_paths.append(corpus_path)

    def add_input_feature(self, annotation_name: str, embeddings_path: str, vocabulary_path: str):
        self.input_features += 1
        self.input_annotation_name.append(annotation_name)
        self.input_embeddings_path.append(embeddings_path)
        self.input_vocabulary_path.append(vocabulary_path)

    def add_output_feature(self, annotation_name: str, vocabulary_path):
        self.output_features += 1
        self.output_annotation_name.append(annotation_name)
        self.output_fixed_vocabulary_path.append(vocabulary_path)
        if annotation_name == self.sense_tag:
            self.output_feature_sense_index = self.output_features - 1

    def add_output_translation(self, translation_name: str, translation_annotation_name: list, vocabulary_path: str):
        self.output_translations += 1
        self.output_translation_name.append(translation_name)
        self.output_translation_fixed_vocabulary_path.append(vocabulary_path)
        self.output_translation_features = len(translation_annotation_name)
        self.output_translation_annotation_name = translation_annotation_name

    def set_corpus_format(self, corpus_format: str):
        self.corpus_format = corpus_format

    def set_input_vocabulary_limit(self, input_vocabulary_limit: int):
        self.input_vocabulary_limit = input_vocabulary_limit

    def set_input_clear_text(self, input_clear_text: list):
        self.input_clear_text = input_clear_text

    def set_output_feature_vocabulary_limit(self, output_feature_vocabulary_limit):
        self.output_feature_vocabulary_limit = output_feature_vocabulary_limit

    def set_output_translation_vocabulary_limit(self, output_translation_vocabulary_limit: int):
        self.output_translation_vocabulary_limit = output_translation_vocabulary_limit

    def set_output_translation_clear_text(self, output_translation_clear_text: bool):
        self.output_translation_clear_text = output_translation_clear_text

    def set_share_translation_vocabulary(self, share_translation_vocabulary: bool):
        self.share_translation_vocabulary = share_translation_vocabulary

    def set_extra_word_keys(self, extra_word_keys: set):
        if extra_word_keys is not None and len(extra_word_keys) == 0:
            self.extra_word_keys = None
        self.extra_word_keys = extra_word_keys

    def prepare_training_file(self):
        if not os.path.exists(self.output_directory_path):
            os.makedirs(self.output_directory_path)
        train_sentences = self.extract_sentences_from_corpora(self.original_train_paths)
        dev_sentences = self.extract_sentences_from_corpora(self.original_dev_paths)
        translation_train_sentences = []  # [[Sentence, Sentence], [Sentence], ...]
        translation_dev_sentences = []

        for i in range(self.output_translations):
            translation_name = self.output_translation_name[i]
            translation_train_paths = [self.get_translation_corpus_name(x, translation_name) for x in
                                       self.original_train_paths]
            translation_dev_paths = [self.get_translation_corpus_name(x, translation_name) for x in
                                     self.original_dev_paths]
            translation_name_train_sentences = self.extract_sentences_from_corpora(translation_train_paths)
            translation_name_dev_sentences = self.extract_sentences_from_corpora(translation_dev_paths)
            assert len(translation_name_train_sentences) == len(train_sentences)
            assert len(translation_name_dev_sentences) == len(dev_sentences)
            translation_train_sentences.append(translation_name_train_sentences)
            translation_dev_sentences.append(translation_name_dev_sentences)

        train_sentences, dev_sentences = self.extract_dev_sentences_parallel(train_sentences,
                                                                             translation_train_sentences,
                                                                             translation_dev_sentences,
                                                                             self.additional_dev_from_train_size)
        
        self.build_extra_word_keys_vocabulary(train_sentences, True)
        self.build_extra_word_keys_vocabulary(dev_sentences, False)

        self.init_input_vocabulary()
        self.init_output_vocabulary()
        self.init_output_translation_vocabulary()

        print("-> Building vocabulary")
        self.build_vocabulary(train_sentences, self.input_annotation_name, self.input_embeddings_path,
                              self.input_vocabulary,
                              True,
                              self.input_vocabulary_limit)
        self.build_vocabulary(train_sentences, self.output_annotation_name, self.output_fixed_vocabulary_path,
                              self.output_vocabulary, False,
                              self.output_feature_vocabulary_limit)
        for i in range(self.output_translations):
            self.build_vocabulary(translation_train_sentences[i], self.output_translation_annotation_name,
                                  self.output_translation_fixed_vocabulary_path, self.output_translation_vocabulary[i],
                                  True,
                                  self.output_translation_vocabulary_limit)
        if self.share_translation_vocabulary and self.output_translations > 0 and self.output_translation_features > 0:
            NeuralDataPreparator.merge_vocabularies(self.input_vocabulary[0], self.output_translation_vocabulary[0][0])
        if self.output_translations == 0:
            train_sentences = NeuralDataPreparator.filter_sentences_without_feature(train_sentences,
                                                                                    self.output_annotation_name,
                                                                                    self.output_vocabulary)
            if self.remove_duplicate_sentences:
                train_sentences = self.remove_duplicates(train_sentences)
        print("-> Remove empty parallel sentences")
        train_sentences = self.remove_empty_parallel_sentences(train_sentences, translation_train_sentences)
        dev_sentences = self.remove_empty_parallel_sentences(dev_sentences, translation_dev_sentences)
        print("-> Write input and output vocabulary")
        for i in range(self.input_features):
            NeuralDataPreparator.write_vocabulary(self.input_vocabulary[i],
                                                  self.output_directory_path + self.input_vocabulary_filename + str(i),
                                                  self.input_clear_text[i])
        for i in range(self.output_features):
            NeuralDataPreparator.write_vocabulary(self.output_vocabulary[i],
                                                  self.output_directory_path + self.output_vocabulary_filename + str(i),
                                                  False)
        for i in range(self.output_translations):
            for j in range(self.output_translation_features):
                NeuralDataPreparator.write_vocabulary(self.output_translation_vocabulary[i][j],
                                                      self.output_directory_path + self.output_translation_vocabulary_filename_1 + str(
                                                          i) + self.output_translation_vocabulary_filename_2 + str(j),
                                                      self.output_translation_clear_text)
        self.write_corpus(train_sentences, translation_train_sentences,
                          self.output_directory_path + self.train_filename)
        self.write_corpus(dev_sentences, translation_dev_sentences, self.output_directory_path + self.dev_filename)
        self.write_config_file(self.output_directory_path + self.config_filename)
        print("Number of sentences for training : ", str(len(train_sentences)))
        print("Number of sentences for validation : ", str(len(dev_sentences)))
        print("Number of words in training data : ", str(sum(len(s.get_words()) for s in train_sentences)))
        print("Number of words in validation data : ", str(sum(len(s.get_words()) for s in dev_sentences)))
        print("Length of vocabulary : ", str(len(self.output_vocabulary[0])))
        num_words_in_sentences_dev = [len(s.get_words()) for s in dev_sentences]
        num_words_in_sentences_train = [len(s.get_words()) for s in train_sentences]
        print("Mean words in train sentences : ", str(statistics.mean(num_words_in_sentences_train)))
        print("Mean words in dev sentences : ", str(statistics.mean(num_words_in_sentences_dev)))
        print("Num words wordsense in train : ", str(sum(1 for s in train_sentences for w in s.get_words() if w.has_annotation("wn30_key"))))
        print("Num words wordsense in dev : ", str(sum(1 for s in dev_sentences for w in s.get_words() if w.has_annotation("wn30_key"))))

    def init_input_vocabulary(self):
        assert self.input_features > 0
        for i in range(self.input_features):
            if self.input_embeddings_path[i] is not None:
                self.input_vocabulary.append(self.load_embeddings(self.input_embeddings_path[i]))
            elif self.input_vocabulary_path[i] is not None:
                self.input_vocabulary.append(self.load_vocabulary(self.input_vocabulary_path[i]))
            else:
                self.input_vocabulary.append(self.create_new_input_vocabulary())

    def init_output_vocabulary(self):
        for i in range(self.output_features):
            if self.output_fixed_vocabulary_path[i] is not None:
                self.output_vocabulary.append(
                    NeuralDataPreparator.read_vocabulary(self.output_fixed_vocabulary_path[i]))
            else:
                self.output_vocabulary.append((self.create_new_output_vocabulary()))

    def init_output_translation_vocabulary(self):
        for i in range(self.output_translations):
            translation_vocabulary = []  # List<Map<String, Integer>>
            assert len(self.output_translation_annotation_name) != 0
            assert len(self.output_translation_annotation_name) == self.output_translation_features
            for j in range(self.output_translation_features):
                if self.output_translation_fixed_vocabulary_path[i] is not None:
                    translation_vocabulary.append(
                        self.load_vocabulary(self.output_translation_fixed_vocabulary_path[i]))
                else:
                    translation_vocabulary.append(self.createNewOutputTranslationVocabulary())
            self.output_translation_vocabulary.append(translation_vocabulary)

    def load_embeddings(self, embeddings_path: str):
        embeddings = TextualModelLoader(False).loadVocabularyOnly(embeddings_path)  # class TextualModelLoader
        vocabulary = self.create_new_input_vocabulary()  # dict with (string, integer)
        i = len(vocabulary)
        for vocab in embeddings.get_vocabulary():
            vocabulary[vocab] = i
            i += 1
        return vocabulary

    def load_vocabulary(self, vocabulary_path: str):
        vocabulary = self.create_new_input_vocabulary()
        i = len(vocabulary)
        in_file = open(vocabulary_path, 'r').readlines()
        for l in in_file:
            vocabulary[l] = i
            i += 1
        return vocabulary

    @staticmethod
    def read_vocabulary(vocabulary_path: str):
        vocabulary = {}
        in_file = open(vocabulary_path, 'r').readlines()
        for l in in_file:
            tokens = l.split(' ')
            vocabulary[tokens[1]] = int(tokens[0])
        return vocabulary

    def create_new_input_vocabulary(self):
        vocabulary = {self.padding_token: 0, self.unknown_token: 1, self.beginning_of_sentence_token: 2,
                      self.end_of_sentence_token: 3}
        return vocabulary

    def create_new_output_vocabulary(self):
        vocabulary = {self.skip_token: 0}
        return vocabulary

    def createNewOutputTranslationVocabulary(self):
        vocabulary = {self.padding_token: 0, self.unknown_token: 1, self.beginning_of_sentence_token: 2,
                      self.end_of_sentence_token: 3}
        return vocabulary

    @staticmethod
    def write_vocabulary(vocabulary: dict, vocabulary_path: str, clear_text: bool):
        with open(vocabulary_path, 'w') as out:
            if not clear_text:
                for vocab in sorted(vocabulary.items(), key=lambda x: x[1]):
                    out.write(f"{vocab[0]}\n")

    def write_config_file(self, config_file_path: str):
        print("-> Write config file")
        config = {"input_features": self.input_features, "input_annotation_name": self.input_annotation_name}
        input_embeddings_path = [os.path.abspath(x) for x in self.input_embeddings_path if x is not None]
        config["input_embeddings_path"] = input_embeddings_path
        input_clear_text = [self.input_clear_text[i] for i in range(self.input_features)]
        config["input_clear_text"] = input_clear_text
        config["output_features"] = self.output_features
        config["output_annotation_name"] = self.output_annotation_name
        config["output_translations"] = self.output_translations
        config["output_translation_name"] = self.output_translation_name
        config["output_translation_features"] = self.output_translation_features
        config["output_translation_annotation_name"] = self.output_translation_annotation_name
        config["output_translation_clear_text"] = self.output_translation_clear_text

        with open(config_file_path, 'w') as f:
            json.dump(config, f)

    def extract_sentences_from_corpora(self, original_corpus_paths: list):
        if self.corpus_format == "xml":
            sentences = NeuralDataPreparator.extract_sentences_from_ufsac_corpora(original_corpus_paths)
        else:
            sentences = self.extract_sentences_from_txt_corpora(original_corpus_paths)
        self.clean_sentences(sentences)
        return sentences

    def extract_sentences_from_txt_corpora(self, original_corpus_paths):
        print(original_corpus_paths)
        all_sentences = []
        if len(self.txt_corpus_features) == 0:
            txt_corpus_features = []
            for i in range(self.output_features):
                txt_corpus_features.append(self.input_annotation_name[i])
            for i in range(self.output_features):
                txt_corpus_features.append(self.output_annotation_name[i])
        for original_corpus_path in original_corpus_paths:
            print("Extracting sentences from corpus ", original_corpus_path)
            with open(original_corpus_path, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if l != "":
                        sentence = Sentence()  # à déterminer
                        words = l.split(' ')
                        for w in words:
                            ufsac_word = Word()
                            word_features = w.split('|')
                            if len(word_features) < 1:
                                print("Warning: empty word in sentence: ", l)
                                word_features = "/"
                            ufsac_word.set_value(word_features[0])
                            for i in range(1, len(self.txt_corpus_features)):
                                if len(word_features) > i:
                                    ufsac_word.set_annotation(self.txt_corpus_features[i], word_features[i])
                            sentence.add_word(ufsac_word)
                        all_sentences.append(sentence)
        return all_sentences

    @staticmethod
    def extract_sentences_from_ufsac_corpora(original_corpus_paths):
        all_sentences = []
        for original_corpus_path in original_corpus_paths:
            print(f"-> Extracting sentences from corpus {original_corpus_path} and cleaning sentences")
            NeuralDataPreparator.read_sentences_from_xml_file(original_corpus_path, all_sentences)
        return all_sentences

    def clean_sentences(self, sentences: list):
        for s in sentences:
            s.limit_sentence_length(self.max_line_length)
            words = s.get_words()
            for w in words:
                if (self.add_monosemics and not w.has_annotation(self.sense_tag) and w.has_annotation(
                        "lemma") and w.has_annotation("pos")):
                    word_key = w.get_annotation_value("lemma") + "%" + POSConverter.to_wn_pos(w.get_annotation_value("pos"))
                    if self.wn.is_word_key_exists(word_key):
                        sense_keys = self.wn.get_sense_key_list_from_word_key(word_key)
                        if len(sense_keys) == 1:
                            w.set_annotation(self.sense_tag, sense_keys[0])
                if w.has_annotation(self.sense_tag):
                    sense_keys = w.get_annotation_values(self.sense_tag, ";")
                    if not w.has_annotation("lemma"):
                        w.set_annotation("lemma", WordnetUtils.extract_lemma_from_sense_key(sense_keys[0]))
                    if not w.has_annotation("pos"):
                        w.set_annotation("pos", WordnetUtils.extract_pos_from_sense_key(sense_keys[0]))
                    word_key = w.get_annotation_value("lemma") + "%" + POSConverter.to_wn_pos(w.get_annotation_value("pos"))
                    if self.add_word_key_from_sense_key:
                        w.set_annotation("word_key", word_key)
                    if self.remove_monosemics and len(self.wn.get_sense_key_list_from_word_key(word_key)) == 1:
                        w.remove_annotation(self.sense_tag)
                        sense_keys = []
                    synset_keys = WordnetUtils.get_unique_synset_keys_from_sense_keys(self.wn, sense_keys)
                    final_synset_keys = []
                    if self.remove_all_coarse_grained and len(synset_keys) > 1:
                        synset_keys = []
                    for synset_key in synset_keys:
                        if self.reduced_output_vocabulary is not None:
                            synset_key = self.reduced_output_vocabulary.get(synset_key, synset_key)
                        final_synset_keys.append(synset_key)
                    if not final_synset_keys:
                        w.remove_annotation(self.sense_tag)
                    else:
                        if not self.multi_senses:
                            final_synset_keys = final_synset_keys[:1]
                        w.set_annotation(self.sense_tag, final_synset_keys, ';')
                if self.lowercase_words:
                    w.set_value(w.get_value().lower())
                if self.uniform_dash:
                    w.set_value(w.get_value().replaceAll("_", "-"))

    def build_extra_word_keys_vocabulary(self, sentences: list, is_train: bool):
        if self.extra_word_keys is None:
            return
        print("Building extra wordkeys vocabulary")
        senses_per_word_key = {}
        for s in sentences:
            words = s.get_words()
            for w in words:
                if w.has_annotation(self.sense_tag):
                    sense_key = w.get_annotation_value(self.sense_tag)
                    for extra_word_key in self.extra_word_keys:
                        is_possible_sense = False
                        for extrasense_key in self.wn.get_sense_key_list_from_word_key(self.extra_word_keys):
                            extra_synset_key = self.wn.get_synset_key_from_sense_key(extrasense_key)
                            if self.reduced_output_vocabulary is not None:
                                if extra_synset_key in self.reduced_output_vocabulary.keys():
                                    extra_synset_key = self.reduced_output_vocabulary[extra_synset_key]
                                else:
                                    extra_synset_key = extra_synset_key
                            if extra_synset_key == sense_key:
                                is_possible_sense = True
                                break
                        if is_possible_sense:
                            w.set_annotation(extra_word_key, sense_key)
                            if is_train:
                                if extra_word_key not in senses_per_word_key.keys():
                                    senses_per_word_key[extra_word_key] = {}
                                senses_per_word_key[extra_word_key] = sense_key
        if is_train:
            for extra_word_key in senses_per_word_key.keys():
                if len(senses_per_word_key[extra_word_key]) > 1:
                    self.add_output_feature(extra_word_key, "")

    def build_vocabulary(self, all_sentences: list, annotation_name: list, fixed_vocabulary_path: list, vocabulary,
                         is_input_vocabulary: bool, vocabulary_limit: int):
        self.vocabulary_frequencies = []
        for i in range(len(annotation_name)):
            self.vocabulary_frequencies.append({})
        for s in all_sentences:
            words = s.get_words()
            for w in words:
                for i in range(len(annotation_name)):
                    if fixed_vocabulary_path[i] is not None:
                        continue
                    if is_input_vocabulary:
                        feature_values = [w.get_annotation_value(annotation_name[i])]
                        if not feature_values[0]:
                            continue
                    else:
                        feature_values = w.get_annotation_values(annotation_name[i], ";")
                        if not feature_values:
                            continue
                    for feature_value in feature_values:
                        feature_frequencies = self.vocabulary_frequencies[i]
                        current_frequency = feature_frequencies.get(feature_value, 0)
                        current_frequency += 1
                        feature_frequencies[feature_value] = current_frequency
        for i in range(len(annotation_name)):
            if fixed_vocabulary_path[i] is not None:
                continue
            feature_frequencies = self.vocabulary_frequencies[i]
            if is_input_vocabulary:
                feature_vocabulary = self.create_new_input_vocabulary()
            else:
                feature_vocabulary = self.create_new_output_vocabulary()
            init_vocabulary_size = len(feature_vocabulary)
            sorted_keys = [k for k, v in sorted(feature_frequencies.items(), key=lambda item: item[1])]
            sorted_keys.reverse()
            if vocabulary_limit <= 0:
                vocabulary_limit = len(sorted_keys)
            else:
                vocabulary_limit = min(vocabulary_limit, len(sorted_keys))
            for j in range(vocabulary_limit):
                feature_vocabulary[sorted_keys[j]] = j + init_vocabulary_size
            vocabulary[i] = feature_vocabulary

    @staticmethod
    def merge_vocabularies(vocabulary1: dict, vocabulary2: dict):
        j = len(vocabulary1)
        for word_in_vocabulary2 in vocabulary2.keys():
            if word_in_vocabulary2 not in vocabulary1.keys():
                vocabulary1[word_in_vocabulary2] = j
                j += 1
        vocabulary2.update(vocabulary1)

    @staticmethod
    def filter_sentences_without_feature(all_sentences: list, annotation_name: list, vocabulary):
        print("-> Filtering sentences without feature")
        filtered_sentences = []
        for s in all_sentences:
            words = s.get_words()
            sentence_has_output_features = False
            for w in words:
                for i in range(len(annotation_name)):
                    feature_values = w.get_annotation_values(annotation_name[i], ';')
                    if not feature_values:
                        continue
                    feature_vocabulary = vocabulary[i]  # is a dict
                    for feature_value in feature_values:
                        if feature_value in feature_vocabulary:
                            sentence_has_output_features = True
            if sentence_has_output_features:
                filtered_sentences.append(s)
        return filtered_sentences

    def remove_duplicates(self, sentences: list):
        print("-> Removing duplicate sentences")
        real_sentences = {}
        for current_sentence in sentences:
            sentence_as_string = current_sentence.to_string()
            if sentence_as_string not in real_sentences:
                real_sentences[sentence_as_string] = current_sentence
            else:
                real_sentence = real_sentences[sentence_as_string]
                assert len(real_sentence.get_words()) == len(current_sentence.get_words())
                for i in range(len(current_sentence.get_words())):
                    current_sentence_word = current_sentence.get_words()[i]
                    real_sentence_word = real_sentence.get_words()[i]
                    if current_sentence_word.has_annotation(self.sense_tag):
                        current_sense_keys = current_sentence_word.get_annotation_values(self.sense_tag, ";")
                        real_sense_keys = list(real_sentence_word.get_annotation_values(self.sense_tag, ";"))
                        if len(real_sense_keys) == 0:
                            real_sentence_word.set_annotation("lemma", current_sentence_word.get_annotation_value("lemma"))
                            real_sentence_word.set_annotation("pos", current_sentence_word.get_annotation_value("pos"))
                        elif real_sentence_word.get_annotation_value("lemma") != current_sentence_word.get_annotation_value(
                                "lemma") or real_sentence_word.get_annotation_value(
                            "pos") != current_sentence_word.get_annotation_value("pos"):
                            continue
                        for sense in current_sense_keys:
                            if sense not in real_sense_keys:
                                real_sense_keys.append(sense)
                        real_sentence_word.set_annotation(self.sense_tag, real_sense_keys, ';')
        sentences.clear()
        real_true_sentences = list(real_sentences.values())
        real_sentences.clear()
        for s in real_true_sentences:
            for w in s.get_words():
                if w.has_annotation(self.sense_tag):
                    sense_keys = w.get_annotation_values(self.sense_tag, ";")
                    if len(sense_keys) > 1:
                        if self.remove_all_coarse_grained:
                            w.remove_annotation(self.sense_tag)
                        elif not self.multi_senses:
                            w.set_annotation(self.sense_tag, sense_keys[:1], ";")
        return real_true_sentences

    def extract_dev_sentences_parallel(self, train_sentences: list, translated_train_sentences,
                                       translated_dev_sentences, count: int):
        if count <= 0:
            return
        if len(train_sentences) <= count:
            return
        print("-> Extracting dev sentences from train")
        l = [i for i in range(len(train_sentences))]
        random.shuffle(l)
        random_indices = l[:count]
        train_sentences_to_extract = []
        translated_train_sentences_to_extract = []
        for i in range(self.output_translations):
            translated_train_sentences_to_extract.append([])
        for index in random_indices:
            train_sentences_to_extract.append(train_sentences[index])
            for i in range(self.output_translations):
                translated_train_sentences_to_extract[i].append(translated_train_sentences[i][index])
        train_sentencess = [x for x in train_sentences if x not in train_sentences_to_extract]
        dev_sentencess = [x for x in train_sentences if x in train_sentences_to_extract]
        for i in range(self.output_translations):
            translated_train_sentences[i] = [x for x in translated_train_sentences[i] if
                                             x not in translated_train_sentences_to_extract[i]]
            translated_dev_sentences[i] = [x for x in translated_dev_sentences[i] if
                                           x in translated_train_sentences_to_extract[i]]
        random.shuffle(train_sentencess)  # shuffle the sentences
        random.shuffle(dev_sentencess)
        return train_sentencess, dev_sentencess

    def remove_empty_parallel_sentences(self, sentences: list, translated_sentences):
        sentence_indices_to_remove = []
        for i in range(len(sentences)):
            for one_language_translated_sentences in translated_sentences:
                translated_sentence = one_language_translated_sentences[i]
                translated_sentence_words = translated_sentence.get_words()
                empty = True
                for w in translated_sentence_words:
                    if not empty:
                        break
                    for annotation in self.output_translation_annotation_name:
                        if w.has_annotation(annotation):
                            empty = False
                            break
                if empty:
                    sentence_indices_to_remove.append(i)
                    break
        sentences = [s for i, s in enumerate(sentences) if i not in sentence_indices_to_remove]
        for t_sentences in translated_sentences:
            t_sentences[:] = [sentence for sentence in t_sentences if sentence not in sentence_indices_to_remove]
        return sentences

    def get_translation_corpus_name(self, original_corpus_name: str, translation_name: str):
        if self.corpus_format == 'xml':
            return original_corpus_name[:original_corpus_name.rindex(".xml")] + "." + translation_name + ".xml"
        else:
            return original_corpus_name[:original_corpus_name.rindex(".")] + "." + translation_name

    def write_corpus(self, sentences: list, translated_sentences, corpus_path: str):
        print("-> Writing corpus ", corpus_path)
        writer = open(corpus_path, 'w')
        for si in range(len(sentences)):
            s = sentences[si]
            words = s.get_words()
            for w in words:
                feature_values = []
                for i in range(self.input_features):
                    feature_value = w.get_annotation_value(self.input_annotation_name[i])
                    feature_vocabulary = self.input_vocabulary[i]
                    if not feature_value or (feature_value not in feature_vocabulary and (
                            self.input_vocabulary_limit <= 0 and not self.input_clear_text[i])):
                        feature_value = self.unknown_token
                    if self.input_clear_text[i]:
                        feature_value = feature_value.replace("/", "<slash>")
                    else:
                        feature_value = str(feature_vocabulary[feature_value])
                    feature_values.append(feature_value)
                writer.write('/'.join(feature_values) + ' ')
            writer.write('\n')
            if self.output_features > 0:
                for w in words:
                    feature_values = []
                    for i in range(self.output_features):
                        feature_vocabulary = self.output_vocabulary[i]
                        this_feature_values = w.get_annotation_values(str(self.output_annotation_name[i]), ";")
                        this_feature_values = [x for x in this_feature_values if x in feature_vocabulary]
                        if not this_feature_values:
                            this_feature_values = [self.skip_token]
                        this_feature_values = [str(feature_vocabulary[x]) for x in this_feature_values]
                        feature_values.append(";".join(this_feature_values))
                    writer.write('/'.join(feature_values) + " ")
                writer.write('\n')
                for w in words:
                    for i in range(self.output_features):
                        if i > 0:
                            writer.write("/")
                        feature_tag = self.output_annotation_name[i]
                        feature_vocabulary = self.output_vocabulary[i]
                        if w.has_annotation(feature_tag) and any(
                                fv in feature_vocabulary for fv in w.get_annotation_values(feature_tag, ";")):
                            if self.output_feature_sense_index == i:
                                restricted_senses = []
                                word_key = w.get_annotation_value("lemma") + "%" + POSConverter.to_wn_pos(
                                    w.get_annotation_value("pos"))
                                for sense_key in self.wn.get_sense_key_list_from_word_key(word_key):
                                    synset_key = self.wn.get_synset_key_from_sense_key(sense_key)
                                    if self.reduced_output_vocabulary is not None:
                                        synset_key = self.reduced_output_vocabulary.get(synset_key, synset_key)
                                    if synset_key in feature_vocabulary:
                                        restricted_senses.append(str(feature_vocabulary[synset_key]))
                                writer.write(";".join(restricted_senses))
                            else:
                                writer.write("-1")
                        else:
                            writer.write("0")
                    writer.write(" ")
                writer.write('\n')
            for ti in range(self.output_translations):
                translated_sentence = translated_sentences[ti][si]
                words = translated_sentence.get_words()
                for w in words:
                    feature_values = []
                    for i in range(self.output_translation_features):
                        feature_value = w.get_annotation_value(self.output_translation_annotation_name[i])
                        feature_vocabulary = self.output_translation_vocabulary[ti][i]
                        if len(feature_value) == 0 or feature_value not in feature_vocabulary:
                            feature_value = self.unknown_token
                        if self.output_translation_clear_text:
                            feature_value = feature_value.replace("/", "<slash>")
                        else:
                            feature_value = str(feature_vocabulary[feature_value])
                        feature_values.append(feature_value)
                    writer.write("/".join(feature_values) + " ")
                writer.write("\n")
