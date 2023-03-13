import re
import json
from ufsac.ufsac.core.Sentence import Sentence
from ufsac.ufsac.core.Word import Word
from ufsac.common.POSConverter import POSConverter
from predicter import Predicter


class NeuralDisambiguator:
    def __init__(self, neural_path, weights_paths, clear_text=False, batch_size=0, translate=False,
                 beam_size=1, extra_lemma=False, wn=None, hf_model=None):
        self.unknown_token = "<unk>"
        self.input_features = 0
        self.input_annotation_names = []
        self.input_clear_text = []
        self.input_vocabulary = []
        self.output_features = 0
        self.sense_feature_index = 0
        self.output_translations = 0
        self.output_annotation_names = []
        self.reversed_output_annotation_names = {}
        self.output_vocabulary = []
        self.reversed_output_vocabulary = []
        self.reversed_output_translation_vocabulary = []
        self.clear_text = clear_text
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.translate = translate
        self.extra_lemma = extra_lemma
        self.disambiguate_choice = True
        self.lowercase_words = True
        self.filter_lemma = True
        self.reduced_output_vocabulary = None
        self.neural_path = neural_path
        self.weights_paths = weights_paths
        self.read_config_file(neural_path)
        self.init_input_vocabulary(neural_path)
        self.init_output_vocabulary(neural_path)
        self.init_translation_output_vocabulary(neural_path)
        self.to_predict = ''
        self.hf_model = hf_model
        self.wn = wn

    def get_input_features(self):
        return self.input_features

    def get_input_annotation_names(self):
        return self.input_annotation_names

    def run_predict(self, to_predict: str):
        predicter = Predicter()
        predicter.training_root_path = self.neural_path
        predicter.ensemble_weights_path = self.weights_paths
        predicter.clear_text = True if self.clear_text else False
        predicter.batch_size = self.batch_size
        predicter.disambiguate = True if self.disambiguate_choice else False
        predicter.beam_size = self.beam_size
        predicter.output_all_features = True if self.extra_lemma else False
        return predicter.predict(to_predict, self.hf_model)

    def read_config_file(self, neural_path):
        f = open(str(neural_path) + "/config.json", 'r')
        config = json.load(f)
        self.input_features = config["input_features"]
        self.input_annotation_names = config["input_annotation_name"]
        self.input_clear_text = config["input_clear_text"]
        self.output_features = config["output_features"]
        self.output_annotation_names = config["output_annotation_name"]
        self.output_translations = config["output_translation_features"]
        self.sense_feature_index = 0

    def init_input_vocabulary(self, neural_path):
        self.input_vocabulary = []
        for i in range(self.input_features):
            self.input_vocabulary.append(self.init_vocabulary(neural_path + "/input_vocabulary" + str(i)))

    def init_output_vocabulary(self, neural_path):
        self.output_vocabulary = []
        self.reversed_output_vocabulary = []
        self.reversed_output_annotation_names = {}
        for i in range(self.output_features):
            vocabulary = self.init_vocabulary(neural_path + "/output_vocabulary" + str(i))
            reversed_vocabulary = {}
            for key in vocabulary.keys():
                reversed_vocabulary[vocabulary[key]] = key
            self.output_vocabulary.append(vocabulary)
            self.reversed_output_vocabulary.append(reversed_vocabulary)
            self.reversed_output_annotation_names[self.output_annotation_names[i]] = i

    def init_translation_output_vocabulary(self, neural_path):
        self.reversed_output_translation_vocabulary = []
        for i in range(self.output_translations):
            vocabulary = self.init_vocabulary(neural_path + "/output_vocabulary" + str(i) + "_vocabulary0")
            reversed_vocabulary = {}
            for key in vocabulary.keys():
                reversed_vocabulary[vocabulary[key]] = key
            self.reversed_output_translation_vocabulary.append(reversed_vocabulary)

    @staticmethod
    def init_vocabulary(file_path):
        ret = {}
        vocab_as_list = []
        reader = open(file_path, 'r').readlines()
        for line in reader:
            line_split = line.split(" ")
            if len(line_split) == 1:
                vocab_as_list.append(line_split[0])
            else:
                vocab_as_list.append(line_split[1])
        for i in range(len(vocab_as_list)):
            ret[vocab_as_list[i].rstrip()] = i
        return ret

    def disambiguate(self, corpus, new_sense_tags=None):
        sentences = corpus.get_sentences()
        self.disambiguate_dynamic_sentence_batch(sentences, new_sense_tags)

    def disambiguate_dynamic_sentence_batch(self, original_sentences, new_sense_tags):
        sentences = original_sentences.copy()
        self.disambiguate_fixed_sentence_batch(sentences, new_sense_tags)

    def disambiguate_no_catch(self, sentences, sense_tag):
        self.write_predict_input(sentences)
        self.read_predict_output(sentences, sense_tag)

    def disambiguate_fixed_sentence_batch(self, sentences, sense_tag):
        try:
            self.disambiguate_no_catch(sentences, sense_tag)
        except Exception as e:
            raise RuntimeError(e)

    def write_predict_input(self, sentences):
        for sentence in sentences:
            words = sentence.get_words()
            self.write_predict_input_sample_x(words)
            self.write_predict_input_sample_z(words)

    def write_predict_input_sample_x(self, words):
        for w in words:
            if self.lowercase_words:
                w.set_value(w.get_value().lower())
            feature_values = []
            for i in range(self.input_features):
                feature_value = w.get_annotation_value(self.input_annotation_names[i])
                if self.input_clear_text[i] or self.clear_text:
                    feature_value = feature_value.replace("/", "<slash>")
                else:
                    feature_vocabulary = self.input_vocabulary[i]
                    if not feature_value or feature_value not in feature_vocabulary:
                        feature_value = self.unknown_token
                    feature_value = str(feature_vocabulary[feature_value])
                feature_values.append(feature_value)
            feature_val_str = "/".join(feature_values) + " "
            self.to_predict += feature_val_str
        self.to_predict += "\n"

    def write_predict_input_sample_z(self, words):
        if self.output_features <= 0:
            return
        if self.extra_lemma:
            return
        for word in words:
            if self.filter_lemma:
                possible_sense_keys = []
                if word.has_annotation("lemma") and word.has_annotation("pos"):
                    pos = POSConverter.to_wn_pos(word.get_annotation_value("pos"))
                    lemmas = word.get_annotation_values("lemma", ";")
                    for lemma in lemmas:
                        word_key = lemma + "%" + pos
                        if not self.wn.is_word_key_exists(word_key):
                            continue
                        possible_sense_keys.extend(self.wn.get_sense_key_list_from_word_key(word_key))
                if len(possible_sense_keys) != 0:
                    possible_sense_key_indices = []
                    for possible_sense_key in possible_sense_keys:
                        possible_synset_key = self.wn.get_synset_key_from_sense_key(possible_sense_key)
                        if self.reduced_output_vocabulary is not None:
                            possible_synset_key = self.reduced_output_vocabulary.get(possible_synset_key,
                                                                                     possible_synset_key)
                        if possible_synset_key in self.output_vocabulary[self.sense_feature_index]:
                            possible_sense_key_indices.append(
                                str(self.output_vocabulary[self.sense_feature_index][possible_synset_key]))
                    if len(possible_sense_key_indices) == 0:
                        self.to_predict += "0 "
                    else:
                        pos = ";".join(possible_sense_key_indices) + " "
                        self.to_predict += pos
                else:
                    self.to_predict += "0 "
            else:
                self.to_predict += "-1 "
        self.to_predict += "\n"

    def read_predict_output(self, sentences, sense_tag):
        lines = self.run_predict(self.to_predict).splitlines()
        self.to_predict = ''
        translations = []
        for i in range(len(sentences)):
            line = lines[i]
            sentence = sentences[i]
            if self.output_features > 0:
                words = sentence.get_words()
                if self.extra_lemma:
                    output = NeuralDisambiguator.parse_predict_output_extra_lemma(line)
                    self.propagate_predict_output_extra_lemma(words, output, sense_tag)
                else:
                    output = NeuralDisambiguator.parse_predict_output(line)
                    self.propagate_predict_output(words, output, sense_tag)
            if self.output_translations > 0:
                if len(line) == 0:
                    line = "0"
                output = line.split(" ")
                translations.append(self.process_translation_output(output))
        return translations

    @staticmethod
    def parse_predict_output(line):
        line_split = list(filter(None, re.split(r"\s+", line)))
        return [int(i) for i in line_split]

    @staticmethod
    def parse_predict_output_extra_lemma(line):
        line_split = list(filter(None, re.split(r"\s+", line)))
        return [[int(j) for j in i.split("/")] for i in line_split]

    def propagate_predict_output(self, words, output, sense_tag):
        print("Predict output ids : ", str(output))
        wordkeys = []
        sense_keys = []
        lemma_sense_keys_all = []
        possible_sense_keys_all = []
        possible_synset_keys_all = []
        possible_synset_keys_all_end = []
        for i in range(len(output)):
            word = words[i]
            if word.has_annotation(sense_tag):
                continue
            if self.filter_lemma:
                if not word.has_annotation("lemma"):
                    wordkeys.append("None")
                    continue
                if not word.has_annotation("pos"):
                    wordkeys.append("None")
                    continue
                word_output = output[i]
                pos = POSConverter.to_wn_pos(word.get_annotation_value("pos"))
                lemmas = word.get_annotation_values("lemma", ";")
                for lemma in lemmas:
                    word_key = lemma + "%" + pos
                    wordkeys.append(word_key)
                    if not self.wn.is_word_key_exists(word_key):
                        possible_synset_keys_all_end.append([])
                        lemma_sense_keys_all.append([])
                        possible_sense_keys_all.append([])
                        continue
                    lemma_sense_keys = self.wn.get_sense_key_list_from_word_key(word_key)
                    lemma_sense_keys_all.append(lemma_sense_keys)
                    sense_key_add = []
                    for possible_sense_key in lemma_sense_keys:
                        possible_synset_key = self.wn.get_synset_key_from_sense_key(possible_sense_key)
                        possible_synset_keys_all.append(possible_synset_key)
                        if self.reduced_output_vocabulary is not None:
                            possible_synset_key = self.reduced_output_vocabulary.get(possible_synset_key,
                                                                                     possible_synset_key)
                        if self.reversed_output_vocabulary[self.sense_feature_index].get(
                                word_output) == possible_synset_key:
                            word.set_annotation(sense_tag, possible_sense_key)
                            sense_key_add.append(True)
                        else:
                            sense_key_add.append(False)                                                  
                    possible_sense_keys_all.append(sense_key_add)
                    possible_synset_keys_all_end.append(possible_synset_keys_all)
                                                                    
            else:
                word_output = output[i]
                possible_synset_key = self.reversed_output_vocabulary[self.sense_feature_index].get(word_output)
                possible_synset_keys_all_end.append(possible_synset_key)
                possible_sense_key = self.wn.get_sense_key_list_from_synset_key(possible_synset_key)[0]
                possible_sense_keys_all.append(possible_sense_key)
                word.set_annotation(sense_tag, possible_sense_key)
        # print(str(wordkeys) + '\n'  + str(lemma_sense_keys_all) + '\n' + str(possible_synset_keys_all_end) +  '\n' + str(possible_sense_keys_all) + '\n')      

    def propagate_predict_output_extra_lemma(self, words, output, sense_tag):
        for i in range(len(output)):
            word = words[i]
            if word.has_annotation(sense_tag):
                continue
            if not word.has_annotation("lemma"):
                continue
            if not word.has_annotation("pos"):
                continue
            word_output = output[i]
            pos = POSConverter.to_wn_pos(word.get_annotation_value("pos"))
            lemmas = word.get_annotation_values("lemma", ";")
            lemma = lemmas[0]
            word_key = lemma + "%" + pos
            if self.wn.is_word_key_exists(word_key):
                continue
            if word_key in self.reversed_output_annotation_names.keys():
                continue
            extra_lemma_feature_index = self.reversed_output_annotation_names[word_key]
            lemma_sense_keys = self.wn.get_sense_key_list_from_word_key(word_key)
            for possible_sense_key in lemma_sense_keys:
                possible_synset_key = self.wn.get_synset_key_from_sense_key(possible_sense_key)
                if self.reduced_output_vocabulary is not None:
                    if possible_synset_key in self.reduced_output_vocabulary.keys():
                        possible_synset_key = self.reduced_output_vocabulary[possible_synset_key]
                    else:
                        possible_synset_key = possible_synset_key
                if self.reversed_output_vocabulary[extra_lemma_feature_index][
                    word_output[extra_lemma_feature_index]] == possible_synset_key:
                    word.set_annotation(sense_tag, possible_sense_key)

    @staticmethod
    def process_translation_output(output):
        translation = Sentence()
        for wordValue in output:
            Word(wordValue, translation)
        return translation

    def disambiguate_and_translate_dynamic_sentence_batch(self, original_sentences, new_sense_tags=None,
                                                          confidence_tag=None):
        if (new_sense_tags and confidence_tag) is None:
            return self.disambiguate_and_translate_dynamic_sentence_batch(original_sentences, "", "")
        else:
            ret = []
            sentences = [original_sentences]
            while len(sentences) > self.batch_size:
                sub_sentences = sentences[:self.batch_size]
                ret.append(self.disambiguate_and_translate_fixed_sentence_batch(sub_sentences, new_sense_tags))
            if len(sentences) != 0:
                padding_size = self.batch_size - len(sentences)
                for i in range(padding_size):
                    sentences.append(Sentence("<pad>"))
                translated_sentences = self.disambiguate_and_translate_fixed_sentence_batch(sentences, new_sense_tags)
                translated_sentences = translated_sentences[0:len(original_sentences)]
                ret.append(translated_sentences)
            return ret

    def disambiguate_and_translate_fixed_sentence_batch(self, sentences, sense_tag):
        try:
            return self.disambiguate_and_translate_no_catch(sentences, sense_tag)
        except Exception as e:
            raise RuntimeError(e)

    def disambiguate_and_translate_no_catch(self, sentences, sense_tag):
        self.write_predict_input(sentences)
        return self.read_predict_output(sentences, sense_tag)
