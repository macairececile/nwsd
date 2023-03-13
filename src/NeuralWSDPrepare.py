import argparse
import ast

from method.neural.NeuralDataPreparator import NeuralDataPreparator
from utils.WordnetUtils import WordnetUtils
from ufsac.common.WordnetHelper import WordnetHelper


class NeuralWSDPrepare:
    def __init__(self):
        self.wn = None

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path")
        parser.add_argument("--train", nargs='+')
        parser.add_argument("--dev", nargs='+', default=[])
        parser.add_argument("--dev_from_train", default=0)
        parser.add_argument("--corpus_format", default="xml")
        parser.add_argument("--txt_corpus_features", default=[None])
        parser.add_argument("--input_features", default=["surface_form"], nargs='+')
        parser.add_argument("--input_embeddings", default=[None], nargs='+')
        parser.add_argument("--input_vocabulary", default=[None], nargs='+')
        parser.add_argument("--input_vocabulary_limit", default=-1)
        parser.add_argument("--input_clear_text", nargs='+')
        parser.add_argument("--output_features", default=["wn30_key"], nargs='+')
        parser.add_argument("--output_feature_vocabulary_limit", default=-1)
        parser.add_argument("--truncate_line_length", default=80)
        parser.add_argument("--exclude_line_length", default=150)
        parser.add_argument("--line_length_tokenizer", default=None)
        parser.add_argument("--lowercase", default='False')
        parser.add_argument("--filter_lemma", default='True')
        parser.add_argument("--uniform_dash", default='False')
        parser.add_argument("--sense_compression_hypernyms", default='True')
        parser.add_argument("--sense_compression_instance_hypernyms", default='False')
        parser.add_argument("--sense_compression_antonyms", default='False')
        parser.add_argument("--sense_compression_file", default="")
        parser.add_argument("--add_wordkey_from_sensekey", default='False')
        parser.add_argument("--add_monosemics", default='False')
        parser.add_argument("--remove_monosemics", default='False')
        parser.add_argument("--remove_duplicates", default='True')

        args = parser.parse_args()

        data_path = args.data_path
        training_corpus_paths = args.train
        dev_corpus_paths = args.dev
        dev_from_train = int(args.dev_from_train)
        corpus_format = args.corpus_format
        txt_corpus_features = args.txt_corpus_features
        input_features = args.input_features
        input_embeddings = args.input_embeddings
        input_vocabulary = args.input_vocabulary
        input_vocabulary_limit = args.input_vocabulary_limit
        input_clear_text = args.input_clear_text
        output_features = args.output_features
        output_feature_vocabulary_limit = args.output_feature_vocabulary_limit
        max_line_length = int(args.truncate_line_length)
        lowercase = ast.literal_eval(args.lowercase)
        filter_lemma = ast.literal_eval(args.filter_lemma)
        uniform_dash = ast.literal_eval(args.uniform_dash)
        sense_compression_hypernyms = ast.literal_eval(args.sense_compression_hypernyms)
        sense_compression_instance_hypernyms = ast.literal_eval(args.sense_compression_instance_hypernyms)
        sense_compression_antonyms = ast.literal_eval(args.sense_compression_antonyms)
        sense_compression_file = args.sense_compression_file
        add_word_key_from_sense_key = ast.literal_eval(args.add_wordkey_from_sensekey)
        add_monosemics = ast.literal_eval(args.add_monosemics)
        remove_monosemics = ast.literal_eval(args.remove_monosemics)
        remove_duplicate_sentences = ast.literal_eval(args.remove_duplicates)

        sense_compression_clusters = None
        self.wn = WordnetHelper.wn30()
        if sense_compression_hypernyms or sense_compression_antonyms:
            sense_compression_clusters = WordnetUtils.get_sense_compression_clusters(self.wn, sense_compression_hypernyms,
                                                                                     sense_compression_instance_hypernyms,
                                                                                     sense_compression_antonyms)
        if len(sense_compression_file) != 0:
            sense_compression_clusters = WordnetUtils.get_sense_compression_clusters_from_file(sense_compression_file)

        input_embeddings = self.padList(input_embeddings, len(input_features), None)
        input_vocabulary = self.padList(input_vocabulary, len(input_features), None)
        input_clear_text = [bool(i) for i in self.padList(input_clear_text, len(input_features), False)]

        preparator = NeuralDataPreparator()
        preparator.add_word_key_from_sense_key = add_word_key_from_sense_key

        if len(txt_corpus_features) == 1 and txt_corpus_features[0] is None:
            txt_corpus_features = []

        preparator.txt_corpus_features = txt_corpus_features
        preparator.set_output_directory_path(data_path)

        for corpusPath in training_corpus_paths:
            preparator.add_training_corpus(corpusPath)

        for corpusPath in dev_corpus_paths:
            preparator.add_development_corpus(corpusPath)

        for i in range(len(input_features)):
            input_feature_annotation_name = input_features[i]
            input_feature_embeddings = None if input_embeddings[i] is None else input_embeddings[i]
            input_feature_vocabulary = None if input_vocabulary[i] is None else input_vocabulary[i]
            preparator.add_input_feature(input_feature_annotation_name, input_feature_embeddings,
                                         input_feature_vocabulary)

        if len(output_features) == 1 and output_features[0] is None:
            output_features = []

        for i in range(len(output_features)):
            preparator.add_output_feature(output_features[i], None)

        preparator.set_corpus_format(corpus_format)
        preparator.set_input_vocabulary_limit(input_vocabulary_limit)
        preparator.set_input_clear_text(input_clear_text)
        preparator.set_output_feature_vocabulary_limit(output_feature_vocabulary_limit)

        preparator.max_line_length = max_line_length
        preparator.lowercase_words = lowercase
        preparator.filter_lemma = filter_lemma
        preparator.uniform_dash = uniform_dash
        preparator.multi_senses = False
        preparator.remove_all_coarse_grained = True
        preparator.add_monosemics = add_monosemics
        preparator.remove_monosemics = remove_monosemics
        preparator.reduced_output_vocabulary = sense_compression_clusters
        preparator.additional_dev_from_train_size = dev_from_train
        preparator.remove_duplicate_sentences = remove_duplicate_sentences

        preparator.prepare_training_file()

    @staticmethod
    def padList(l, pad_size, pad_value):
        new_list = l
        while len(new_list) < pad_size:
            new_list.append(pad_value)
        return new_list


NeuralWSDPrepare().main()
