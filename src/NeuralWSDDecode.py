import ast
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

from method.FirstSenseDisambiguator import FirstSenseDisambiguator
from method.neural.NeuralDisambiguator import NeuralDisambiguator
from ufsac.ufsac.core.Sentence import Sentence
from utils.WordnetUtils import WordnetUtils
from ufsac.common.WordnetHelper import WordnetHelper
from ufsac.ufsac.utils.CorpusPOSTaggerAndLemmatizer import CorpusPOSTaggerAndLemmatizer


class NeuralWSDDecode:
    def __init__(self):
        self.neural_disambiguator = None
        self.first_sense_disambiguator = None
        self.mfs_backoff = None

    def main(self):
        parser = ArgumentParser(description="Decode a text with the NWSD trained system",
                                formatter_class=RawTextHelpFormatter)
        parser.add_argument("--data_path")
        parser.add_argument("--weights", nargs='+')
        parser.add_argument("--lowercase", default='False')
        parser.add_argument("--sense_compression_hypernyms", default='False')
        parser.add_argument("--sense_compression_instance_hypernyms", default='False')
        parser.add_argument("--sense_compression_antonyms", default='False')
        parser.add_argument("--sense_compression_file", default="")
        parser.add_argument("--clear_text", default="True")
        parser.add_argument("--batch_size", default=1)
        parser.add_argument("--truncate_max_length", default=150)
        parser.add_argument("--filter_lemma", default="True")
        parser.add_argument("--mfs_backoff", default="True")

        arguments = parser.parse_args()

        data_path = arguments.data_path
        weights = arguments.weights
        lowercase = ast.literal_eval(arguments.lowercase)
        sense_compression_hypernyms = ast.literal_eval(arguments.sense_compression_hypernyms)
        sense_compression_instance_hypernyms = ast.literal_eval(arguments.sense_compression_instance_hypernyms)
        sense_compression_antonyms = ast.literal_eval(arguments.sense_compression_antonyms)
        sense_compression_file = arguments.sense_compression_file
        clear_text = ast.literal_eval(arguments.clear_text)
        batch_size = int(arguments.batch_size)
        filter_lemma = ast.literal_eval(arguments.filter_lemma)
        truncate_max_length = int(arguments.truncate_max_length)
        self.mfs_backoff = ast.literal_eval(arguments.mfs_backoff)

        sense_compression_clusters = None
        wn = WordnetHelper.wn30()
        if sense_compression_hypernyms or sense_compression_antonyms:
            sense_compression_clusters = WordnetUtils.get_sense_compression_clusters(wn,
                                                                                     sense_compression_hypernyms,
                                                                                     sense_compression_instance_hypernyms,
                                                                                     sense_compression_antonyms)
        if len(sense_compression_file) != 0:
            sense_compression_clusters = WordnetUtils.get_sense_compression_clusters_from_file(sense_compression_file)

        tagger = CorpusPOSTaggerAndLemmatizer()
        self.first_sense_disambiguator = FirstSenseDisambiguator(wn)

        self.neural_disambiguator = NeuralDisambiguator(data_path, weights, clear_text,
                                                        batch_size, wn=wn)
        self.neural_disambiguator.lowercase_words = lowercase
        self.neural_disambiguator.filter_lemma = filter_lemma
        self.neural_disambiguator.reduced_output_vocabulary = sense_compression_clusters

        sentences = []
        content = sys.stdin.read().split('\n')
        for line in content:
            if line != "":
                sentence = Sentence(line)
                if len(sentence.get_words()) > truncate_max_length:
                    to_remove = list(sentence.get_words())[truncate_max_length:]
                    for word in to_remove:
                        sentence.remove_word(word)
                if filter_lemma:
                    tagger.tag(sentence.get_words())
                sentences.append(sentence)
        self.decode_sentence_batch(sentences)

    def decode_sentence_batch(self, sentences: list):
        self.neural_disambiguator.disambiguate_dynamic_sentence_batch(sentences, "wsd")
        for s in sentences:
            if self.mfs_backoff:
                self.first_sense_disambiguator.disambiguate(s, "wsd")
            for word in s.get_words():
                sys.stdout.write(word.get_value().replace("|", "/"))
                if word.has_annotation("wsd"):
                    sys.stdout.write("|" + word.get_annotation_value("wsd"))
                sys.stdout.write(" ")
            sys.stdout.write('\n')


NeuralWSDDecode().main()
