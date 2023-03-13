import ast
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

from method.neural.NeuralDisambiguator import NeuralDisambiguator
from method.FirstSenseDisambiguator import FirstSenseDisambiguator
from method.MonosemicDisambiguator import MonosemicDisambiguator
from evaluation.WSDEvaluator import WSDEvaluator
from ufsac.ufsac.core.Sentence import Sentence
from ufsac.ufsac.core.Word import Word
from ufsac.ufsac.core.Corpus import Corpus
from ufsac.ufsac.core.Document import Document
from ufsac.ufsac.core.Paragraph import Paragraph
from utils.WordnetUtils import WordnetUtils
from ufsac.common.WordnetHelper import WordnetHelper
from ufsac.common.XMLHelper import XMLHelper
import pandas as pd
import ast
import re
from method.result.MultipleDisambiguationResult import MultipleDisambiguationResult


class NeuralWSDDecode:
    def __init__(self):
        self.neural_disambiguator = None
        self.first_sense_disambiguator = None
        self.mfs_backoff = None

    def read_data_from_csv(self, file):
        return pd.read_csv(file, sep='\t')

    def create_txt_to_wsd_from_corpus(self, file):
        corpus = Corpus()
        doc = Document()
        doc.set_annotation("id", "doc1")
        corpus.add_document(doc)
        par = Paragraph()
        doc.add_paragraph(par)
        sentences_corpus = []
        data = self.read_data_from_csv(file)
        sentences = data["sentence"].tolist()
        # sense_keys = [ast.literal_eval(i) for i in data['sense_keys'].tolist()]
        for i, s in enumerate(sentences):
            sent = Sentence()
            par.add_sentence(sent)
            sent.set_annotation("id", str(i))
            words_in_sent = s.split()
            final_words = []
            for w in words_in_sent:
                if "'" in w:
                    sp = w.split("'")
                    sp[0] += "'"
                    final_words.extend(sp)
                else:
                    final_words.append(w)
            for j, w_f in enumerate(final_words):
                w_proc = Word(XMLHelper.from_valid_xml_entity(w_f.lower()))
                # if sense_keys[i][j]:
                #    sense_keys_word = ";".join(sense_keys[i][j])
                #    w_proc.set_annotation("wn30_key", XMLHelper.from_valid_xml_entity(sense_keys_word), ";")
                sent.add_word(w_proc)
            sentences_corpus.append(sent)
        return sentences_corpus, corpus

    def main(self):
        parser = ArgumentParser(description="Decode the text from eval corpus with a NWSD trained system",
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
        parser.add_argument("--corpus")
        parser.add_argument("--saved_path")

        arguments = parser.parse_args()

        self.data_path = arguments.data_path
        self.weights = arguments.weights
        self.lowercase = ast.literal_eval(arguments.lowercase)
        sense_compression_hypernyms = ast.literal_eval(arguments.sense_compression_hypernyms)
        sense_compression_instance_hypernyms = ast.literal_eval(arguments.sense_compression_instance_hypernyms)
        sense_compression_antonyms = ast.literal_eval(arguments.sense_compression_antonyms)
        sense_compression_file = arguments.sense_compression_file
        self.clear_text = ast.literal_eval(arguments.clear_text)
        self.batch_size = int(arguments.batch_size)
        self.filter_lemma = ast.literal_eval(arguments.filter_lemma)
        self.mfs_backoff = ast.literal_eval(arguments.mfs_backoff)
        corpus_eval = arguments.corpus
        self.outpath = arguments.saved_path

        self.sense_compression_clusters = None
        self.wn = WordnetHelper.wn30()
        if sense_compression_hypernyms or sense_compression_antonyms:
            self.sense_compression_clusters = WordnetUtils.get_sense_compression_clusters(self.wn,
                                                                                     sense_compression_hypernyms,
                                                                                     sense_compression_instance_hypernyms,
                                                                                     sense_compression_antonyms)
        if len(sense_compression_file) != 0:
            self.sense_compression_clusters = WordnetUtils.get_sense_compression_clusters_from_file(sense_compression_file)

        self.first_sense_disambiguator = FirstSenseDisambiguator(self.wn)

        self.neural_disambiguator = NeuralDisambiguator(self.data_path, self.weights, self.clear_text,
                                                        self.batch_size, wn=self.wn)
        self.neural_disambiguator.lowercase_words = self.lowercase
        self.neural_disambiguator.filter_lemma = self.filter_lemma
        self.neural_disambiguator.reduced_output_vocabulary = self.sense_compression_clusters

        self.monosemicDisambiguator = MonosemicDisambiguator(self.wn)
        self.firstSenseDisambiguator = FirstSenseDisambiguator(self.wn)

        self.evaluator = WSDEvaluator()
        self.evaluator.print_results = True

        # decode sentences
        sentences, self.corpus = self.create_txt_to_wsd_from_corpus(corpus_eval)
        self.decode_sentence_batch(sentences)
        print("\n------ Evaluate the score of an ensemble of models ------")
        # self.evaluate()
        
        print("\n------ Evaluate the scores of individual models ------")

        # self.evaluate_mean_scores()
        

    def decode_sentence_batch(self, sentences: list):
        self.neural_disambiguator.disambiguate_dynamic_sentence_batch(sentences, "wsd_test")
        for s in sentences:
            if self.mfs_backoff:
                self.first_sense_disambiguator.disambiguate(s, "wsd_test")
            for word in s.get_words():
                sys.stdout.write(word.get_value().replace("|", "/"))
                if word.has_annotation("wsd_test"):
                    sys.stdout.write("|" + word.get_annotation_value("wsd_test"))
                sys.stdout.write(" ")
            sys.stdout.write('\n')

    def evaluate(self):
        print("Evaluate without backoff \n---------------")
        self.evaluator.evaluate(self.neural_disambiguator, self.corpus, "wn30_key", self.wn)
        print("Evaluate with monosemics \n---------------")
        self.evaluator.evaluate(self.monosemicDisambiguator, self.corpus, "wn30_key", self.wn)
        print("Evaluate with backoff first sense \n---------------")
        self.evaluator.evaluate(self.firstSenseDisambiguator, self.corpus, "wn30_key", self.wn)
        self.evaluator.save_result_to_file(self.corpus.get_documents(), "wsd_test", self.outpath)

    def evaluate_mean_scores(self):
        neural_disambiguators = []
        for weight in self.weights:
            neural_disambiguator = NeuralDisambiguator(self.data_path, [weight], self.clear_text, self.batch_size, wn=self.wn)
            neural_disambiguator.lowercase_words = self.lowercase
            neural_disambiguator.filter_lemma = self.filter_lemma
            neural_disambiguator.reduced_output_vocabulary = self.sense_compression_clusters
            neural_disambiguators.append(neural_disambiguator)
        
        results_backoff_zero = MultipleDisambiguationResult()
        results_backoff_monosemics = MultipleDisambiguationResult()
        results_backoff_first_sense = MultipleDisambiguationResult()

        for i in range(len(self.weights)):
            neural_disambiguator = neural_disambiguators[i]
            print("" + str(i) + " : Evaluate without backoff \n---------------")
            result_backoff_zero = self.evaluator.evaluate(neural_disambiguator, self.corpus, "wn30_key", self.wn)
            print("" + str(i) + " : Evaluate with monosemics \n---------------")
            result_backoff_monosemics = self.evaluator.evaluate(self.monosemicDisambiguator, self.corpus, "wn30_key", self.wn)
            print("" + str(i) + " : Evaluate with backoff first sense \n---------------")
            result_backoff_first_sense = self.evaluator.evaluate(self.firstSenseDisambiguator, self.corpus, "wn30_key", self.wn)
            results_backoff_zero.add_disambiguation_result(result_backoff_zero)
            results_backoff_monosemics.add_disambiguation_result(result_backoff_monosemics)
            results_backoff_first_sense.add_disambiguation_result(result_backoff_first_sense)

        print("\nMean of scores without backoff: " + str(results_backoff_zero.score_mean()))
        print("Standard deviation without backoff: " + str(results_backoff_zero.score_standard_deviation()))
        print("Mean of scores with monosemics: " + str(results_backoff_monosemics.score_mean()))
        print("Standard deviation with monosemics: " + str(results_backoff_monosemics.score_standard_deviation()))
        print("Mean of scores with backoff first sense: " + str(results_backoff_first_sense.score_mean()))
        print("Standard deviation with backoff first sense: " + str(results_backoff_first_sense.score_standard_deviation()))


NeuralWSDDecode().main()

