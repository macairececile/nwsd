import argparse
import ast

from evaluation.WSDEvaluator import WSDEvaluator
from method.FirstSenseDisambiguator import FirstSenseDisambiguator
from method.MonosemicDisambiguator import MonosemicDisambiguator
from method.neural.NeuralDisambiguator import NeuralDisambiguator
from method.result.MultipleDisambiguationResult import MultipleDisambiguationResult
from utils.WordnetUtils import WordnetUtils
from ufsac.common.WordnetHelper import WordnetHelper
from ufsac.ufsac.core.Corpus import Corpus


class NeuralWSDEvaluate:
    def __init__(self):
        self.wn = None
        self.evaluator = None
        self.firstSenseDisambiguator = None
        self.monosemicDisambiguator = None
        self.sense_compression_clusters = None
        self.sense_compression_clusters = None
        self.filter_lemma = None
        self.batch_size = None
        self.clear_text = None
        self.test_corpus_paths = None
        self.lowercase = None
        self.weights = None
        self.data_path = None

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path")
        parser.add_argument("--weights", nargs='+')
        parser.add_argument("--corpus", nargs='+')
        parser.add_argument("--lowercase", default='False')
        parser.add_argument("--sense_compression_hypernyms", default='True')
        parser.add_argument("--sense_compression_instance_hypernyms", default='False')
        parser.add_argument("--sense_compression_antonyms", default='False')
        parser.add_argument("--sense_compression_file", default="")
        parser.add_argument("--filter_lemma", default='True')
        parser.add_argument("--clear_text", default="False")
        parser.add_argument("--batch_size", default=1)

        arguments = parser.parse_args()

        self.data_path = arguments.data_path
        self.weights = arguments.weights
        self.lowercase = ast.literal_eval(arguments.lowercase)
        self.test_corpus_paths = arguments.corpus
        sense_compression_hypernyms = ast.literal_eval(arguments.sense_compression_hypernyms)
        sense_compression_instance_hypernyms = ast.literal_eval(arguments.sense_compression_instance_hypernyms)
        sense_compression_antonyms = ast.literal_eval(arguments.sense_compression_antonyms)
        sense_compression_file = arguments.sense_compression_file
        self.clear_text = ast.literal_eval(arguments.clear_text)
        self.batch_size = int(arguments.batch_size)
        self.filter_lemma = ast.literal_eval(arguments.filter_lemma)

        self.wn = WordnetHelper.wn30()
        if sense_compression_hypernyms or sense_compression_antonyms:
            self.sense_compression_clusters = WordnetUtils.get_sense_compression_clusters(self.wn,
                                                                                          sense_compression_hypernyms,
                                                                                          sense_compression_instance_hypernyms,
                                                                                          sense_compression_antonyms)
        if len(sense_compression_file) != 0:
            self.sense_compression_clusters = WordnetUtils.get_sense_compression_clusters_from_file(sense_compression_file)

        self.monosemicDisambiguator = MonosemicDisambiguator(self.wn)
        self.firstSenseDisambiguator = FirstSenseDisambiguator(self.wn)

        self.evaluator = WSDEvaluator()

        print("\n------ Evaluate the score of an ensemble of models ------")

        self.evaluate_ensemble()

        print("\n------ Evaluate the scores of individual models ------")

        self.evaluate_mean_scores()
        

    def evaluate_ensemble(self):
        neural_disambiguator = NeuralDisambiguator(self.data_path, self.weights, self.clear_text,
                                                   self.batch_size, wn=self.wn)
        neural_disambiguator.lowercase_words = self.lowercase
        neural_disambiguator.filter_lemma = self.filter_lemma
        neural_disambiguator.reduced_output_vocabulary = self.sense_compression_clusters

        for test_corpus_path in self.test_corpus_paths:
            print("Evaluate on corpus " + test_corpus_path)
            test_corpus = Corpus().load_from_xml(test_corpus_path)
            print("Evaluate without backoff \n---------------")
            self.evaluator.evaluate(neural_disambiguator, test_corpus, "wn30_key", self.wn)
            self.evaluator.save_result_to_file(test_corpus.get_documents(), "wsd_test", "/".join(self.weights[0].split("/")[:-1]))
            print("Evaluate with monosemics \n---------------")
            self.evaluator.evaluate(self.monosemicDisambiguator, test_corpus, "wn30_key", self.wn)
            print("Evaluate with backoff first sense \n---------------")
            self.evaluator.evaluate(self.firstSenseDisambiguator, test_corpus, "wn30_key", self.wn)

    def evaluate_mean_scores(self):
        neural_disambiguators = []
        for weight in self.weights:
            neural_disambiguator = NeuralDisambiguator(self.data_path, [weight], self.clear_text, self.batch_size,
                                                       wn=self.wn)
            neural_disambiguator.lowercase_words = self.lowercase
            neural_disambiguator.filter_lemma = self.filter_lemma
            neural_disambiguator.reduced_output_vocabulary = self.sense_compression_clusters
            neural_disambiguators.append(neural_disambiguator)

        for testCorpusPath in self.test_corpus_paths:
            print("Evaluate on corpus " + testCorpusPath)
            results_backoff_zero = MultipleDisambiguationResult()
            results_backoff_monosemics = MultipleDisambiguationResult()
            results_backoff_first_sense = MultipleDisambiguationResult()

            for i in range(len(self.weights)):
                neural_disambiguator = neural_disambiguators[i]
                test_corpus = Corpus().load_from_xml(testCorpusPath)
                print("" + str(i) + " : Evaluate without backoff \n---------------")
                result_backoff_zero = self.evaluator.evaluate(neural_disambiguator, test_corpus, "wn30_key", self.wn)
                print("" + str(i) + " : Evaluate with monosemics \n---------------")
                result_backoff_monosemics = self.evaluator.evaluate(self.monosemicDisambiguator, test_corpus,
                                                                    "wn30_key", self.wn)
                print("" + str(i) + " : Evaluate with backoff first sense \n---------------")
                result_backoff_first_sense = self.evaluator.evaluate(self.firstSenseDisambiguator, test_corpus,
                                                                     "wn30_key", self.wn)
                results_backoff_zero.add_disambiguation_result(result_backoff_zero)
                results_backoff_monosemics.add_disambiguation_result(result_backoff_monosemics)
                results_backoff_first_sense.add_disambiguation_result(result_backoff_first_sense)

            print("\nMean of scores without backoff: " + str(results_backoff_zero.score_mean()))
            print("Standard deviation without backoff: " + str(results_backoff_zero.score_standard_deviation()))
            print("Mean of scores with monosemics: " + str(results_backoff_monosemics.score_mean()))
            print("Standard deviation with monosemics: " + str(results_backoff_monosemics.score_standard_deviation()))
            print("Mean of scores with backoff first sense: " + str(results_backoff_first_sense.score_mean()))
            print("Standard deviation with backoff first sense: " + str(
                results_backoff_first_sense.score_standard_deviation()))


NeuralWSDEvaluate().main()
