from evaluation.WSDEvaluator import WSDEvaluator
from method.result.MultipleDisambiguationResult import *
import scipy.stats as stats


class MultipleWSDEvaluator:
    def __init__(self):
        self.inputs = []
        self.test_corpus_path = ""
        self.sense_annotation_tag = ""
        self.single_evaluator = WSDEvaluator()

    def set_print_failed(self, print_failed):
        self.single_evaluator.set_print_failed(print_failed)

    def set_test_corpus(self, corpus_path, sense_annotation_tag):
        self.test_corpus_path = corpus_path
        self.sense_annotation_tag = sense_annotation_tag

    def add_disambiguator(self, wsd):
        self.inputs.append(wsd)

    def evaluate(self, n):
        res = [MultipleDisambiguationResult() for _ in self.inputs]
        for i in range(len(res)):
            res[i] = self.single_evaluator.evaluate(self.inputs[i], self.test_corpus_path, self.sense_annotation_tag, n)
        print("Recap : ")
        for i in range(len(res)):
            print("Test " + str(i) + " (" + str(self.inputs[i]) + ")")
            print("Mean Scores : " + str(res[i].score_mean()))
            print("Standard Deviation Scores : " + str(res[i].score_standard_deviation()))
            print("Mean Times : " + str(res[i].time_mean()))
            print()
        for i in range(len(res)):
            for j in range(len(res)):
                print("MWUTest " + str(i) + " (" + str(self.inputs[i]) + ") / " + str(j) + " (" + str(
                    self.inputs[j]) + ") : " + str(stats.mannwhitneyu(res[i].all_scores(), res[j].all_scores())))
