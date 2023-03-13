from method.result.DisambiguationResult import DisambiguationResult
from method.result.MultipleDisambiguationResult import *
import math
import os.path
import io
from ufsac.common.POSConverter import POSConverter
from ufsac.common.WordnetHelper import WordnetHelper


class WSDEvaluator:
    def __init__(self, print_failed=False):
        self.print_results = False
        self.print_failed = print_failed

    def set_print_failed(self, print_failed):
        self.print_failed = print_failed

    def evaluate(self, disambiguator, corpus, sense_annotation_tag, wordnet, n=None):
        if n is None:
            return self.evaluate3(disambiguator, corpus, sense_annotation_tag, wordnet)
        else:
            return self.evaluate2(disambiguator, corpus, sense_annotation_tag, wordnet, n)

    def evaluate2(self, disambiguator, corpus, sense_annotation_tag, wordnet, n):
        results = MultipleDisambiguationResult()
        print("WSD " + disambiguator)
        for i in range(n):
            print(str(i + 1) + "/" + str(n) + " ")
            total_score = self.evaluate3(disambiguator, corpus, sense_annotation_tag, wordnet)
            results.add_disambiguation_result(total_score)
        print()
        print("Mean Scores : " + results.score_mean())
        print("Standard Deviation Scores : " + results.score_standard_deviation())
        print()
        return results

    def evaluate3(self, disambiguator, corpus, sense_annotation_tag, wordnet):
        total_score = DisambiguationResult()
        for document in corpus.get_documents():
            disambiguator.disambiguate(document, "wsd_test")
            document_score = self.compute_disambiguation_result(document.get_words(), sense_annotation_tag, "wsd_test",
                                                                wn=wordnet)
            document_score_ratio_percent = document_score.score_f1()
            print("(" + document.get_annotation_value("id") + ") " + "[{:.2f}] ".format(document_score_ratio_percent))
            total_score.concatenate_result(document_score)
        print("good/bad/missed/total : {}/{}/{}/{}".format(total_score.good, total_score.bad, total_score.missed(),
                                                           total_score.total))
        print("C/P/R/F1 : {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(total_score.coverage(), total_score.score_precision(),
                                                              total_score.score_recall(), total_score.score_f1()))
        for pos in ["n", "v", "a", "r", "x"]:
            print("[{}] good/bad/missed/total : {}/{}/{}/{}".format(pos, total_score.good_per_pos[pos],
                                                                    total_score.bad_per_pos[pos],
                                                                    total_score.missed_per_pos(pos), total_score.total_per_pos[pos]))
            print("[{}] C/P/R/F1 : {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(pos, total_score.coverage_per_pos(pos),
                                                                       total_score.score_precision_per_pos(pos),
                                                                       total_score.score_recall_per_pos(pos),
                                                                       total_score.score_f1_per_pos(pos)))
        # self.save_result_to_file(corpus.get_documents(), "wsd_test")
        return total_score

    def compute_disambiguation_result(self, word_list, reference_sense_tag, candidate_sense_tag,
                                      confidence_value_tag=None,
                                      confidence_threshold=None,
                                      wn=None):
        if confidence_value_tag is None and confidence_threshold is None and wn is None:
            return self.compute_disambiguation_result(word_list, reference_sense_tag, candidate_sense_tag, None, 0,
                                                      WordnetHelper.wn30())
        elif wn is None:
            return self.compute_disambiguation_result(word_list, reference_sense_tag, candidate_sense_tag,
                                                      confidence_value_tag,
                                                      confidence_threshold,
                                                      WordnetHelper.wn30())
        else:
            res = DisambiguationResult()
            for i in range(len(word_list)):
                word = word_list[i]
                word_pos = POSConverter.to_wn_pos(word.get_annotation_value("pos"))
                reference_sense_keys = word.get_annotation_values(reference_sense_tag, ";")
                if len(reference_sense_keys) == 0:
                    continue
                reference_synset_keys = []
                for ref_sense_key in reference_sense_keys:
                    ref_sense_key = ref_sense_key.lower()
                    if not wn.is_sense_key_exists(ref_sense_key):
                        continue
                    ref_synset_key = wn.get_synset_key_from_sense_key(ref_sense_key)
                    if ref_synset_key not in reference_synset_keys:
                        reference_synset_keys.append(ref_synset_key)
                if len(reference_synset_keys) == 0:
                    continue

                res.total += 1
                res.total_per_pos[word_pos] = res.total_per_pos.get(word_pos) + 1
                candidate_sense_key = word.get_annotation_value(candidate_sense_tag)
                if len(candidate_sense_key) == 0:
                    continue
                candidate_sense_key = candidate_sense_key.lower()
                if not wn.is_sense_key_exists(candidate_sense_key):
                    continue
                if word.has_annotation(confidence_value_tag):
                    confidence_value = float(word.get_annotation_value(confidence_value_tag))
                    if confidence_value != math.inf and confidence_value < confidence_threshold:
                        continue
                candidate_synset_key = wn.get_synset_key_from_sense_key(candidate_sense_key)
                res.bad += 1
                res.bad_per_pos[word_pos] = res.bad_per_pos[word_pos] + 1
                for ref_synset_key in reference_synset_keys:
                    if ref_synset_key == candidate_synset_key:
                        res.good += 1
                        res.bad -= 1
                        res.good_per_pos[word_pos] = res.good_per_pos[word_pos] + 1
                        res.bad_per_pos[word_pos] = res.bad_per_pos[word_pos] - 1
                        break
            return res

    def save_result_to_file(self, documents, candidate_sense_tag, outpath):
        if self.print_results:
            try:
                with open(outpath + "/wsd_test_decode.txt", "w", encoding="utf-8") as f:
                    for i in range(len(documents)):
                        sentences = documents[i].get_sentences()
                        for s in sentences:
                            words_in_s = s.get_words()
                            to_write = ''
                            for w in words_in_s:
                                word = w.get_annotation_value("surface_form")
                                word_sense_tag = ';'.join(w.get_annotation_values("wn30_key", ";"))
                                word_pred_sense_tag = w.get_annotation_value(candidate_sense_tag)
                                to_write += word + '|' + str(word_sense_tag) + '|' + str(word_pred_sense_tag) + '	'
                            f.write(to_write + '\n')
            except Exception as e:
                raise RuntimeError(e)    
