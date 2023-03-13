import argparse

from method.FirstSenseDisambiguator import FirstSenseDisambiguator
from method.neural.NeuralDisambiguator import NeuralDisambiguator
from utils.WordnetUtils import WordnetUtils
from ufsac.common.WordnetHelper import WordnetHelper
from ufsac.ufsac.streaming.modifier.StreamingCorpusModifierSentence import StreamingCorpusModifierSentence
from ufsac.ufsac.utils.CorpusPOSTaggerAndLemmatizer import CorpusPOSTaggerAndLemmatizer


class NeuralWSDDecodeUFSAC:
    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("python_path")
        parser.add_argument("data_path")
        parser.add_argument("weights")
        parser.add_argument("input")
        parser.add_argument("output")
        parser.add_argument("lowercase", default=False)
        parser.add_argument("sense_reduction", default=True)
        parser.add_argument("clear_text", default=True)
        parser.add_argument("batch_size", default="1")
        parser.add_argument("mfs_backoff", default=True)

        args = parser.parse_args()
        pythonPath = args.python_path
        dataPath = args.data_path
        weights = args.weights
        inputPath = args.input
        outputPath = args.output
        lowercase = args.lowercase
        senseReduction = args.sense_reduction
        clearText = args.clear_text
        batchSize = args.batch_size
        mfsBackoff = args.mfs_backoff

        wn = WordnetHelper.wn30()
        tagger = CorpusPOSTaggerAndLemmatizer()
        firstSenseDisambiguator = FirstSenseDisambiguator(wn)
        neuralDisambiguator = NeuralDisambiguator(pythonPath, dataPath, weights, clearText, batchSize, wn=wn)
        neuralDisambiguator.lowercase_words = lowercase
        if senseReduction:
            neuralDisambiguator.reduced_output_vocabulary = WordnetUtils.get_reduced_synset_keys_with_hypernyms3(
                wn)
        else:
            neuralDisambiguator.reduced_output_vocabulary = None

        modifier = StreamingCorpusModifierSentence()

        def modifySentence(sentence):
            tagger.tag(sentence.get_words())
            neuralDisambiguator.disambiguate(sentence, "wsd")
            if mfsBackoff:
                firstSenseDisambiguator.disambiguate(sentence, "wsd")

        modifier.load(inputPath, outputPath)
