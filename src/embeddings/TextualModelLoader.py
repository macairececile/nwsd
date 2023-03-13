import re
from embeddings.WordVectors import WordVectors

class TextualModelLoader:
    def __init__(self, skipFirstLine=False, verbose=None):
        self.verbose = verbose
        self.skipFirstLine = skipFirstLine

    def load(self, modelPath):
        try:
            return self.loadNoCatch(modelPath)
        except IOError as e:
            raise RuntimeError(e)

    def loadVocabularyOnly(self, modelPath):
        try:
            return self.loadVocabularyOnlyNoCatch(modelPath)
        except IOError as e:
            raise RuntimeError(e)

    def loadNoCatch(self, modelPath):
        i = 0
        j = 0
        reader = open(modelPath, 'r')
        if self.skipFirstLine:
            reader.readline()
        lines = reader.readlines()
        for line in lines:
            if self.verbose:
                print("Loading Word Vectors... (" + str(i + 1) + ")")
                lineSplitted = line.split(" ")
                j = len(lineSplitted) - 1
                i += 1
        if self.verbose:
            print()
        vectorCount = i
        vectorSize = j
        vectors = [[0.0] * vectorSize] * vectorCount
        words = [""] * vectorCount
        wordsIndexes = {}
        i = 0
        reader = open(modelPath, 'r')

        if self.skipFirstLine:
            reader.readline()
        lines = reader.readlines()
        for line in lines:
            if self.verbose:
                print("Loading Word Vectors... (" + str(i + 1) + "/" + str(vectorCount) + ")\r")
                lineSplitted = re.split(r'\s+', line)
                words[i] = lineSplitted[0]
                wordsIndexes[words[i]] = i
                for k in range(1, len(lineSplitted)):
                    vectors[i][k - 1] = float(lineSplitted[k])
                i += 1
        if self.verbose:
            print()
        return WordVectors(vectorCount, vectorSize, vectors, words, wordsIndexes)

    def loadVocabularyOnlyNoCatch(self, modelPath):
        i = 0
        reader = open(modelPath, 'r')
        lines = reader.readlines()
        for line in lines:
            if self.verbose:
                print("Loading Word Vectors... (count " + str(i) + ")\r")
                i += 1
        if self.verbose:
            print()
        vectorCount = i
        vectorSize = 0
        vectors = [[0.0] * vectorSize] * vectorCount
        words = ["" * vectorCount]
        wordsIndexes = {}
        i = 0
        for line in lines:
            if self.verbose:
                print("Loading Word Vectors... (" + str(i) + "/" + str(vectorCount) + ")\r")
                words[i] = line.split(" ")[0]
                wordsIndexes[words[i]] = i
                i += 1
        if self.verbose:
            print()
        return WordVectors(vectorCount, vectorSize, vectors, words, wordsIndexes)
