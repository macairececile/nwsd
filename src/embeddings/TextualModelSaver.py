class TextualModelSaver:
    def save(self, wordVectors, modelPath):
        try:
            self.saveNoCatch(wordVectors, modelPath)
        except IOError as e:
            raise RuntimeError(e)

    def saveNoCatch(self, wordVectors, modelPath):
        with open(modelPath, "w") as f:
            for word in wordVectors.get_vocabulary():
                vector = wordVectors.get_word_vector(word)
                f.write(word)
                for scalar in vector:
                    f.write(" {}".format(scalar))
                f.write("\n")
