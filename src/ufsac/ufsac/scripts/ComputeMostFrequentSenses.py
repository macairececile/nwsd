class ComputeMostFrequentSenses:
    def main(self, corpusPaths):
        wordKeyToSenseKeyCount = {} # dict of string of dict
        wn = WordnetHelper.wn()
        for wordKey in wn.get_vocabulary():
            senseKeyCount = {}
            for senseKey in wn.get_sense_key_list_from_word_key(wordKey):
                senseKeyCount[senseKey] = 0
            wordKeyToSenseKeyCount[wordKey] = senseKeyCount

        corpus = StreamingCorpusReaderWord()

        def readWord(word):
            lemma = word.get_annotation_value("lemma")
            pos = word.get_annotation_value("pos")
            senseKey = word.get_annotation_value("wn" + wn.get_version() + "_key")
            if len(lemma) != 0 and len(pos) != 0 and len(senseKey) != 0:
                pos = POSConverter.to_wn_pos(pos)
                wordKey = lemma + "%" + pos
                senseKeyCount = wordKeyToSenseKeyCount.get(wordKey)
                exValue = senseKeyCount.get(senseKey)
                newValue = exValue + 1
                senseKeyCount[senseKey] = newValue

        for corpusPath in corpusPaths:
            corpus.load(corpusPath)

        for wordKey in wordKeyToSenseKeyCount.keys():
            mostFrequentSenseKey = ""
            mostFrequentSenseKeyCount = -1
            for senseKey in wordKeyToSenseKeyCount[wordKey].keys():
                if wordKeyToSenseKeyCount[wordKey][senseKey]:
                    mostFrequentSenseKey = senseKey
                    mostFrequentSenseKeyCount = wordKeyToSenseKeyCount[wordKey][senseKey]
            print(wordKey + " " + mostFrequentSenseKey)
