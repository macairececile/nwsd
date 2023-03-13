import struct
from embeddings.WordVectors import *
from io import BytesIO


class BinaryModelLoader:
    def __int__(self, skipEOLChar=False, verbose=True):
        self.skipEOLChar = skipEOLChar
        self.verbose = verbose

    def load(self, modelPath):
        try:
            return self.loadNoCatch(modelPath)
        except IOError as e:
            raise RuntimeError(e)

    def loadNoCatch(self, modelPath):
        with open(modelPath, 'rb') as f:
            data = f.read()
        vectorCount = int(self.readString(data))
        vectorSize = int(self.readString(data))
        vectors = [[0] * vectorSize for i in range(vectorCount)]
        words = [''] * vectorCount
        wordsIndexes = {}
        lastPercentage = 0
        if self.verbose:
            print("Loading " + str(vectorCount) + " vectors of size " + str(vectorSize) + " from " + modelPath)
        for i in range(0, vectorCount):
            if self.verbose:
                currentPercentage = (int(((float(i + 1)) / (float(vectorCount))) * 100.0))
                if currentPercentage > lastPercentage:
                    print("Loading vectors... (" + str(currentPercentage) + "%)\r")
                    lastPercentage = currentPercentage
            words[i] = self.readString(data)
            if len(words[i]) == 0:
                for j in range(0, vectorSize):
                    vectors[i][j] = ByteOperations.readFloat(data)
                    vectors[i][j] = 0
                if self.skipEOLChar:
                    data.readBytes()
            else:
                wordsIndexes[words[i]] = i
                for j in range(0, vectorSize):
                    vectors[i][j] = ByteOperations.readFloat(data)
                if self.skipEOLChar:
                    dis = BytesIO(data).read(1)[0]
                if VectorOperation.norm(vectors[i]) != 0:
                    vectors[i] = VectorOperation.normalize(vectors[i])
        if self.verbose:
            print()
        return WordVectors(vectorCount, vectorSize, vectors, words, wordsIndexes)

    @staticmethod
    def readString(dis):
        buffer_size = 50
        bytess = [bytes(buffer_size)]
        b = struct.unpack('b', dis.read(1))[0]
        i = -1
        sb = ''
        while b != '' and b != '\n' and b != '\t' and b != '\0':
            i += 1
            bytess[i] = b
            b = struct.unpack('b', dis.read(1))[0]
            if i == buffer_size - 1:
                sb += str(bytess)
                i = -1
                bytess = [bytes(buffer_size)]
        sb += str(bytess[0:i+1])
        return sb






