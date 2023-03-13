import math
import random
import re

import numpy as np


class VectorOperation:
    @staticmethod
    def mul(a, b):
        return [x * b for x in a]

    @staticmethod
    def sub(a, b):
        return [x - y for x, y in zip(a, b)]

    @staticmethod
    def add(a, b):
        return [x + y for x, y in zip(a, b)]

    @staticmethod
    def sum(*vectors):
        return [sum(x) for x in zip(*vectors)]

    @staticmethod
    def norm(v):
        return np.linalg.norm(v)

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)

    @staticmethod
    def dot_product(a, b):
        return np.dot(a, b)

    @staticmethod
    def term_to_term_product_squared(a, b):
        return [math.copysign(math.sqrt(math.fabs(x * y)), x * y) for x, y in zip(a, b)]

    @staticmethod
    def term_to_term_product(a, b):
        ret = [0] * len(a)
        for i in range(len(ret)):
            ret[i] = a[i] * b[i]
        return ret

    @staticmethod
    def to_vector(string):
        if not string or string == "[]":
            return []
        str_values = re.sub(r"[\[\] ]", "", string).split(",")
        return [float(x) for x in str_values]

    @staticmethod
    def same(a, b):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True

    @staticmethod
    def generateRandomUnitVector(dimension):
        a = [0] * dimension
        while VectorOperation.norm(a) == 0:
            for i in range(len(dimension)):
                a[i] = random.gauss(0, 1)
        return VectorOperation.normalize(a)

    @staticmethod
    def toString1(vector):
        res = ""
        for i in range(len(vector)):
            res += str(vector[i]) + " "
        return res
