import heapq
import math

class MathUtil:

    def __init__(self):
        return

    @staticmethod
    def dbm2Watt(dbm):
        return 2 ** (dbm / 10.0)

    @staticmethod
    def watt2Dbm(watt):
        if watt <= 0:
            return -100
        return 10.0 * math.log(watt) / math.log(2)

    @staticmethod
    def entropy(a):
        total = sum(a)
        if total == 0:
            return 0
        result = 0
        for x in a:
            if x != 0:
                p = x * 1.0 / total
                result -= p * math.log(p,2)
        return result

    @staticmethod
    def topIs(a):
        topIs = []
        if len(a) == 0:
            return topIs

        max = a[0]
        for i in range(len(a)):
            if a[i] > max:
                max = a[i]
                topIs = [i]
            elif a[i] == max:
                topIs.append(i)
        return topIs

    @staticmethod
    def topKIs(a, k):
        if k == 0:
            return []

        entries = []
        for i in range(len(a)):
            heapq.heappush(entries, (a[i], i))
            if len(entries) > k:
                heapq.heappop(entries)

        topKIs = []
        for entry in entries:
            topKIs.append(entry[1])

        return topKIs

    @staticmethod
    def lwAvg(a, w):
        if len(a) != len(w):
            return 0

        sumW = sum(w)
        normW = [0] * len(w)
        for i in range(len(normW)):
            normW[i] = w[i] * 1.0 / sumW

        return MathUtil.lwAvgNormalized(a, normW)

    @staticmethod
    def lwAvgNormalized(values, weights):
        sum = 0
        for idx in range(len(values)):
            v = values[idx]
            w = weights[idx]
            sum += v * w
        return sum

    @staticmethod
    def cosineSimilarity(a, b):
        dotProduct = 0
        aNormSquare = 0
        bNormSquare = 0

        for i in range(len(a)):
            dotProduct += a[i] * b[i]
            aNormSquare += a[i] * a[i]
            bNormSquare += b[i] * b[i]

        if aNormSquare > 0 and bNormSquare > 0:
            cosSim = dotProduct / (math.sqrt(aNormSquare * bNormSquare))
        else:
            cosSim = 0

        return cosSim