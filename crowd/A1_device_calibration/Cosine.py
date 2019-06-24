from MathUtil import *
from Position import *
import time

class Cosine:

    def __init__(self):
        return

    @staticmethod
    def estimate(refers, target, cosineConfig):
        startTime = int(round(time.time() * 1000))

        # calculate cosine similarity
        sims = Cosine.sims(refers, target, False)

        # top K
        topRIs = MathUtil.topKIs(sims, cosineConfig.K)

        # weighted average
        est = Cosine.wAvg(refers, sims, topRIs)

        totalTime = int(round(time.time() * 1000)) - startTime

        return CosineResult(sims, topRIs, est, totalTime)

    @staticmethod
    def sims(refers, target, DBM):
        sims = [0] * len(refers)
        for ri in range(len(refers)):
            sims[ri] = Cosine.sim(refers[ri], target, DBM)
        return sims

    @staticmethod
    def sim(refer, target, DBM):
        rLevels = [0] * len(target.tInfoMap)
        tLevels = [0] * len(target.tInfoMap)
        if len(list(set(refer.rInfoMap.keys()).intersection(set(target.tInfoMap.keys())))) < 3:
            return 0.0
        i = 0
        for apid in target.tInfoMap.keys():
            if apid in refer.rInfoMap.keys():
                rLevels[i] = refer.rInfoMap[apid].mean if DBM else refer.rInfoMap[apid].meanPower
            else:
                rLevels[i] = -100 if DBM else 0
            tLevels[i] = target.tInfoMap[apid].level if DBM else target.tInfoMap[apid].levelPower
            i += 1
        return MathUtil.cosineSimilarity(rLevels, tLevels)

    @staticmethod
    def wAvg(refers, cosSims, topRIs):
        N = len(topRIs)
        xs = [0] * N
        ys = [0] * N
        ws = [0] * N
        for i in range(0, N):
            xs[i] = refers[topRIs[i]].pos.x
            ys[i] = refers[topRIs[i]].pos.y
            if cosSims[topRIs[i]] == 1:
                cosSims[topRIs[i]] = 0.99
            ws[i] = 1.0 / math.pow(1-cosSims[topRIs[i]], 2)

        x = MathUtil.lwAvg(xs, ws)
        y = MathUtil.lwAvg(ys, ws)

        return Position(x, y)


class CosineConfig:
    def __init__(self, K):
        self.K = K

    def __str__(self):
        return '[Cosine K='+str(self.K)+']'


class CosineResult:
    def __init__(self, sims, topRIs, est, totalTime):
        self.sims = sims
        self.topRIs = topRIs
        self.est = est
        self.totalTime = totalTime