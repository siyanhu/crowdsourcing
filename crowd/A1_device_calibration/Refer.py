from MathUtil import *

class Refer:
    def __init__(self, rid, pos, rInfoMap):
        self.rid = rid
        self.pos = pos
        self.rInfoMap = rInfoMap # key: apid


class ReferInfo:
    def __init__(self, mean, std):
        self.mean = mean
        self.meanPower = MathUtil.dbm2Watt(mean)
        self.std = std