from MathUtil import *

class Target:
    def __init__(self, tid, pos, tInfoMap):
        self.tid = tid
        self.pos = pos
        self.tInfoMap = tInfoMap # key: apid

    def __str__(self):
        return 'Target['+str(self.tid)+'@'+str(self.pos)+']: '+str(self.tInfoMap)

class TargetInfo:
    def __init__(self, x, DBM):
        if DBM:
            self.level = x
            self.levelPower = MathUtil.dbm2Watt(x)
        else:
            self.level = MathUtil.watt2Dbm(x)
            self.levelPower = x