from Position import *
from Target import *

class CommonUtil:
    def __init__(self):
        return

    @staticmethod
    def formatTarget(rawTarget, apidMap):
        tInfoMap = {}
        for bssid in rawTarget:
            if int(bssid, 16) not in apidMap.bssid2Apid:
                continue
            tInfoMap[apidMap.getApid(bssid)] = TargetInfo(rawTarget[bssid], True)
        return Target(0, Position(0, 0), tInfoMap)

    @staticmethod
    def calibrationFactor2String(cf):
        return str(cf.needCalibration)+'_'+str(cf.offset)

    @staticmethod
    def string2Target(apidMap, fingerprint):
        tInfoMap = {}
        for bssidRssi in fingerprint.split('__'):
            pair = bssidRssi.split('_')
            bssid = pair[0].upper()
            rssi = float(pair[1])
            apid = apidMap.getApid(bssid)
            tInfoMap[apid] = TargetInfo(rssi, True)
        return Target(0, Position(0, 0), tInfoMap)

    @staticmethod
    def string2Targets(apidMap, fingerprints):
        targets = []
        for fingerprint in fingerprints.split('___'):
            targets.append(CommonUtil.string2Target(apidMap, fingerprint))
        return targets