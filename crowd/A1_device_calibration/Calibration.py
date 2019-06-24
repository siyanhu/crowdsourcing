from Cosine import *
from Target import *
import numpy as np

class CalibrationConfig:
    def __init__(self, CALIBRATION_SIM_LB, CALIBRATION_AVG_RSSI_DIFF_LB):
        self.CALIBRATION_SIM_LB = CALIBRATION_SIM_LB
        self.CALIBRATION_AVG_RSSI_DIFF_LB = CALIBRATION_AVG_RSSI_DIFF_LB

    def __str__(self):
        return '[CalibrationConfig CALIBRATION_SIM_LB='+str(self.CALIBRATION_SIM_LB) + \
               ' CALIBRATION_AVG_RSSI_DIFF_LB='+str(self.CALIBRATION_AVG_RSSI_DIFF_LB)

class CalibrationServerConfig:
    DATA_DIR = ''
    CALIBRATION_CONFIG = CalibrationConfig(0.90, 3)
    NUM_OFFSET_UB = 1000

    def __init__(self, data_dir):
        self.DATA_DIR = data_dir

class CalibrationFactor:
    def __init__(self, needCalibration, offset):
        self.needCalibration = needCalibration
        self.offset = offset


class Calibration:
    def __init__(self):
        return

    @staticmethod
    def computeCalibrationFactor(refers, targets, calibrationConfig):
        count = 0
        avgDiff = 0.0
        for target in targets:
            diffs = Calibration.rssiDiffs(Calibration.findRefersWithHighSimilarity(refers, target, calibrationConfig.CALIBRATION_SIM_LB), target)
            if len(diffs) > 0:
                newCount = count + len(diffs)
                newAvgDiff = avgDiff * ((count * 1.0)/newCount) + np.sum(diffs) * 1.0 / newCount
                count = newCount
                avgDiff = newAvgDiff
        return Calibration.calibrationFactor(calibrationConfig.CALIBRATION_AVG_RSSI_DIFF_LB, avgDiff)

    @staticmethod
    def findRefersWithHighSimilarity(refers, target, CALIBRATION_SIM_LB):
        result = []
        for refer in refers:
            sims = Cosine.sim(refer, target, False)

            if sims >= CALIBRATION_SIM_LB:
                result.append(refer)
        return result

    @staticmethod
    def rssiDiffs(refers, target):
        result = []
        for refer in refers:
            for apid in target.tInfoMap:
                tLevel = target.tInfoMap[apid].level
                if apid in refer.rInfoMap:
                    rLevel = refer.rInfoMap[apid].mean
                    result.append(rLevel-tLevel)
        return result

    @staticmethod
    def calibrationFactor(CALIBRATION_AVG_RSSI_DIFF_LB, offset):
        return CalibrationFactor(abs(offset) >= CALIBRATION_AVG_RSSI_DIFF_LB, offset)

    @staticmethod
    def calibrate(target, cf):
        if not cf.needCalibration:
            return target
        tInfoMap = {}
        for apid in target.tInfoMap:
            level = target.tInfoMap[apid].level
            tInfoMap[apid] = TargetInfo(level + cf.offset, True)
        return Target(target.tid, Position(target.pos.x, target.pos.y), tInfoMap)
