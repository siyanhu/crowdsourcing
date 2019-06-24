from Position import *
from Refer import *
from Target import *

class FileUtil:

    def __init__(self):
        return

    @staticmethod
    def readReferFile(referFilePath, apidMap):
        refers = []
        f = open(referFilePath, 'r+')
        for line in f.readlines():
            line = line.replace('\r', '').replace('\n', '').replace(':', ',')
            if len(line) == 0:
                continue
            columns = line.split(' ')

            # require at least 3 dimensions (APs or beacons)
            if len(columns) < 4:
                continue

            # x, y
            values = columns[0].split(',')
            x = float(values[0])
            y = float(values[1])
            pos = Position(x, y)

            # ap
            rInfoMap = {}
            for column in columns[1:]:
                if len(column) == 0:
                    continue
                values = column.split(',')
                bssid = values[0].upper()
                apid = apidMap.getApid(bssid)
                mean = float(values[1])
                std = float(values[2])
                rInfoMap[apid] = ReferInfo(mean, std)
            refers.append(Refer(len(refers), pos, rInfoMap))
        f.close()
        return refers

    @staticmethod
    def readTargetFile(targetFilePath, apidMap):
        targets = []
        f = open(targetFilePath, 'r+')
        for line in f.readlines():
            line = line.replace('\r', '').replace('\n', '').replace(':', ',')
            if len(line) == 0:
                continue
            columns = line.split(' ')

            # require at least 3 dimensions (APs or beacons)
            if len(columns) < 4:
                continue

            # x, y
            values = columns[0].split(',')
            x = float(values[0])
            y = float(values[1])
            pos = Position(x, y)

            # ap
            tInfoMap = {}
            for column in columns[1:]:
                if len(column) == 0:
                    continue
                values = column.split(',')
                bssid = values[0].upper()
                apid = apidMap.getApid(bssid)
                level = float(values[1])
                tInfoMap[apid] = TargetInfo(level, True)
            targets.append(Target(len(targets), pos, tInfoMap))
        f.close()
        return targets

    @staticmethod
    def readConstraintFile(filename):
        ret = []
        f = open(filename, 'r+')
        for line in f.readlines():
            line = line.replace('\n', '').replace('\r', '')
            if len(line) == 0:
                continue
            ret.append(tuple(map(int, line.split(' '))))
        f.close()

        # remove redundant first and last coordinates if exist
        if len(ret) > 0 and ret[0] == ret[len(ret)-1]:
            del ret[len(ret)-1]
        return ret
