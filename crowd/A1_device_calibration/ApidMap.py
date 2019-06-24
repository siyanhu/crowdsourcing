class ApidMap:
    def __init__(self):
        self.bssid2Apid = {}

    def getApid(self, bssid):
        key = int(bssid, 16)
        return self.getId(key)

    def getId(self, key):
        if key not in self.bssid2Apid:
            self.bssid2Apid[key] = len(self.bssid2Apid)
        id = self.bssid2Apid[key]
        return id
