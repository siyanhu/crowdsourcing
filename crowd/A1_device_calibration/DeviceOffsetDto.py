class DeviceOffsetDto:
    offsets = None
    avgOffset = None

    def __init__(self):
        self.offsets = []
        self.avgOffset = 0.0

    def remove(self):
        return self.offsets.pop(0)

    def add(self, i):
        self.offsets.append(i)

    def addAll(self, l):
        self.offsets += l
