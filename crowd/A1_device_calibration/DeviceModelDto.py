class DeviceModelDto:
    deviceModel = None

    def __init__(self, deviceModel):
        self.deviceModel = deviceModel

    def __eq__(self, other):
        if other is None:
            return False

        return self.deviceModel.lower() == other.deviceModel.lower()

    def __hash__(self):
        return hash(self.deviceModel.lower())
