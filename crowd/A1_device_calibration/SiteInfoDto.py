class SiteInfoDto:
    area_id = None

    def __init__(self, area_id):
        self.area_id = area_id

    def __eq__(self, other):
        if other is None:
            return False

        return self.area_id == other.area_id

    def __hash__(self):
        return hash(self.area_id)
