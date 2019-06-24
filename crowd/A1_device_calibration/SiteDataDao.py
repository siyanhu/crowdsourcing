from FileUtil import *
from ApidMap import *
from SiteDataDto import *
import os

class SiteDataDao:
    serverConfig = None

    def __init__(self, serverConfig):
        self.serverConfig = serverConfig
        return

    def getSiteData(self):
        apidMap = ApidMap()
        refers = []
        for file_name in os.listdir(self.serverConfig.DATA_DIR):
            if file_name[0] == '.':
                continue
            if not (file_name[:len(file_name)-4].isdigit()):
                continue
            file_path = self.serverConfig.DATA_DIR + file_name

            refers_area = FileUtil.readReferFile(file_path, apidMap)
            refers += refers_area
        siteData = SiteDataDto(apidMap, refers)
        return siteData