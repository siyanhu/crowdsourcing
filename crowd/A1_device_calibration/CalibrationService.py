from SiteInfoDto import *
from SiteDataDao import *
from DeviceModelDto import *
from DeviceOffsetDto import *
from CommonUtil import *
from Calibration import *
import numpy as np

class CalibrationService:
    serverConfig = None
    siteDataDao = None

    siteDataCache = None
    deviceOffsetCache = None

    def __init__(self, serverConfig):
        self.serverConfig = serverConfig
        self.siteDataDao = SiteDataDao(self.serverConfig)
        self.siteDataCache = {}
        self.deviceOffsetCache = {}

    def getCalibrationFactorRaw(self, area_id, device_model):
        siteInfo = SiteInfoDto(area_id)
        deviceModel = DeviceModelDto(device_model)
        cf = self.getCalibrationFactor(siteInfo, deviceModel)
        return CommonUtil.calibrationFactor2String(cf)

    def getCalibrationFactor(self, siteInfo, deviceModel):
        avgOffset = 0.0
        if siteInfo in self.deviceOffsetCache and deviceModel in self.deviceOffsetCache[siteInfo]:
            avgOffset = self.deviceOffsetCache[siteInfo][deviceModel].avgOffset
        return Calibration.calibrationFactor(self.serverConfig.CALIBRATION_CONFIG.CALIBRATION_AVG_RSSI_DIFF_LB, avgOffset)

    def computeCalibrationFactorGivenTargetsRaw(self, area_id, device_model, fingerprints):
        siteInfo = SiteInfoDto(area_id)
        deviceModel = DeviceModelDto(device_model)
        apidMap = self.getSiteData(siteInfo).apidMap
        targets = CommonUtil.string2Targets(apidMap, fingerprints)
        cf = self.computeCalibrationFactorGivenTargets(siteInfo, deviceModel, targets)
        return CommonUtil.calibrationFactor2String(cf)

    def computeCalibrationFactorGivenTargets(self, siteInfo, deviceMode, targets):
        cf = self.getCalibrationFactor(siteInfo, deviceMode)
        for target in targets:
            cf = self.computeCalibrationFactorGivenTarget(siteInfo, deviceMode, target)
        return cf

    def computeCalibrationFactorGivenTargetRaw(self, area_id, device_model, fingerprint):
        siteInfo = SiteInfoDto(area_id)
        deviceModel = DeviceModelDto(device_model)
        apidMap = self.getSiteData(siteInfo).apidMap
        target = CommonUtil.string2Target(apidMap, fingerprint)
        cf = self.computeCalibrationFactorGivenTarget(siteInfo, deviceModel, target)
        return CommonUtil.calibrationFactor2String(cf)

    def computeCalibrationFactorGivenTarget(self, siteInfo, deviceModel, target):
        siteData = self.getSiteData(siteInfo)
        highSimRefers = Calibration.findRefersWithHighSimilarity(siteData.refers, target, self.serverConfig.CALIBRATION_CONFIG.CALIBRATION_SIM_LB)
        offsets = Calibration.rssiDiffs(highSimRefers, target)
        if len(offsets) == 0:
            return self.getCalibrationFactor(siteInfo, deviceModel)
        if len(offsets) > self.serverConfig.NUM_OFFSET_UB:
            for i in range(len(offsets)-1, self.serverConfig.NUM_OFFSET_UB, -1):
                offsets.pop(i)
        if siteInfo not in self.deviceOffsetCache:
            self.deviceOffsetCache[siteInfo] = {}
        deviceModel2offset = self.deviceOffsetCache[siteInfo]

        if deviceModel not in deviceModel2offset:
            deviceOffset = DeviceOffsetDto()
            deviceOffset.addAll(offsets)
            avgOffset = np.mean(offsets)
            deviceOffset.avgOffset = avgOffset
            deviceModel2offset[deviceModel] = deviceOffset

            return Calibration.calibrationFactor(self.serverConfig.CALIBRATION_CONFIG.CALIBRATION_AVG_RSSI_DIFF_LB, avgOffset)

        deviceOffset = deviceModel2offset[deviceModel]
        numOffsetsOld = len(deviceOffset.offsets)
        numOffsetsAdded = len(offsets)
        numOffsetsOutdated = numOffsetsOld + numOffsetsAdded - self.serverConfig.NUM_OFFSET_UB if numOffsetsOld + numOffsetsAdded > self.serverConfig.NUM_OFFSET_UB else 0
        numOffsetsNew = numOffsetsOld + numOffsetsAdded - numOffsetsOutdated
        sumOffsetsNewMinusEliminated = 0.0
        for i in range(numOffsetsOutdated):
            sumOffsetsNewMinusEliminated -= deviceOffset.remove()
        deviceOffset.addAll(offsets)
        sumOffsetsNewMinusEliminated += np.sum(offsets)
        newAvgOffset = deviceOffset.avgOffset * ((numOffsetsOld*1.0)/numOffsetsNew) + sumOffsetsNewMinusEliminated*1.0/numOffsetsNew

        return Calibration.calibrationFactor(self.serverConfig.CALIBRATION_CONFIG.CALIBRATION_AVG_RSSI_DIFF_LB, newAvgOffset)


    def getSiteData(self, siteInfo):
        if siteInfo in self.siteDataCache:
            return self.siteDataCache[siteInfo]

        siteData = self.siteDataDao.getSiteData()
        self.siteDataCache[siteInfo] = siteData

        return siteData