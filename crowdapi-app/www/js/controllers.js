angular.module('starter.controllers', [])
  .controller('DashCtrl', function ($scope) {
    console.log("Controller Dash");
  })
  .controller('BrowseCtrl', function ($scope, $interval, $http, $cordovaDevice) {
    console.log("Controller Browse");
    document.addEventListener("deviceready",
      function () {
        var timeout = 1000;
        var lastedtResult = {};
        $scope.wifi = function () {
          WifiWizard.getScanResults(function (listHandler) {
            $scope.date = new Date();
            $scope.rssi = JSON.stringify(listHandler);
            lastedtResult = listHandler;

          }, function (error) {
            console.log("getScanResults error" + JSON.stringify(error));
          });
        };
        $scope.upload = function () {
          var rssiString = "";
          for (var i = 0; i < lastedtResult.length; i++) {
            var currentResult = lastedtResult[i];
            rssiString = rssiString + currentResult['BSSID'].replace(/:/g, "") + ":" + currentResult['level'] + ",0.0,0.0 ";
          }
          var pushdata = [{
            deviceid: $cordovaDevice.getUUID(),
            rssi: rssiString
          }];
          console.log("push data " + pushdata);
          $http.post('http://crowd.compathnion.com/api/v1.0/crowdsignatures/', pushdata)
            .then(function successCallback(response) {
              console.log("success" + JSON.stringify(response));

            }, function errorCallback(response) {
              console.log("error" + JSON.stringify(response));
            });
        };
        $interval($scope.wifi, timeout);
      },
      false);
  });

