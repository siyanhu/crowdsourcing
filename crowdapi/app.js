/**
 * Module dependencies.
 */

var express = require('express');
var routes = require('./routes');
var user = require('./routes/user');
var http = require('http');
var https = require('https');
var path = require('path');
var AWS = require('aws-sdk');
var appRoot = "/home/ubuntu/workspace/crowdapi";
console.log("appRoot = "+appRoot);
AWS.config.loadFromPath(appRoot+'/'+'config.json');
AWS.config.api_version = 'v1.0';
var db = new AWS.DynamoDB();
var app = express();
var zlib = require('zlib');
var moment = require('moment');

// all environments
app.set('port', process.env.PORT || 8080);
app.set('views', __dirname + '/views');
app.set('view engine', 'jade');
app.use(express.favicon());
app.use(express.logger('dev'));
app.use(express.bodyParser());
app.use(express.methodOverride());
app.use(app.router);
app.use(express.static(path.join(__dirname, 'public')));

if ('development' === app.get('env')) {
	app.use(express.errorHandler());
}

app.get('/', function(req, res) {
	res.render('index', {
		title : "Wherami Crowd API"
	});
});

app.get('/api/v1.0/crowdsignatures/:deviceid', function(request, response) {
    console.log("/api/v1.0/crowdsignaturs/"+request.params.deviceid)
	var params = {
		"TableName" : "crowdsignatures",
		"KeyConditions" : {
			"deviceid" : {
				ComparisonOperator : 'EQ',
				AttributeValueList : [ {
					"S" : request.params.deviceid
				} ]
			}
		}
	};
	db.query(params, function(err, data) {
		if (err) {
			console.log("get signature error");
			console.log(err);
			response.send(err);
		} else {
			console.log(data);
			if (data.Count == 1) {
				response.json({
					'deviceid' : data.Items[0].deviceid.S,
					'timestamp' : data.Items[0].timestamp.S,
					'rssi' : data.Items[0].rssi.S
				});
			} else {
				response.status(403).send("signature not found");
			}
		}
	});
});

app.post('/api/v1.0/crowdsignatures', function(request, response) {
	console.log('/api/v1.0/crowdsignatures');
	for (var i = 0; i < request.body.length; i++) {
		var item = request.body[i];
	    console.log(item);
		var now = moment.utc().format('YYYY-MM-DD HH:mm:ss.ms');
		var params = {
			"TableName" : "crowdsignatures",
			"Item" : {
				"deviceid" : {
					"S" : item.deviceid
				},
                "rssi" : {
                    "S" : item.rssi
                },
				"timestamp" : {
					"S" : now
				}
			}
		};

		// console.log(params);
		db.putItem(params, function(err, data) {
			console.log("db.putItem signature " + JSON.stringify(params));
			if (err) {
				console.log(err);
				var ret = {
					"success" : 0,
					"error" : err
				};
				response.end(JSON.stringify(ret));
			}
		});
	}
	response.writeHead(200, {
		'Content-Type' : 'application/json'
	});
	var ret = {
		"success" : 1,
		"count" : request.body.length
	};
	response.end(JSON.stringify(ret));
});

http.createServer(app).listen(app.get('port'), function() {
	db.listTables(function(err, data) {
		if (err) {
			console.log("db.listTables error!");
		} else {
			console.log(data.TableNames);
		}
	});
	console.log(moment.utc().format());
	console.log('Express server listening on port ' + app.get('port'));
});
