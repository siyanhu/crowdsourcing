#!/usr/bin/env python
import json
import requests
import string
import random
from datetime import datetime
import gzip
import StringIO
import sys
import os
import time
import random
from optparse import OptionParser
import uuid

prod = 'http://crowd.compathnion.com'
signatures_endpoint ="/api/v1.0/crowdsignatures/"

###############################
def printHttp(r):
    print "Request Headers: %s" % r.request.headers
    print "Request Body: %s\n" % r.request.body
    print "Response Status code: %s" % r.status_code
    print "Response Headers: %s" % r.headers
    print "Response Body: %s\n" % r.content    

def createdata(total):
    deviceids = []
    for i in range(total):
        deviceids.append({'deviceid':str(uuid.uuid4()),'rssi':"a 1 b 2 c 3 d 4",})
    return deviceids

def postsignatures(host, endpoint, data):
    headers = {'Content-type': 'application/json'}
    r = requests.post(host+endpoint, data=json.dumps(data), headers=headers, verify=False)
    return r

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--server", dest="server", action="store", default='prod',
                  help="server type [e.g. 'prod', 'dev', 'local']")
    (options, args) = parser.parse_args()
    print "options.server %s" % options.server

    argstr = ""
    for a in sys.argv[1:]:
        argstr = argstr + a + " "
    print "argstr = %s" % argstr
    
    host = prod
    data = createdata(5)
    print "deviceids = ",data

    r = postsignatures(host, signatures_endpoint, data)
    printHttp(r)
    result = json.loads(r.content)
    assert result['success']==1
    assert result['count']==5

    done = False
    sleep = 0;

    deviceids = [d['deviceid'] for d in data]
    m = {}
    for d in data:
        m[d['deviceid']] = d['rssi']
    print m
    for deviceid in deviceids:
        r = requests.get(host+'/api/v1.0/crowdsignatures/'+deviceid, verify=False)
        data = json.loads(r.content)
        assert m[data['deviceid']] == data['rssi']
