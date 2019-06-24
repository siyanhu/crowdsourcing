import boto.dynamodb2
from boto.dynamodb2.fields import HashKey, RangeKey, GlobalAllIndex
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import STRING
import time

table = 'crowdsignatures'
server_regin = 'ap-southeast-1'
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def fetchTable(name, region):
    raw_data = []
    try:
        print "fetching crowdsignatures table ..."
        devices = Table(table_name=name, connection=boto.dynamodb2.connect_to_region(region))
    except:
        print 'table does not exist'
    else:
        print "fetching and deleting Items"
        for rssi in devices.scan():
            raw_data.append(rssi['rssi'])
            rssi.delete()
    return raw_data

if __name__ == '__main__':
    device_data = fetchTable(table, server_regin)
    if len(device_data) > 0:
        try:
            f = open('crowd_target_raw.txt', 'w+')
            print "writing into the file ..."
            for i in device_data:
                f.write(i + '\n')
            f.close()
            print "finish writing, done!"
        except Exception, e:
            print 'IO error happen!'
    else:
        print 'no data in the table!'
