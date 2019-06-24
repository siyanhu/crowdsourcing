#!/usr/bin/env python
import boto.dynamodb2
from boto.dynamodb2.fields import HashKey, RangeKey, AllIndex, GlobalAllIndex
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import STRING
import time

print "deleting crowdsignatures table ..."
try:
    Table(table_name='crowdsignatures', connection=boto.dynamodb2.connect_to_region('ap-southeast-1')).delete()
    print "done"
except:
    print "crowdsignatures table not found"

done = False
print "creating crowdsignatures table ..."
while not done:
    try:
        crowdsignatures = Table.create('crowdsignatures',
            schema=[HashKey('deviceid'), RangeKey('timestamp', data_type=STRING),],
            global_indexes=[
                     GlobalAllIndex('Signatures',
                              parts=[HashKey('signatures'),
                              RangeKey('timestamp', data_type=STRING),]
                              ),
                    ],
            connection= boto.dynamodb2.connect_to_region('ap-southeast-1')
        )
        print "done"
        print "crowdsignatures.count() = ",crowdsignatures.count()
        done = True
    except:
        time.sleep(3) 
        print "..."
