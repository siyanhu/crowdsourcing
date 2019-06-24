#!/bin/bash
cd /home/ubuntu/workspace/crowdapi/
export aws_access_key_id="AKIAI7T5BFYU7KBOW7UA"
export aws_secret_access_key="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"
export AWS_PROFILE=crowd.compathnion.com
source /home/ubuntu/Envs/crowdapi/bin/activate 
python fetch_table.py >> /home/ubuntu/workspace/crowdapi/log/fetch_table.log
