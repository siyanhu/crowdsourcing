from __future__ import division

import re
import math
import datetime as dt
from copy import deepcopy

import numpy as np
from matplotlib import cm
import matplotlib.mlab as ml
import pylab as pl
from pylab import imread
import pandas as pd

import os.path, time
from datetime import datetime
from  os import listdir
from  shutil import move
import shutil
from  shutil import copyfile
from array import array
from scipy import spatial
import numpy.matlib
from  scipy import special
from  scipy import linalg
import A0_global_settings as A0

### the following modules are for function call visulization
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#from pycallgraph import Config
#from pycallgraph import GlobbingFilter
### the above modules are for function call visulization

################################################################################################ utilities
# input: a single RSSI value, e.g. -65 (dBm), normally,  -20dBm < rssi < -90dBm
# output: a single vale, e.g. 55
# this function is to transform a negtive RSSI value to a positive value, for the propose of similicity in algorithm
def rssi_negative_2_positive ( rssi ):
    if rssi==0:
        return 0
    else:
        return rssi+A0.global_settings['rssi_negative_2_positive_shift']

# input: a single vale, e.g. 55
# output: a single RSSI value, e.g. -65 (dBm), normally,  -20dBm < rssi < -90dBm
# this function is to transform a positive RSSI value to a negtive value, for the propose of similicity in algorithm
def rssi_positive_2_negative( rssi ):
    if rssi==0:
        return 0
    elif rssi-A0.global_settings['rssi_negative_2_positive_shift']<-95:
        return 0
    else:
        return rssi-A0.global_settings['rssi_negative_2_positive_shift']

# input: the prefix of txt file
# output: the newest file whose filename contains the prefix
# this function is to find the newest file based on timestamp of file creation
def get_newest_txt_file(filename_prefix):
    newest_fingerprint_file_name=''
    newest_fingerprint_file_time = dt.datetime.now().replace(year=1900)

    files = [f for f in os.listdir(A0.global_settings['fingerprint_txt_file_folder']) if os.path.isfile(A0.global_settings['fingerprint_txt_file_folder']+f)]
    for file in files:
        if filename_prefix in file:
            modified_date = datetime.strptime(time.ctime(os.path.getmtime(A0.global_settings['fingerprint_txt_file_folder']+file)),  '%a %b %d %H:%M:%S %Y')
            if modified_date>newest_fingerprint_file_time:
                newest_fingerprint_file_time = modified_date
                newest_fingerprint_file_name = file
                continue
    return A0.global_settings['fingerprint_txt_file_folder']+newest_fingerprint_file_name

# input: (1)fingerprints_byRP_txt_file: the fingerprint file,
#        (2)AP_dic,RP_dic: two dictionary used to contain reference points(RP), and AP mac address
# output:(1)fingerprint_byRP_3D_array: a 3D arrary which is fingerprint database,
#           the data structure is mainly organized according to the reference points (RP)
# this function is to load fingerprint. For similicity, RP and AP are all indexed using dictionary
def fingerprints_byRP_txt_2_3D_array (fingerprints_byRP_txt_file, AP_dic,RP_dic):
    _AP_dic=deepcopy(AP_dic)
    _RP_dic=deepcopy(RP_dic)
    to_process_initial_fingerprint = 0
    if _AP_dic=={} and _RP_dic=={}:
        to_process_initial_fingerprint=1
    f = open(fingerprints_byRP_txt_file, 'r')
    total_RP_num = sum(1 for line in f)
    f.seek(0)
    estimated_max_AP_num_in_each_RP = A0.global_settings['estimated_max_AP_num_in_each_RP']
    data_num_in_each_AP = 3  # AP_index + RSSI+sd
    fingerprint_byRP_3D_array = np.zeros((total_RP_num,  estimated_max_AP_num_in_each_RP,  data_num_in_each_AP))

    rp_num_count = -1
    for line in f:
        rp_num_count +=1

        temp=re.split(',|:|\s',line)
        # put (x, y, d) into dic
        if to_process_initial_fingerprint==1:
            k=1
            while ((temp[0], temp[1],str(k)) in _RP_dic):
                k+=1
            temp[2]=str(k)
            _RP_dic [(temp[0], temp[1], temp[2])] =int( len(_RP_dic)/2 +1)
            _RP_dic [ _RP_dic[(temp[0], temp[1], temp[2])] ] = (temp[0], temp[1], temp[2])

            #if (temp[0], temp[1], temp[2], rp_num_count) not in _RP_dic:
            #    _RP_dic [(temp[0], temp[1], temp[2])] =int( len(_RP_dic)/2 +1)
            #    _RP_dic [ _RP_dic[(temp[0], temp[1], temp[2])] ] = (temp[0], temp[1], temp[2])
            #else:
            #    continue # the initial fingerprint file has duplicated records, ignore them
        else:
            if (temp[0], temp[1], temp[2]) not in _RP_dic:
                continue # ignore the new x_y_d, the x_y_d is not in the list


        # to read mac+rssi in each line
        ap_num_count = 0
        for t in range(4,len(temp) +1,4):
            rp_index = _RP_dic[(temp[0], temp[1], temp[2])]
            fingerprint_byRP_3D_array[rp_num_count, 0, 0] = rp_index
            if len(temp)<=4:
                continue
            if temp[t]=='':
                break
            if ap_num_count >= estimated_max_AP_num_in_each_RP:
                raise Exception('fingerprints_byRP_txt_2_3D_array()', 'fingerprint_byRP_3D_array is too small\n')
            if to_process_initial_fingerprint==1:
                if temp[t] not in _AP_dic:  # mac -> index
                    _AP_dic[temp[t]] = int( len(_AP_dic)/2 + 1)
                    _AP_dic[ _AP_dic[temp[t]] ] = temp[t]
            else:
                if temp[t] not in _AP_dic:
                    continue # ignore the new AP which is not in the ap list

            ap_num_count+=1
            ap_index = _AP_dic[temp[t]]
            fingerprint_byRP_3D_array[rp_num_count, ap_num_count, 0] = ap_index
            fingerprint_byRP_3D_array[rp_num_count, ap_num_count, 1] = rssi_negative_2_positive(float(temp[t+1]))
            if temp[t+2]=='':
                temp[t+2]='0.0'
            fingerprint_byRP_3D_array[rp_num_count, ap_num_count, 2] = temp[t+2]

    if rp_num_count+1 < total_RP_num: # have duplicated RP
        fingerprint_byRP_3D_array=fingerprint_byRP_3D_array[:rp_num_count+1]

    return fingerprint_byRP_3D_array, _AP_dic, _RP_dic

# input: (1)fingerprint_byRP_3D_array: a 3D arrary which is fingerprint database
#        (2)AP_dic, RP_dic: the RP/AP dictionary associated with fingerprint database
#        (3)output_posterior_txt_file: the file name for output txt file
# output: a txt file
# this function is a duel function of 'fingerprints_byRP_txt_2_3D_array()',
# it is to transform fingeprint database to a txt file
def fingerprints_byRP_3D_array_2_txt (fingerprint_byRP_3D_array, AP_dic, RP_dic, output_posterior_txt_file):
    RP_num = len (fingerprint_byRP_3D_array)
    AP_no = len (fingerprint_byRP_3D_array[0])
    #if RP_num != len (RP_dic) / 2:
    #    raise Exception('fingerprints_byRP_3D_array_2_txt()', 'RP_num is inconsistent\n')

    #f = open('fingerprints_byRP_txt_file.txt', 'w')
    f = open(output_posterior_txt_file, 'w')
    for rp_num in range(RP_num ):
        rp_index = fingerprint_byRP_3D_array[rp_num,0,0]
        if rp_index==0:
            continue
        line = (RP_dic[rp_index][0]) + ',' + (RP_dic[rp_index][1]) +  ',' \
            + (RP_dic[rp_index][2]) +  ',0 '    # x, y, d, index
        for ap in range(1, AP_no, 1):
            if fingerprint_byRP_3D_array[rp_num, ap, 0] == 0:
                break
            if rssi_positive_2_negative( fingerprint_byRP_3D_array[rp_num, ap, 1])==0:
                continue
            line = line + AP_dic[ fingerprint_byRP_3D_array[rp_num, ap, 0] ] + ':' \
                + ("%.3f" % rssi_positive_2_negative( fingerprint_byRP_3D_array[rp_num, ap, 1])) + ',' \
                + ("%.3f" % fingerprint_byRP_3D_array[rp_num, ap, 2]) + ',0.0 '
        f.write( line + '\n')

    f.close()
    return 0

# input: fingerprint_byRP_3D_array: a 3D arrary which is fingerprint database organized according to different RP
# output: fingerprint_byAP_3D_array: a 3D arrary which is fingerprint database organized according to different AP
# this function is to tranform the internal data structure used to represent fingerprint database
def fingerprints_byRP_2_byAP(fingerprint_byRP_3D_array):
    total_RP_num = len(fingerprint_byRP_3D_array)

    ap_index_list=[]
    for rp_num in range(0,total_RP_num ,1):
        ap_num = len(fingerprint_byRP_3D_array[rp_num])
        for ap in range(1, ap_num,1):
            if fingerprint_byRP_3D_array[rp_num, ap, 0] == 0:
                continue
            if fingerprint_byRP_3D_array[rp_num, ap, 0] not in ap_index_list:
                ap_index_list.append(fingerprint_byRP_3D_array[rp_num, ap, 0])
    total_AP_num=len(ap_index_list)

    data_num_in_each_RP = 3 # rp_index, rssi, sd
    fingerprint_byAP_3D_array = np.zeros((total_AP_num, total_RP_num + 1, data_num_in_each_RP))
    for i in range(total_RP_num): # initilize the rp_index for each ap
        fingerprint_byAP_3D_array[:, i+1, 0]=fingerprint_byRP_3D_array[i,0,0]

    temp_AP_dic = {}
    for rp_num in range(0,total_RP_num ,1):
        rp_index = fingerprint_byRP_3D_array[rp_num, 0, 0] #           0
        for ap in range(1, len(fingerprint_byRP_3D_array[rp_num]), 1):
            if fingerprint_byRP_3D_array[rp_num, ap, 0] == 0:
                continue
            ap_index = fingerprint_byRP_3D_array[rp_num,ap,0] # ap_index       0,

            if ap_index not in temp_AP_dic:
                temp_AP_dic[ap_index] = len(temp_AP_dic)
            ap_count = temp_AP_dic[ap_index]

            fingerprint_byAP_3D_array[ap_count,0, 0] = ap_index
            if fingerprint_byAP_3D_array[ap_count, rp_num+1, 0] != rp_index:
                raise Exception('fingerprints_byRP_2_byAP()', 'index is inconsistent!!\n')
            fingerprint_byAP_3D_array[ap_count, rp_num+1, 1] = fingerprint_byRP_3D_array[rp_num, ap, 1] # rssi  1,
            fingerprint_byAP_3D_array[ap_count, rp_num+1, 2] = fingerprint_byRP_3D_array[rp_num, ap, 2] # sd    2,

    return fingerprint_byAP_3D_array

# input: fingerprint_byAP_3D_array: a 3D arrary which is fingerprint database organized according to different AP
# output: fingerprint_byRP_3D_array: a 3D arrary which is fingerprint database organized according to different RP
# this is a duel function of 'fingerprints_byRP_2_byAP()'
def fingerprints_byAP_2_byRP(fingerprint_byAP_3D_array):
    total_AP_num = len(fingerprint_byAP_3D_array)
    if total_AP_num==0:
        return []
    total_RP_num = len(fingerprint_byAP_3D_array[0]) - 1

    estimated_max_AP_num_in_each_RP = A0.global_settings['estimated_max_AP_num_in_each_RP']
    data_num_in_each_AP = 3  # AP_index + RSSI+sd
    fingerprint_byRP_3D_array = np.zeros((total_RP_num,  estimated_max_AP_num_in_each_RP,  data_num_in_each_AP))

    for i in range(total_RP_num): # initilize rp_index
        fingerprint_byRP_3D_array[i,0,0] = fingerprint_byAP_3D_array[0,i+1,0]

    for rp_index in range(1,total_RP_num+1,1):
        ap_count=0
        for ap_num in range(total_AP_num):
            if fingerprint_byAP_3D_array[ap_num,rp_index,0]!=fingerprint_byRP_3D_array[rp_index-1,0,0]:
                raise Exception('fingerprints_byAP_2_byRP()','fingerprint_byAP_3D_array[ap_num,rp_index,0]!=rp_index\n')
            if fingerprint_byAP_3D_array[ap_num,rp_index,1] !=0:
                ap_count+=1
                if ap_count>=A0.global_settings['estimated_max_AP_num_in_each_RP']:
                    print('Too many APs at a single RP location, ignored! rp_index='+str(rp_index)+'\n')
                    continue
                fingerprint_byRP_3D_array[rp_index-1,ap_count,0]=fingerprint_byAP_3D_array[ap_num,0,0] # ap_index
                fingerprint_byRP_3D_array[rp_index-1,ap_count,1]=fingerprint_byAP_3D_array[ap_num,rp_index,1] # rssi
                fingerprint_byRP_3D_array[rp_index-1,ap_count,2]=fingerprint_byAP_3D_array[ap_num,rp_index,2] # sd

    return fingerprint_byRP_3D_array

# input: (1)fingerprints_byAP_3D_array: a 3D arrary which is fingerprint database organized according to different AP
#        (2)prior_AP_dic,prior_RP_dic: RP/AP dictionary associated with fingerprint database
#        (3)output_filename_prefix: output txt file name
# output: a txt fingerprint file
# this function is to transform fingerpint 3D array to txt file, it is similar to 'fingerprints_byRP_3D_array_2_txt()'
def fingerprints_byAP_3D_array_2_txt(fingerprints_byAP_3D_array,prior_AP_dic,prior_RP_dic,output_filename_prefix):
    now=dt.datetime.now()
    output_posterior_txt_file=A0.global_settings['fingerprint_txt_file_folder']+output_filename_prefix+'_' + str(now)[0:10]+'-'+str(now.time())[0:8].replace(':','-')+'.txt'
    # output fingerprint
    fingerprints_byRP_3D_array = fingerprints_byAP_2_byRP(fingerprints_byAP_3D_array)
    fingerprints_byRP_3D_array_2_txt(fingerprints_byRP_3D_array, prior_AP_dic, prior_RP_dic,output_posterior_txt_file)
    return 0

# input: (1)byAP_3D_array: fingerprint dababase organized by different AP
#        (2)ap_list_1: a list of AP index, it is not AP mac address
# output:(1)byAP_3D_array_1: fingerprint daba which only contains AP of ap_list_1
#        (2)byAP_3D_array_2: the remaining fingeprint data
# this function is to split the fingerprint data into two parts
def byAP_3D_array_split(byAP_3D_array,ap_list_1):
    byAP_3D_array_2=np.zeros((byAP_3D_array.shape[0]-len(ap_list_1),byAP_3D_array.shape[1],byAP_3D_array.shape[2]))
    byAP_3D_array_1=np.zeros((len(ap_list_1),byAP_3D_array.shape[1],byAP_3D_array.shape[2]))
    count=-1
    for ap_num in range(byAP_3D_array.shape[0]):
        if byAP_3D_array[ap_num,0,0] in ap_list_1:
            count+=1
            byAP_3D_array_1[count,:,:]=byAP_3D_array[ap_num,:,:]
        else:
            byAP_3D_array_2[ap_num-count-1,:,:]=byAP_3D_array[ap_num,:,:]
    return byAP_3D_array_1,byAP_3D_array_2

# input: (1)old_byAP_3D_array: the first part of fingerprint data
#        (2)new_byAP_3D_array: the second part of fingerprint data
# output: byAP_3D_array: fingerpint data conbine
# this is a duel function of 'byAP_3D_array_split()'
def byAP_3D_array_combine(old_byAP_3D_array,new_byAP_3D_array):
    if len(old_byAP_3D_array)==0:
        return new_byAP_3D_array
    if len(new_byAP_3D_array)==0:
        return old_byAP_3D_array
    byAP_3D_array=np.zeros((old_byAP_3D_array.shape[0]+new_byAP_3D_array.shape[0],\
                            old_byAP_3D_array.shape[1], old_byAP_3D_array.shape[2]))
    byAP_3D_array[:old_byAP_3D_array.shape[0],:,:]=old_byAP_3D_array[:,:,:]

    for new_ap in range(new_byAP_3D_array.shape[0]):
        byAP_3D_array[old_byAP_3D_array.shape[0]+new_ap, 0,0]=new_byAP_3D_array[new_ap,0,0]
        byAP_3D_array[old_byAP_3D_array.shape[0]+new_ap, 1:,0] = byAP_3D_array[old_byAP_3D_array.shape[0]-1,1:,0]
        for new_rp in range(1,len(new_byAP_3D_array[new_ap]),1):
            byAP_3D_array[old_byAP_3D_array.shape[0]+new_ap,new_byAP_3D_array[new_ap,new_rp,0],1:]=new_byAP_3D_array[new_ap,new_rp,1:]

    return byAP_3D_array

# input: (1)byRP_3D_array: a 3D arrary which contains crowdsource data organized according to different RP
#        (2)RP_dic: the RP dictionary associated with fingerprint database
# output: byRP_ap_dic: a dictionary which contains crowdsource data organized according to different RP
# this function is to transform the data structure of crowdsource data
def byRP_3D_array_2_byRP_dic(byRP_3D_array,RP_dic):
    byRP_ap_dic={}
    for rp_num in range(byRP_3D_array.shape[0]):
        rp_index=byRP_3D_array[rp_num,0,0];rp_x=RP_dic[rp_index][0];rp_y=RP_dic[rp_index][1]
        if (rp_x,rp_y) not in byRP_ap_dic:
            byRP_ap_dic[(rp_x,rp_y)]={}
        for ap_num in range(1,byRP_3D_array.shape[1],1):
            ap_index=byRP_3D_array[rp_num,ap_num,0]
            if ap_index==0:
                continue
            if ap_index not in byRP_ap_dic[(rp_x,rp_y)]:
                byRP_ap_dic[(rp_x,rp_y)][ap_index]=[]
            byRP_ap_dic[(rp_x,rp_y)][ap_index].append([byRP_3D_array[rp_num,ap_num,1],byRP_3D_array[rp_num,ap_num,2]])
    byRP_dic_2_byAP_dic(byRP_ap_dic)
    return byRP_ap_dic

# input: byRP_ap_dic: a dictionary which contains crowdsource data organized according to different RP
# output: byAP_rp_dic: a dictionary which contains crowdsource data organized according to different AP
# this function is to transform data structure of crowdsource data
def byRP_dic_2_byAP_dic(byRP_ap_dic):
    byAP_rp_dic={}
    for rp_xy in byRP_ap_dic:
        for ap_index in byRP_ap_dic[rp_xy]:
            if ap_index not in byAP_rp_dic:
                byAP_rp_dic[ap_index]={}
            byAP_rp_dic[ap_index][rp_xy]=byRP_ap_dic[rp_xy][ap_index]
    return byAP_rp_dic

# input: AP_crowd_prior_byRP: a dictionary which contains crowdsource signal data according to different RP
# output:(1)AP_crowd_prior_byAP: a dicitonary which contains crowdsource signal data according to different AP
#        (2)ap_list: a list AP index
# this funciton is to tranform data structure of fingerprint data
def AP_crowd_prior_byRP_2_byAP(AP_crowd_prior_byRP):
    AP_crowd_prior_byAP={}
    ap_list=[]
    for rp_xy in AP_crowd_prior_byRP:
        for ap_index in AP_crowd_prior_byRP[rp_xy]:
            if ap_index not in ap_list:
                ap_list.append(ap_index)
            if ap_index not in AP_crowd_prior_byAP:
                AP_crowd_prior_byAP[ap_index]={}
            AP_crowd_prior_byAP[ap_index][rp_xy]=AP_crowd_prior_byRP[rp_xy][ap_index]
    return AP_crowd_prior_byAP,ap_list

# input: (1)AP_crowd_prior_byRP: a dictionary which contains crowdsource signal data according to different RP
#        (2)output_txt_file_name: output txt file name
# output: a txt file
# this function is to tranform crowd data to txt file
def AP_crowd_prior_byRP_2_txt(AP_crowd_prior_byRP,output_txt_file_name):
    f1=open( output_txt_file_name[:len(output_txt_file_name)-4] +'_byRP_Num.txt','w')
    f2=open( output_txt_file_name[:len(output_txt_file_name)-4] +'_byRP_avgRSSI.txt','w')
    for rp_xy in AP_crowd_prior_byRP:
        ## cal mean rssi
        temp_prior=[]
        temp_crowd=[]
        for ap_index in AP_crowd_prior_byRP[rp_xy]:
            for prior_rssi in range(len(AP_crowd_prior_byRP[rp_xy][ap_index][0])):
                temp_prior.append(AP_crowd_prior_byRP[rp_xy][ap_index][0][prior_rssi][0])
            for crowd_rssi in range(len(AP_crowd_prior_byRP[rp_xy][ap_index][1])):
                temp_crowd.append(AP_crowd_prior_byRP[rp_xy][ap_index][1][crowd_rssi])
        line=str(rp_xy).strip('()').replace(' ','')
        f1.write( line +','+str(len(AP_crowd_prior_byRP[rp_xy])) + '\n')
        if not temp_prior:
            temp_prior.append(0)
        if not temp_crowd:
            temp_crowd.append(0)
        f2.write(line+','+ str(round(np.mean(temp_prior),1)) + ','+ str(round(np.mean(temp_crowd),1)) + '\n')
    f1.close()
    f2.close()
    return 0

# input: (1)AP_crowd_prior_byAP: a dicitonary which contains crowdsource signal data according to different AP
#        (2)AP_dic: a dictionary which contains AP mac address and AP index
#        (3)output_subfolder_name: output file folder
# output: a batch of txt files, each file contains crowdsource information of an AP
# this function is to tranform crowdsource data for each AP, it is similar with 'AP_crowd_prior_byRP_2_txt()'
def AP_crowd_prior_byAP_2_txt(AP_crowd_prior_byAP,AP_dic,output_subfolder_name):
    if not os.path.exists(output_subfolder_name):
        os.makedirs(output_subfolder_name)
    f=open( output_subfolder_name+'.txt','w' )
    for ap_index in AP_crowd_prior_byAP:
        line=AP_dic[ap_index]+':'
        for rp_xy in AP_crowd_prior_byAP[ap_index]:
            line=line+str(rp_xy).replace(' ','')+':['
            for prior_rssi in range(len(AP_crowd_prior_byAP[ap_index][rp_xy][0])):
                line=line+str(AP_crowd_prior_byAP[ap_index][rp_xy][0][prior_rssi][0])
            line=line+'],'+ str(AP_crowd_prior_byAP[ap_index][rp_xy][1]).replace(' ','')+';'
        f.write(line+'\n')
    f.close()

    for ap_index in AP_crowd_prior_byAP:
        if len(AP_crowd_prior_byAP[ap_index])<5:
            continue
        f_temp=open(output_subfolder_name+'/'+AP_dic[ap_index]+'.txt','w')
        for rp_xy in AP_crowd_prior_byAP[ap_index]:
            line=str(rp_xy).strip('()').replace(' ','')+','
            temp_prior=[]
            temp_crowd=[]
            for prior_rssi in range(len(AP_crowd_prior_byAP[ap_index][rp_xy][0])):
                temp_prior.append( AP_crowd_prior_byAP[ap_index][rp_xy][0][prior_rssi][0] )
            for crowd_rssi in range(len(AP_crowd_prior_byAP[ap_index][rp_xy][1])):
                temp_crowd.append(AP_crowd_prior_byAP[ap_index][rp_xy][1][crowd_rssi])
            if not temp_prior:
                temp_prior.append(0)
            if not temp_crowd:
                temp_crowd.append(0)
            line=line+ str(round(np.mean(temp_prior),1))+','+str(round(np.mean(temp_crowd),1))
            f_temp.write(line+'\n')
        f_temp.close()
    return 0

# input: null
# output: a number of folders created to accommondate plots,results, txt files
# this function is to create empty folders when the program starts
def creat_folders():
    #if not os.path.exists('Archive_crowd_signals'):
    #    os.makedirs('Archive_crowd_signals')
    if os.path.exists('Visualization_APdetection_byRP'):
        shutil.rmtree('Visualization_APdetection_byRP');os.makedirs('Visualization_APdetection_byRP')
    else:
        os.makedirs('Visualization_APdetection_byRP')

    if os.path.exists('Visualization_APdetection_byAP'):
        shutil.rmtree('Visualization_APdetection_byAP'); os.makedirs('Visualization_APdetection_byAP')
    else:
        os.makedirs('Visualization_APdetection_byAP')
    return 0

# input: (1)crowd_with_without_loc_txt_file: a txt file which contains batch of crowdsource signal vectors
#        (2)RP_dic,AP_dic: the RP/AP dictionary associated with fingerprint database
# output:(1)crowd_byRP_3D_array: a 3D array which contains all crowdsource data
#        (2)crowd_shared_ap_list: a list of AP index, which appear both in fingerprint database and crowdsource signals
#        (3)crowd_fresh_ap_list: a list of AP index, which appear only in crowdsource signals
# this function is to load crowdsource data, and identify two AP list
def crowd_byRP_txt_2_3D_array(crowd_with_without_loc_txt_file,RP_dic,_AP_dic):
    AP_dic=deepcopy(_AP_dic)
    crowd_file = open(crowd_with_without_loc_txt_file, 'r')
    total_crowd_RP_num = sum(1 for line in crowd_file)
    crowd_file.seek(0)

    estimated_max_AP_num_in_each_RP = A0.global_settings['estimated_max_AP_num_in_each_RP']
    data_num_in_each_AP = 3  # AP_index + RSSI+sd
    crowd_byRP_3D_array = np.zeros((total_crowd_RP_num,  estimated_max_AP_num_in_each_RP,  data_num_in_each_AP))

    crowd_shared_ap_list=[]
    crowd_fresh_ap_list=[]
    rp_count=-1
    for line in crowd_file:
        temp=re.split(',|:|\s',line)
        if temp=='':
            continue
        crowd_vector=np.zeros((estimated_max_AP_num_in_each_RP, data_num_in_each_AP))  # mac_index + rssi + sd (0)

        if str(temp[0]).replace('.','',1).isdigit(): # has location
            if temp[0]=='-1'and temp[1]=='-1':
                print('The crowdsource signal vector does not have correct location label,ignored..\n')
                continue
            rp_count += 1
            ###############
            x, y, d = nearest_fingerprint_xyd(temp[0],temp[1],temp[2], RP_dic)
            ###############
            crowd_vector[0,0] = RP_dic[(x,y,d)]
            #crowd_byRP_3D_array[rp_count] = crowd_vector
        else:
            print('The crowdsource signal vector does not have location label,ignored..\n')
            continue
        ap_no=0
        for t in range(0,len(temp),4):
            if temp[t]=='':
                break
            if t==0 and len(temp[t]) < A0.global_settings['ap_mac_address_string_length']: # not mac address
                continue

            #if temp[t] not in AP_dic: # ignore other APs
            #    crowd_fresh_ap_list.append(temp[t])
            #    continue
            #if (AP_dic[temp[t]] not in crowd_shared_ap_list):
            #    crowd_shared_ap_list.append(AP_dic[temp[t]])

            if temp[t] not in AP_dic:
                AP_dic[temp[t]]=len(AP_dic)/2+1
                AP_dic[AP_dic[temp[t]]]=temp[t]
                crowd_fresh_ap_list.append(AP_dic[temp[t]])
            else:
                if (AP_dic[temp[t]] not in crowd_shared_ap_list):
                    crowd_shared_ap_list.append(AP_dic[temp[t]])

            ap_no+=1
            if ap_no >= estimated_max_AP_num_in_each_RP:
                raise Exception('crowd_byRP_txt_2_3D_array()', 'crowd_byRP_3D_array is too small\n')
            crowd_vector[ap_no,0] = AP_dic[temp[t]]  # mac_index
            crowd_vector[ap_no,1] = rssi_negative_2_positive(float(temp[t+1])) # rssi
            crowd_vector[ap_no,2] = 0 # sd
        if ap_no==0:
            continue
        crowd_byRP_3D_array[rp_count] = crowd_vector

    crowd_file.close()

    if rp_count==-1:
        crowd_byRP_3D_array=crowd_byRP_3D_array[~(crowd_byRP_3D_array==0).all(1)]
        return crowd_byRP_3D_array,crowd_shared_ap_list,crowd_fresh_ap_list
    if rp_count+1<total_crowd_RP_num:
        crowd_byRP_3D_array=crowd_byRP_3D_array[:rp_count+1,:,:]
    return crowd_byRP_3D_array,crowd_shared_ap_list,crowd_fresh_ap_list,AP_dic

# input: (1) (x,y,d): a location, d means direction, but it is useless and just for data format
#        (2)prior_RP_dic: the RP dictionary associated with fingerprint database
# output:(nearest_x, nearest_y, nearest_d): the physical nearest RP
# this function is to find the physical nearest RP location, given a random location (x,y)
def nearest_fingerprint_xyd(x,y,d, prior_RP_dic):
    if (x,y,d) in prior_RP_dic:
        return x,y,d
    nearest_x=''
    nearest_y=''
    nearest_d=str(int(d))
    nearest_dis=500  # meter
    for xyd_index in range(1,int(len(prior_RP_dic)/2)+1, 1):
        dis = (math.sqrt((float(x) - float(prior_RP_dic[xyd_index][0]))**2 + \
                        ( float(y) - float(prior_RP_dic[xyd_index][1]))**2)) / A0.global_settings['one_meter_equals_pixels']
        if dis < nearest_dis:
            nearest_dis = dis
            nearest_x=prior_RP_dic[xyd_index][0]
            nearest_y=prior_RP_dic[xyd_index][1]

    if (nearest_x,nearest_y,nearest_d) not in prior_RP_dic:
        for i in range(1,5,1):
            nearest_d=str(int(i))
            if (nearest_x,nearest_y,nearest_d) in prior_RP_dic:
                break

    return nearest_x, nearest_y, nearest_d

# input: (1)crowd_rp_index: the index of an RP
#        (2)RP_dic: the RP dictionary associated with fingerprint database
# output: rp_index_list: a list of RP index which have same location (x,y)
def find_rp_index_at_same_loc(crowd_rp_index, RP_dic):
    x=RP_dic[crowd_rp_index][0]
    y=RP_dic[crowd_rp_index][1]
    rp_index_list=[]
    k=1
    while ((x,y,str(k)) in RP_dic) and k<=20:
        rp_index_list.append(RP_dic[x,y,str(k)])
        k+=1
    return rp_index_list
################################################################################################ utilities


################################################################################################ core functions,Part 1
# input: (1)crowd_byRP_3D_array: a 3D array which contains crowdsource signal vectors,data organized by different RP
#        (2)prior_byRP_3D_array: a 3D array which contains fingerprint signal vectors,data organized by different RP
#        (3)RP_dic,AP_dic: RP/AP dictionary associated with fingerprint data
# output:(1)crowd_changed_ap_list: a list of AP index,these APs have big changes after analysis
#        (2)a txt file which contains a list of APs that disappear in crowdsource vectors compared with fingerprint data
# this function is to identify a list of AP,which need to be reconstructed,
# several visualization modules can be enabled/disabled in this function
def crowd_prior_signal_analysis(crowd_byRP_3D_array,prior_byRP_3D_array,RP_dic,AP_dic):
    print ('Analyzing crowd signals...\n')
    ## analysis by physical location
    _,AP_prior_missing_byRP_dic,AP_prior_missing_changed_byRP_dic, AP_crowd_changed_byRP_dic,_ =\
        crowd_prior_signal_detection_byRP(crowd_byRP_3D_array,prior_byRP_3D_array,RP_dic)
    AP_crowd_prior_byRP_2_txt(AP_prior_missing_byRP_dic,'Visualization_APdetection_byRP/AP_prior_missing.txt')
    AP_crowd_prior_byRP_2_txt(AP_prior_missing_changed_byRP_dic, 'Visualization_APdetection_byRP/AP_prior_missing_changed.txt')
    AP_crowd_prior_byRP_2_txt(AP_crowd_changed_byRP_dic,'Visualization_APdetection_byRP/AP_crowd_changed.txt')
    if A0.global_settings['Visualization_APdetection']:
        map_file=A0.global_settings['background_floor_map']
        heatmap_byRP(map_file,'Visualization_APdetection_byRP')

    ## analysis by individual AP
    AP_prior_missing_byAP_dic,_ = AP_crowd_prior_byRP_2_byAP(AP_prior_missing_byRP_dic)
    AP_crowd_changed_byAP_dic,crowd_changed_ap_list  = AP_crowd_prior_byRP_2_byAP(AP_crowd_changed_byRP_dic)
    AP_crowd_prior_byAP_2_txt(AP_prior_missing_byAP_dic,AP_dic,'Visualization_APdetection_byAP/AP_prior_missing_byAP')
    AP_crowd_prior_byAP_2_txt(AP_crowd_changed_byAP_dic,AP_dic,'Visualization_APdetection_byAP/AP_crowd_changed_byAP')
    if A0.global_settings['Visualization_APdetection']:
        map_file=A0.global_settings['background_floor_map']
        heatmap_byAP(map_file,'Visualization_APdetection_byAP')
    return crowd_changed_ap_list,AP_prior_missing_byAP_dic

# input: (1)crowd_byRP_3D_array: a 3D array which contains crowdsource signal vectors,data organized by different RP
#        (2)prior_byRP_3D_array: a 3D array which contains fingerprint signal vectors,data organized by different RP
#        (3)RP_dic: RP dictionary associated with fingerprint data
# output:(1)AP_prior_full_byRP_dic: a dictionary which contains fingerprint data, organized by different RP
#        (2)AP_prior_missing_byRP_dic: a dictionary which contains missing fingerpint data
#        (3)AP_prior_missing_changed_byRP_dic:
#        (4)AP_crowd_changed_byRP_dic: a dictionary which contains changed AP signal data
#        (5)AP_crowd_fresh_byRP_dic: empty dictionary
# this function is to analyze AP signal missing, changing, according to crowdsource data and previous fingerprint data
def crowd_prior_signal_detection_byRP(crowd_byRP_3D_array,prior_byRP_3D_array,RP_dic):
    crowd_rp_dic={}
    for crowd_rp_num in range(crowd_byRP_3D_array.shape[0]):
        if crowd_byRP_3D_array[crowd_rp_num,0,0] not in RP_dic:
            continue
        rp_x= int(RP_dic[crowd_byRP_3D_array[crowd_rp_num,0,0]][0])
        rp_y= int(RP_dic[crowd_byRP_3D_array[crowd_rp_num,0,0]][1])
        if (rp_x,rp_y) not in crowd_rp_dic:
            crowd_rp_dic[(rp_x,rp_y)]={}
        for crowd_ap_num in range(1,crowd_byRP_3D_array.shape[1],1):
            crowd_ap=crowd_byRP_3D_array[crowd_rp_num,crowd_ap_num,0]
            crowd_rssi=int(rssi_positive_2_negative(crowd_byRP_3D_array[crowd_rp_num,crowd_ap_num,1]))
            if crowd_ap==0:
                continue
            if crowd_ap not in crowd_rp_dic[(rp_x,rp_y)]:
                crowd_rp_dic[(rp_x,rp_y)][crowd_ap]=[]
            crowd_rp_dic[(rp_x,rp_y)][crowd_ap].append(crowd_rssi)
    prior_rp_dic={}
    for piror_rp_num in range(prior_byRP_3D_array.shape[0]):
        rp_x=int(RP_dic[prior_byRP_3D_array[piror_rp_num,0,0]][0])
        rp_y=int(RP_dic[prior_byRP_3D_array[piror_rp_num,0,0]][1])
        #if (rp_x,rp_y) not in crowd_rp_dic:
        #    continue
        if (rp_x,rp_y) not in prior_rp_dic:
            prior_rp_dic[(rp_x,rp_y)]={}
        for prior_ap_num in range(1,prior_byRP_3D_array.shape[1],1):
            prior_ap=prior_byRP_3D_array[piror_rp_num,prior_ap_num,0]
            prior_rssi=int(rssi_positive_2_negative(prior_byRP_3D_array[piror_rp_num,prior_ap_num,1]))
            prior_sd=int(prior_byRP_3D_array[piror_rp_num,prior_ap_num,2])
            if prior_ap==0:
                continue
            if prior_ap not in prior_rp_dic[(rp_x,rp_y)]:
                prior_rp_dic[(rp_x,rp_y)][prior_ap]=[[],[]]
            prior_rp_dic[(rp_x,rp_y)][prior_ap][0].append([prior_rssi,prior_sd])

    ## to find missing ap list, to find changed ap list
    AP_prior_full_byRP_dic=deepcopy(prior_rp_dic)
    AP_prior_missing_byRP_dic=deepcopy(prior_rp_dic)
    AP_prior_missing_changed_byRP_dic=deepcopy(prior_rp_dic)
    AP_crowd_changed_byRP_dic_tmp=deepcopy(prior_rp_dic); crowd_changed_ap_list=[]
    AP_crowd_fresh_byRP_dic_tmp=deepcopy(crowd_rp_dic)

    for prior_rp_xy in prior_rp_dic:
        if prior_rp_xy not in crowd_rp_dic:
            del AP_prior_missing_byRP_dic[prior_rp_xy]
            del AP_prior_missing_changed_byRP_dic[prior_rp_xy]
            continue
        for prior_ap_index in prior_rp_dic[prior_rp_xy]:
            if prior_ap_index in crowd_rp_dic[prior_rp_xy]:
                del AP_prior_missing_byRP_dic[prior_rp_xy][prior_ap_index]
                del AP_crowd_fresh_byRP_dic_tmp[prior_rp_xy][prior_ap_index]
                AP_crowd_changed_byRP_dic_tmp[prior_rp_xy][prior_ap_index][1]=crowd_rp_dic[prior_rp_xy][prior_ap_index]
                AP_prior_missing_changed_byRP_dic[prior_rp_xy][prior_ap_index][1]=crowd_rp_dic[prior_rp_xy][prior_ap_index]
                ## to check crowd changed or not
                tmp_xy_prior_crowd_dic={}
                tmp_xy_prior_crowd_dic[prior_rp_xy]=\
                [prior_rp_dic[prior_rp_xy][prior_ap_index][0] ,crowd_rp_dic[prior_rp_xy][prior_ap_index]]
                if ap_rp_xy_check(tmp_xy_prior_crowd_dic): # changed
                    if prior_ap_index not in crowd_changed_ap_list:
                        crowd_changed_ap_list.append(prior_ap_index)
                else:
                    del AP_prior_missing_changed_byRP_dic[prior_rp_xy][prior_ap_index]

    ### to check completely missing AP at crowded rp_xy locations
    prior_ap_list_tmp=[]; crowd_ap_list_tmp=[]
    for crowd_rp_xy in crowd_rp_dic:
        for prior_ap_index in prior_rp_dic[crowd_rp_xy]:
            if prior_ap_index not in prior_ap_list_tmp:
                prior_ap_list_tmp.append(prior_ap_index)
        for crowd_ap_index in crowd_rp_dic[crowd_rp_xy]:
            if crowd_ap_index not in crowd_ap_list_tmp:
                crowd_ap_list_tmp.append(crowd_ap_index)
    prior_missing_ap_list=deepcopy(prior_ap_list_tmp)
    for prior_ap in prior_ap_list_tmp:
        if prior_ap in crowd_ap_list_tmp:
            prior_missing_ap_list.remove(prior_ap)

    ### to get crowd changed AP at crowded rp_xy locations
    AP_crowd_changed_byRP_dic=deepcopy(AP_crowd_changed_byRP_dic_tmp)
    for prior_rp_xy in AP_crowd_changed_byRP_dic_tmp:
        for prior_ap_index in AP_crowd_changed_byRP_dic_tmp[prior_rp_xy]:
            if prior_ap_index not in crowd_changed_ap_list:
                del AP_crowd_changed_byRP_dic[prior_rp_xy][prior_ap_index]
                if not AP_crowd_changed_byRP_dic[prior_rp_xy]:
                    del AP_crowd_changed_byRP_dic[prior_rp_xy]

    ### to get crowd fresh AP list
    AP_crowd_fresh_byRP_dic={}
    for crowd_rp_xy in AP_crowd_fresh_byRP_dic_tmp:
        if len(AP_crowd_fresh_byRP_dic_tmp[crowd_rp_xy])==0:
            continue
        if crowd_rp_xy not in AP_crowd_fresh_byRP_dic:
            AP_crowd_fresh_byRP_dic[crowd_rp_xy]={}
        for crowd_ap_index in AP_crowd_fresh_byRP_dic_tmp[crowd_rp_xy]:
            if crowd_ap_index not in AP_crowd_fresh_byRP_dic[crowd_rp_xy]:
                AP_crowd_fresh_byRP_dic[crowd_rp_xy][crowd_ap_index]=[[],[]]
            AP_crowd_fresh_byRP_dic[crowd_rp_xy][crowd_ap_index][1]=AP_crowd_fresh_byRP_dic_tmp[crowd_rp_xy][crowd_ap_index]

    return AP_prior_full_byRP_dic,AP_prior_missing_byRP_dic,AP_prior_missing_changed_byRP_dic,\
            AP_crowd_changed_byRP_dic,AP_crowd_fresh_byRP_dic

# input: prior_crowd_xy_dic: a dictionary which contains both prior fingerprint data
#                            and crowdsource data at specific locations
# output:prior_crowd_xy_dic_new: a dictionary which only contains changed AP signal data at specific locations
# this function is to analyze whether an AP's signal at a specific location has been changed or not
def ap_rp_xy_check(prior_crowd_xy_dic):
    prior_crowd_xy_dic_new=deepcopy(prior_crowd_xy_dic)
    for xy_num in prior_crowd_xy_dic:
        has_changed=1

        ################## if both prior and crowd signals are weak, treat it unchanged
        rssi_weak_threshold=A0.global_settings['AP_RP_rssi_weak_threshold'] #dBm
        prior_rssi_max=-A0.global_settings['rssi_negative_2_positive_shift']
        for prior_rssi_num in range(len(prior_crowd_xy_dic[xy_num][0])):
            prior_rssi=prior_crowd_xy_dic[xy_num][0][prior_rssi_num][0]
            if prior_rssi>prior_rssi_max and prior_rssi<0:
                prior_rssi_max=prior_rssi
        crowd_rssi_max=-A0.global_settings['rssi_negative_2_positive_shift']
        for crowd_rssi_num in range(len(prior_crowd_xy_dic[xy_num][1])):
            crowd_rssi=prior_crowd_xy_dic[xy_num][1][crowd_rssi_num]
            if crowd_rssi>crowd_rssi_max and crowd_rssi<0:
                crowd_rssi_max=crowd_rssi
        if prior_rssi_max<rssi_weak_threshold  and\
            crowd_rssi_max<rssi_weak_threshold:
            has_changed=0
        if has_changed==0: # to delete those RPs which do not change too much
            del prior_crowd_xy_dic_new[xy_num]
            continue
        ###################

        ###################
        for crowd_rssi_num in range(len(prior_crowd_xy_dic[xy_num][1])):
            crowd_rssi=prior_crowd_xy_dic[xy_num][1][crowd_rssi_num]
            if crowd_rssi==0:
                continue
            for prior_rssi_num in range(len(prior_crowd_xy_dic[xy_num][0])):
                prior_rssi=prior_crowd_xy_dic[xy_num][0][prior_rssi_num][0]
                if prior_rssi==0:
                    continue
                deltaRSSI=crowd_rssi-prior_rssi
                prior_sd=prior_crowd_xy_dic[xy_num][0][prior_rssi_num][1]
                if abs(deltaRSSI)<A0.global_settings['rssi_change_up_to_?_SD']*prior_sd:
                    has_changed=0
                    break
                if prior_sd==0 and abs(deltaRSSI)<A0.global_settings['rssi_change_up_to_?_dB']:
                    has_changed=0
                    break
            if has_changed==0:
                break
        if has_changed==0: # to delete those RPs which do not change too much
            del prior_crowd_xy_dic_new[xy_num]
            continue
        ####################
    return prior_crowd_xy_dic_new

# input: (1)bg_map_file: the background indoor floor map
#        (2)data_folder: the folder of files which contains AP signals at different locations
# output: heatmaps for each txt file
# this function is to visualize the average AP signal distribution on different locations
def heatmap_byRP(bg_map_file,data_folder):
    files=os.listdir(data_folder)
    #files=[f for f in os.listdir(data_folder) if os.path.isfile(f)]
    for file in files:
        if file[(len(file)-4):len(file)]=='.txt':
            print('generating heatmap:'+file+'\n')
            plot_heatmap(bg_map_file, data_folder+'/'+file, data_folder+'/'+ file[:(len(file)-4)])
    return 0

# input: (1)bg_map_file: the background indoor floor map
#        (2)data_folder: the folder of files, each file contains a single AP signal distribution at different locations
# output: heatmaps for each txt file
# this function is to visualize each AP signal distribution on different locations
def heatmap_byAP(bg_map_file,data_folder):
    subfolders=os.listdir(data_folder)
    for subfolder in subfolders:
        if subfolder[(len(subfolder)-4): len(subfolder)]!='.txt':
            if not os.path.exists(data_folder+'/'+subfolder+'_heatmap'):
                os.makedirs(data_folder+'/'+subfolder+'_heatmap')
            files=os.listdir( data_folder+'/'+ subfolder)
            for file in files:
                print('generating heatmap:'+file+'\n')
                plot_heatmap(bg_map_file,data_folder+'/'+subfolder+'/'+file,data_folder+'/'+subfolder+'_heatmap/'+file[:(len(file)-4)])
    return 0

# input: (1)bgimFile:the background indoor floor map
#        (2)txt: data file
#        (3)outImage: output image file name
# output: an image
# this function is to directly draw plot
def plot_heatmap(bgimFile, txt, outImage):
    try:
        mat = pd.read_csv(txt, header=None)
    except:
            return 0
    col_num=mat.shape[1]

    im = imread(bgimFile)
    image_height, image_width = im.shape[:2]
    num_x = image_width / 5
    num_y = num_x / (image_width / image_height)
    x = np.linspace(0, image_width, num_x)
    y = np.linspace(0, image_height, num_y)

    if col_num==3:
        x_list_3,y_list_3,z_list_3=rule_out_zero_rows(mat,3)
        # to draw or not?
        if len(x_list_3)<5:
            return 0
        figure = pl.figure(figsize=(10, 10), dpi=100)
        ax = pl.gca()
        try:
            z = ml.griddata(mat[0], mat[1], mat[2], x, y,interp='nn')
        except:
            return 0
        cs = ax.contourf(x, y, z, alpha=0.6, zorder=2, cmap=cm.jet)
        pl.colorbar(cs)
        pl.plot(mat[0], mat[1], '+', alpha=0.6, markersize=1.5, zorder=3)
        ax.imshow(im, alpha=0.3, zorder=0)
        pl.savefig(outImage)
        pl.clf()
        pl.close(figure)

    if col_num==4:
        x_list_3,y_list_3,z_list_3=rule_out_zero_rows(mat,3)
        x_list_4,y_list_4,z_list_4=rule_out_zero_rows(mat,4)

        # to draw or not?
        if len(x_list_3)<5:
            return 0
        ## to draw only one plot
        if len(x_list_4)<5:
            figure = pl.figure(figsize=(10, 10), dpi=100)
            ax = pl.gca()
            try:
                z = ml.griddata(mat[0], mat[1], mat[2], x, y,interp='nn')
            except:
                return 0
            cs = ax.contourf(x, y, z, alpha=0.6, zorder=2, cmap=cm.jet)
            pl.colorbar(cs)
            pl.plot(mat[0], mat[1], '+', alpha=0.6, markersize=1.5, zorder=3)
            ax.imshow(im, alpha=0.3, zorder=0)
            pl.savefig(outImage)
            pl.clf()
            pl.close(figure)
            return 0

        figure = pl.figure(figsize=(10, 10), dpi=100)
        ax = pl.subplot(211)
        try:
            z = ml.griddata(mat[0], mat[1], mat[2], x, y,interp='nn')
        except:
            return 0
        cs = ax.contourf(x, y, z, alpha=0.6, zorder=1, cmap=cm.jet)
        pl.colorbar(cs)
        pl.plot(mat[0], mat[1], '+', alpha=0.6, markersize=1.5, zorder=3)
        ax.imshow(im, alpha=0.3, zorder=0)

        ax = pl.subplot(212)
        x_list,y_list,z_list=rule_out_zero_rows(mat,4)
        #z = ml.griddata(mat[0], mat[1], mat[3], x, y,interp='nn')
        try:
            z = ml.griddata(x_list, y_list, z_list, x, y,interp='nn')
        except:
            return 0
        cs = ax.contourf(x, y, z, alpha=0.6, zorder=1, cmap=cm.jet)
        pl.colorbar(cs)
        pl.plot(mat[0], mat[1], '+', alpha=0.6, markersize=1.5, zorder=3)
        ax.imshow(im, alpha=0.3, zorder=0)

        pl.savefig(outImage)
        pl.clf()
        pl.close(figure)
    return 0

# this function is to rule out possible zero values of a 2D array before drawing plots,
# otherwise there may be execptions when plotting
def rule_out_zero_rows(mat,col_num):
    x_list=[];y_list=[];z_list=[]
    for i in range(mat.shape[0]):
        if mat.values[i,col_num-1]!=0:
            x_list.append(mat.values[i,0])
            y_list.append(mat.values[i,1])
            z_list.append(mat.values[i,col_num-1])
    return x_list,y_list,z_list
################################################################################################ core functions, Part 1

################################################################################################ core functions, Part 2
# input: (1)crowd_shared_byAP_3D_array: a 3D array which contains partial crowdsource data, organized by different AP
#        (2)prior_shared_byAP_3D_array: a 3D array which contains partial fingerprint data, organized by different AP
#        (3)AP_dic,RP_dic: RP/AP dictionary assocaited with fingerprint database
#        (4)recon_ap_list: a list of AP index, these APs are going to be reconstructed
# output: posterior_shared_byAP_3D_array: an updated version of fingerprint database,
#         which is a replacement of previous 'prior_shared_byAP_3D_array'
# this function is to reconstruct a selected number of AP
def reconstruction(crowd_shared_byAP_3D_array,prior_shared_byAP_3D_array,AP_dic,RP_dic,recon_ap_list):
    prior_shared_byAP_3D_array_tmp=deepcopy(prior_shared_byAP_3D_array)
    cycles=5
    posterior_shared_byAP_3D_array=[]
    for i in range(cycles):
        print('Reconstructing fingerprint...\n')
        posterior_shared_byAP_3D_array = crowd_byAP_reconstruction(crowd_shared_byAP_3D_array, prior_shared_byAP_3D_array_tmp, RP_dic, recon_ap_list)
        prior_shared_byAP_3D_array_tmp=deepcopy(posterior_shared_byAP_3D_array)

    ## to plot
    folder='Visualization_BCSupdating'
    if A0.global_settings['BCS_updating_visualization']:
        if os.path.exists(folder):
            shutil.rmtree(folder);os.makedirs(folder)
        else:
            os.makedirs(folder)
        prior_byRP_3D_array=fingerprints_byAP_2_byRP(prior_shared_byAP_3D_array)
        posterior_byRP_3D_array=fingerprints_byAP_2_byRP(posterior_shared_byAP_3D_array)
        prior_posterior_heatmap_byAP(folder,prior_byRP_3D_array,posterior_byRP_3D_array,AP_dic,RP_dic,recon_ap_list)

    return posterior_shared_byAP_3D_array

# input: (1)crowd_byAP_3D_array: a 3D array which contains crowdsource data, organized by different AP
#        (2)prior_byAP_3D_array: a 3D array which contains fingerprint data, organized by different AP
#        (3)RP_dic: RP dictionary assocaited with fingerprint database
#        (4)recon_ap_list: a list of AP index, these APs are going to be reconstructed
# output: posterior_byAP_3D_array: an updated version of fingerprint database,
#         which is a replacement of previous 'prior_byAP_3D_array'
# this function is to reconstruct a selected number of AP
def crowd_byAP_reconstruction(crowd_byAP_3D_array, prior_byAP_3D_array, RP_dic, recon_ap_list):
    if not recon_ap_list:
        print('No AP to recover..\n')
        return prior_byAP_3D_array

    posterior_byAP_3D_array=deepcopy(prior_byAP_3D_array)
    for crowd_ap_num in range(len(crowd_byAP_3D_array)):
        if crowd_byAP_3D_array[crowd_ap_num,0,0] not in recon_ap_list:
            continue
        for prior_ap_num in range(len(prior_byAP_3D_array)):
            if prior_byAP_3D_array[prior_ap_num,0,0] == crowd_byAP_3D_array[crowd_ap_num,0,0]:
                crowd_deltaY_Phi_2D_array = crowd_deltaY_Phi_v2(crowd_byAP_3D_array[crowd_ap_num], prior_byAP_3D_array[prior_ap_num], RP_dic)
                if crowd_deltaY_Phi_2D_array.shape[0]<=1: # no useful crowd data
                    break
                posterior_byAP_3D_array[prior_ap_num] = BCS_vb(crowd_deltaY_Phi_2D_array, prior_byAP_3D_array[prior_ap_num])
                break

    return posterior_byAP_3D_array

# input: (1)crowd_Y_2D_array: crowdsource signal data for an individual AP
#        (2)prior_X_2D_array: fingerprint signal data for the same individual AP
#        (3)RP_dic: the RP dictionary associated with fingerprint database
# output: crowd_deltaY_Phi_2D_array: the sensing matrix for crowdsource data
# this function is to get the sensing matrix for crowdsource data of an single AP,
# this is for BCS reconstruction algorithm
def crowd_deltaY_Phi_v2(_crowd_Y_2D_array, prior_X_2D_array, RP_dic):
    crowd_Y_2D_array=deepcopy(_crowd_Y_2D_array)
    temp_M = crowd_Y_2D_array.shape[0]
    temp_N = prior_X_2D_array.shape[0]
    crowd_deltaY_Phi_2D_array = np.zeros((temp_M,len(RP_dic)/2 +2))
    for i in range( int(len(RP_dic)/2)): # initilize the rp_index
        crowd_deltaY_Phi_2D_array[0,i+2] = i+1

    useful_crowd_rp_count=0
    for crowd_rp_num in range(1,temp_M,1):
        crowd_rp_index=crowd_Y_2D_array[crowd_rp_num,0]
        rp_index_list=find_rp_index_at_same_loc(crowd_rp_index,RP_dic)
        min_delta_rssi=130;min_delta_rssi_rp_index=0
        for i in range(len(rp_index_list)):
            tp=crowd_Y_2D_array[crowd_rp_num,1]-prior_X_2D_array[rp_index_list[i],1]
            if abs(tp)<abs(min_delta_rssi):
                min_delta_rssi=tp
                min_delta_rssi_rp_index=rp_index_list[i]
        crowd_Y_2D_array[crowd_rp_num,0]=min_delta_rssi_rp_index
        if crowd_Y_2D_array[crowd_rp_num,1]==0 and min_delta_rssi==0:
            continue # useless crowd signal

        useful_crowd_rp_count+=1
        crowd_deltaY_Phi_2D_array[useful_crowd_rp_count, 0] = crowd_Y_2D_array[crowd_rp_num,0]
        crowd_deltaY_Phi_2D_array[useful_crowd_rp_count, 1] = crowd_Y_2D_array[crowd_rp_num,1] - prior_X_2D_array[min_delta_rssi_rp_index,1]
        for prior_point2 in range(1,temp_N,1):
            crowd_deltaY_Phi_2D_array[useful_crowd_rp_count, prior_point2+1]=\
            radial_basis_functional_kernel_v2(prior_X_2D_array[min_delta_rssi_rp_index],prior_X_2D_array[prior_point2],RP_dic)


    if useful_crowd_rp_count+1<=temp_M:
        crowd_deltaY_Phi_2D_array=crowd_deltaY_Phi_2D_array[:useful_crowd_rp_count+1]

    crowd_deltaY_Phi_2D_array_final=np.zeros((crowd_deltaY_Phi_2D_array.shape[0],crowd_deltaY_Phi_2D_array.shape[1]))
    crowd_deltaY_Phi_2D_array_final[0]=crowd_deltaY_Phi_2D_array[0]
    useful_rp_final_count=0
    for i in range(1,crowd_deltaY_Phi_2D_array.shape[0],1):
        if crowd_deltaY_Phi_2D_array[i,0]==0:
            continue
        rp_index_i=crowd_deltaY_Phi_2D_array[i,0]
        if i==crowd_deltaY_Phi_2D_array.shape[0]-1:
            kk=0
        for j in range(i+1,crowd_deltaY_Phi_2D_array.shape[0],1):
            if crowd_deltaY_Phi_2D_array[j,0]==0:
                continue
            rp_index_j=crowd_deltaY_Phi_2D_array[j,0]
            if rp_index_i==rp_index_j:
                if abs(crowd_deltaY_Phi_2D_array[i,1])>abs(crowd_deltaY_Phi_2D_array[j,1]):
                    crowd_deltaY_Phi_2D_array[i,1]=crowd_deltaY_Phi_2D_array[j,1]
                crowd_deltaY_Phi_2D_array[j,0]=0
                crowd_deltaY_Phi_2D_array[j,1]=0
        useful_rp_final_count+=1
        crowd_deltaY_Phi_2D_array_final[useful_rp_final_count]=crowd_deltaY_Phi_2D_array[i]
    if useful_rp_final_count+1<=crowd_deltaY_Phi_2D_array.shape[0]:
        crowd_deltaY_Phi_2D_array_final=crowd_deltaY_Phi_2D_array_final[:useful_rp_final_count+1]

    return crowd_deltaY_Phi_2D_array_final

# input: (1)point1: a signal point,i.e. location (x,y) + rssi value
#        (2)point2: a signal point,i.e. location (x,y) + rssi value
# output: similarity: the similarity between the above two signal points
# this function is to cacluate the similarity between two signal points in fingeprint database
def radial_basis_functional_kernel_v2(point1,point2,RP_dic):
    sigma=A0.global_settings['RBF_sigma']
    weight=A0.global_settings['RBF_weight']
    ######### hallway data
    sigma=1.25
    weight=0.85
    ######### corridor data -> APpower ajustment, loc_noise=0
    #sigma=1.65
    #weight=0.6


    ######### corridor data -> APpower ajustment, loc_noise=1
    #sigma=1.65
    #weight=0.6
    ######### corridor data -> APlocation ajustment
    #sigma=1.75
    #weight=0.8
    ######### corridor data -> wherame test, 50% crowd data
    #sigma=1.4
    #weight=0.7
    ######### corridor data -> wherame test, 25% crowd data
    #sigma=1.4
    #weight=0.75
    ######### corridor data -> wherame test, 12% crowd data
    #sigma=1.5
    #weight=0.8
    ######### spacious data -> 12% crowd data
    #sigma=0.2
    #weight=0.4
    #if point1[1]*point2[1]==0:
    #    return 0

    ######### corridor data -> wherame test, 12% crowd data
    #sigma=1.5
    #weight=0.8

    distance= (np.sqrt((float(RP_dic[point1[0]][0])-float(RP_dic[point2[0]][0]))**2 +\
                        (float(RP_dic[point1[0]][1])-float(RP_dic[point2[0]][1]))**2) )/A0.global_settings['one_meter_equals_pixels']
    deltaRSSI=np.abs(point1[1] - point2[1])
    list1=[distance*weight, deltaRSSI*(1-weight)]
    list2=[0,0]
    eucd = spatial.distance.euclidean(list1, list2)
    similarity=math.exp(- eucd**2 / (2*sigma**2))
    #if similarity>0.8:
    #    print('point1[0]='+str(point1[0])+',point2[0]='+str(point2[0])+',sim='+str(round(similarity,2))+'\n')
    return round(similarity,2)

# input: (1)crowd_deltaY_Phi_2D_array: the crowdsource signal data and sensing matrix for an AP
#        (2)prior_X_2D_array: the fingeprint data of that same AP
# output: posterior_byAP_3D_array: an updated version of signal map of the same AP
# this function is to update signal map of an AP using BCS algorithm
def BCS_vb(crowd_deltaY_Phi_2D_array, prior_X_2D_array):
    #################
    crowd_NonZero_prior_NonZero_rp_list=[]
    crowd_Zero_prior_NonZero_rp_list=[]
    prior_Zero_rp_list=[]
    for k in range(1,crowd_deltaY_Phi_2D_array.shape[0],1):
        for j in range(1,prior_X_2D_array.shape[0],1):
            if prior_X_2D_array[j,0]==crowd_deltaY_Phi_2D_array[k,0]:
                if prior_X_2D_array[j,1]==0:
                    prior_Zero_rp_list.append( crowd_deltaY_Phi_2D_array[k,0] )
                    break
                if prior_X_2D_array[j,1]+crowd_deltaY_Phi_2D_array[k,1]==0:
                    crowd_Zero_prior_NonZero_rp_list.append(crowd_deltaY_Phi_2D_array[k,0] )
                    break
                if prior_X_2D_array[j,1]+crowd_deltaY_Phi_2D_array[k,1]!=0:
                    crowd_NonZero_prior_NonZero_rp_list.append(crowd_deltaY_Phi_2D_array[k,0])
                    break
    #################
    deltaY=np.zeros((len(crowd_NonZero_prior_NonZero_rp_list),1))
    Phi=np.zeros((len(crowd_NonZero_prior_NonZero_rp_list),crowd_deltaY_Phi_2D_array.shape[1]-2))
    count=0
    for k in range(1,crowd_deltaY_Phi_2D_array.shape[0],1):
        if crowd_deltaY_Phi_2D_array[k,0] in crowd_NonZero_prior_NonZero_rp_list:
            deltaY[count,0]=crowd_deltaY_Phi_2D_array[k,1]
            Phi[count,:]=crowd_deltaY_Phi_2D_array[k,2:]
            count+=1
    #deltaY =deepcopy(crowd_deltaY_Phi_2D_array[1:,1].reshape(crowd_deltaY_Phi_2D_array.shape[0]-1,1))
    #Phi = crowd_deltaY_Phi_2D_array[1:,2:]
    #################

    for i in range(len(deltaY)):
        #t=np.sum(Phi[i,:])
        #print('np.sum(Phi[i,:])='+str(t)+'\n')
        deltaY[i,0]=deltaY[i,0]*np.sum(Phi[i,:])
    #array_2_txt(deltaY,'deltaY.txt')
    #array_2_txt(Phi, 'Phi.txt')
    #################
    if len(crowd_NonZero_prior_NonZero_rp_list)!=0:

        plotflag, tol, niter=0, 1e-2, 1
        #a0, b0, c0, d0 = 1e-6, 1e-6, 1e-6, 1e-6  #hyperpara
        a0, b0, c0, d0 = 1, 4, 4, 6
        M = Phi.shape[0] # number of CS measurements
        N = Phi.shape[1] # sparse signal length
        # Initialization
        MU_theta = np.zeros((N,1))
        VAR_theta_diag = d0/c0 * np.ones((N, 1))
        a=a0
        b=b0
        c=c0*np.ones((N,1))
        d=d0*np.ones((N,1))
        #precomputation
        Phiv= np.dot(Phi.transpose() , deltaY)
        # VB iteration
        iter_i = 0
        curr_lb=2*tol+1
        last_lb=1
        while iter_i < niter and abs((curr_lb-last_lb)/last_lb) > tol :
            iter_i += 1
            #(1) alpha
            c=c0+0.5 # value
            d=d0+0.5*(VAR_theta_diag + np.power(MU_theta, 2))  # vector
            #(2) theta
            invA=np.divide(d, c)
            CC = np.multiply (Phi , ( np.matlib.repmat(invA.transpose(), M, 1) ) )
            DD =  np.linalg.inv(  (np.eye(M) * b/a) + np.dot(CC, Phi.transpose())  )
            MU_theta = a/b * (  np.multiply(invA, Phiv)  -  np.dot( CC.transpose() , np.dot( DD , np.dot(CC, Phiv) ) )  )
            VAR_theta_diag = invA  -  (  np.sum(  np.multiply( (  np.dot(DD , CC)  ), CC ) ,  axis=0) ).transpose().reshape(N,1)
            #(3) alpha0
            alpha0inv = b/a
            res = alpha0inv * ( np.dot(DD , deltaY))
            tmp = M * alpha0inv - alpha0inv * alpha0inv *  np.sum(  np.diag( DD)  , axis=0  )
            a = a0 + 0.5*M
            b = b0 + 0.5*( np.dot( res.transpose(), res) + tmp)
            #(4) lower bound
            J1= -M/2 * np.log(2*np.pi) + M/2 * ( special.psi(a) - np.log(b) )  -  (a/b) * (b-b0)
            J2= -N/2 * np.log(2*np.pi)  +  0.5* np.sum(  special.psi(c) - np.log(d) - np.multiply(  np.divide(c, d), (VAR_theta_diag + np.power(MU_theta, 2) ) ) , axis=0  )
            PP, LL, UU = linalg.lu(DD)
            logdetVAR_theta =  np.real (  np.sum( np.log(np.diag(UU)), axis=0)  )  +  np.sum( np.log(invA) , axis=0)  +  M* np.log(b/a)
            J3= N/2 * ( np.log(2*np.pi)+1)  + 0.5 * logdetVAR_theta
            J4= - (  a*np.log(b)-a0*np.log(b0) - special.gammaln(a)  + special.gammaln(a0)  + (a-a0) * (special.psi(a) - np.log(b) )  - a * (1 - b0/b)  )
            J5= - np.sum(  np.multiply(c, np.log(d))  - np.multiply(c0, np.log(d0))  - special.gammaln(c) + special.gammaln(c0) + np.multiply( (c-c0), (special.psi(c) -np.log(d)) )  -  np.multiply(c,  (1- np.divide(d0, d)) ) ,   axis=0)
            J=J1+J2+J3+J4+J5
            last_lb=curr_lb
            curr_lb = J
            print (' Iteration = ' , iter_i , '  VB lower bound =  ' , J , '\n')

        # output
        #array_2_txt(MU_theta,'MU_theta.txt')
        #array_2_txt(VAR_theta_diag,'VAR_theta_diag.txt')
        #delta_X = MU_theta
        #detla_X_sd = VAR_theta_diag

        #######################
        # to test, without bcs reconstruction
        #for i in range(N):
        #    MU_theta[i,0]=0
        # to check
        for i in range(M):
            MU_theta[ crowd_deltaY_Phi_2D_array[i+1,0]-1 ,0] = crowd_deltaY_Phi_2D_array[i+1,1]
            #VAR_theta_diag[crowd_deltaY_Phi_2D_array[i+1,0]-1 ,0] =
        #######################


    #######################
    posterior_X_2D_array=deepcopy(prior_X_2D_array)
    if len(crowd_NonZero_prior_NonZero_rp_list)!=0:
        for i in range(Phi.shape[1]):
            posterior_X_2D_array[i+1,1]=prior_X_2D_array[i+1,1]+ round( MU_theta[i,0],3)
            if VAR_theta_diag[i,0] != d0/c0:
                posterior_X_2D_array[i+1,2] = round( VAR_theta_diag[i,0],3)

    for i in range(1,crowd_deltaY_Phi_2D_array.shape[0],1):
        posterior_X_2D_array[crowd_deltaY_Phi_2D_array[i,0],1]=prior_X_2D_array[crowd_deltaY_Phi_2D_array[i,0],1]+crowd_deltaY_Phi_2D_array[i,1]

        ###prior_Zero_rp_list
        #if prior_X_2D_array[i+1,1] ==0:
        #    for j in range(1,crowd_deltaY_Phi_2D_array.shape[0],1):
        #        if crowd_deltaY_Phi_2D_array[j,0]==prior_X_2D_array[i+1,0]:
        #            posterior_byAP_3D_array[i+1,1]=crowd_deltaY_Phi_2D_array[j,1]
        #            break
        #    continue
        ###crowd_NonZero_prior_NonZero_rp_list
        #if prior_X_2D_array[i+1,0] in crowd_NonZero_prior_NonZero_rp_list:
        #    posterior_byAP_3D_array[i+1,1]=prior_X_2D_array[i+1,1]+ round( MU_theta[i,0],3)
        #    if VAR_theta_diag[i,0] != d0/c0:
        #        posterior_byAP_3D_array[i+1,2] = round( VAR_theta_diag[i,0],3)
        #    continue
        ###crowd_Zero_prior_NonZero_rp_list
        #if prior_X_2D_array[i+1,0] in crowd_Zero_prior_NonZero_rp_list:
        #    continue
    #######################

    return posterior_X_2D_array

# input: crowd_byRP_3D_array: a 3D array which contains crowdsource signal vectors,data organized by different RP
# output: crowd_byRP_3D_array_mean: a 3D array which contains crowdsource signal vectors,data organized by different RP
# if there are multiple crowdsource vectors that are sampled at the same location (or similar locations),
# this function is to cluster such multiple vectors and caculate mean RSSI value for each AP
def crowd_byRP_3D_array_cal_mean(crowd_byRP_3D_array,RP_dic):
    crowd_RP_AP_dic={}
    for crowd_rp in range(crowd_byRP_3D_array.shape[0]):
        crowd_rp_index=crowd_byRP_3D_array[crowd_rp,0,0]
        if crowd_rp_index==0:
            continue
        rp_list_at_same_loc=find_rp_index_at_same_loc(crowd_rp_index, RP_dic)
        if crowd_rp_index not in crowd_RP_AP_dic:
            crowd_RP_AP_dic[crowd_rp_index]={}
        for crowd_rp_t in range(crowd_rp,crowd_byRP_3D_array.shape[0],1):
            if not rp_list_at_same_loc:
                break
            if crowd_byRP_3D_array[crowd_rp_t,0,0] in rp_list_at_same_loc:
                rp_list_at_same_loc.remove(crowd_byRP_3D_array[crowd_rp_t,0,0])
                crowd_byRP_3D_array[crowd_rp_t,0,0]=0
                for crowd_ap in range(1,crowd_byRP_3D_array.shape[1],1 ):
                    crowd_ap_index=crowd_byRP_3D_array[crowd_rp_t,crowd_ap,0]
                    if crowd_ap_index==0:
                        continue
                    if crowd_ap_index not in crowd_RP_AP_dic[crowd_rp_index]:
                        crowd_RP_AP_dic[crowd_rp_index][crowd_ap_index]=[]
                    crowd_RP_AP_dic[crowd_rp_index][crowd_ap_index].append(crowd_byRP_3D_array[crowd_rp_t,crowd_ap,1])

    crowd_rp_num=len(crowd_RP_AP_dic)
    estimated_max_AP_num_in_each_RP = A0.global_settings['estimated_max_AP_num_in_each_RP']
    data_num_in_each_AP = 3  # AP_index + RSSI+sd
    crowd_byRP_3D_array_mean=np.zeros([crowd_rp_num,estimated_max_AP_num_in_each_RP,data_num_in_each_AP])
    rp_count=-1
    for crowd_rp in crowd_RP_AP_dic:
        rp_count+=1
        crowd_byRP_3D_array_mean[rp_count,0,0]=crowd_rp
        ap_count=0
        for crowd_ap in crowd_RP_AP_dic[crowd_rp]:
            ap_count+=1
            crowd_byRP_3D_array_mean[rp_count,ap_count,0]=crowd_ap
            crowd_byRP_3D_array_mean[rp_count,ap_count,1]=round(np.mean(crowd_RP_AP_dic[crowd_rp][crowd_ap]),1)
            crowd_byRP_3D_array_mean[rp_count,ap_count,2]=round(np.std(crowd_RP_AP_dic[crowd_rp][crowd_ap]),1)

    return crowd_byRP_3D_array_mean

# input: (1)prior_byAP_3D_array:a 3D array which contains fingerprint data, organized by different AP
#        (2)AP_prior_missing_byAP_dic: a dictionary which contains missing fingerpint data
#        (3)RP_dic: RP dictionary associated with fingerprint database
# output: prior_byAP_3D_array: a 3D array which contains fingerprint data, organized by different AP
# this function is to remove those signal points which no longer appear on RP in crowdsource data
def prior_crowd_missing_ap_excluded_4_updating(_prior_byAP_3D_array,AP_prior_missing_byAP_dic,RP_dic):
    prior_byAP_3D_array=deepcopy(_prior_byAP_3D_array)
    if len(prior_byAP_3D_array)==0 or len(AP_prior_missing_byAP_dic)==0:
        return prior_byAP_3D_array
    for prior_ap_num in range(prior_byAP_3D_array.shape[0]):
        prior_ap_index=prior_byAP_3D_array[prior_ap_num,0,0]
        if prior_ap_index in AP_prior_missing_byAP_dic:
            for rp_xy in AP_prior_missing_byAP_dic[prior_ap_index]:
                # to confirm the signal is lost at such location
                weakest_rssi=0
                for rssi_no in range(len(AP_prior_missing_byAP_dic[prior_ap_index][rp_xy][0])):
                    if AP_prior_missing_byAP_dic[prior_ap_index][rp_xy][0][rssi_no][0]<weakest_rssi:
                        weakest_rssi=AP_prior_missing_byAP_dic[prior_ap_index][rp_xy][0][rssi_no][0]
                if weakest_rssi<=A0.global_settings['AP_RP_rssi_weak_threshold']:
                    continue
                # the prior signal is strong, but confirmed to be lost
                i=1; rp_index_list=[]
                while (str(rp_xy[0]),str(rp_xy[1]),str(i)) not in RP_dic and i<20:
                    i+=1
                if i>=20:
                    continue
                rp_index_list=find_rp_index_at_same_loc(RP_dic[(str(rp_xy[0]),str(rp_xy[1]),str(i))],RP_dic )
                for rp_index in rp_index_list:
                    prior_byAP_3D_array[prior_ap_num,rp_index,1]=0
                    prior_byAP_3D_array[prior_ap_num,rp_index,2]=0

    return prior_byAP_3D_array

# input: (1)folder: output file folder
#        (2)prior_byRP_3D_array: privous fingerprint database
#        (3)posterior_byRP_3D_array: an updated version of fingerprint database
#        (4)AP_dic,RP_dic: the RP/AP dictionary associated with fingerprint database
#        (5)recon_ap_list: a list of AP index
# output: a list of heatmaps which show the prior and updated signal maps for each updated AP
def prior_posterior_heatmap_byAP(folder,prior_byRP_3D_array,posterior_byRP_3D_array,AP_dic,RP_dic,recon_ap_list):
    if len(prior_byRP_3D_array)==0 or len(posterior_byRP_3D_array)==0 or len(recon_ap_list)==0:
        return 0
    prior_byAP_rp_dic = byRP_dic_2_byAP_dic( byRP_3D_array_2_byRP_dic(prior_byRP_3D_array,RP_dic) )
    posterior_byAP_rp_dic = byRP_dic_2_byAP_dic( byRP_3D_array_2_byRP_dic(posterior_byRP_3D_array,RP_dic) )
    ##### creat txt files
    for ap_index in prior_byAP_rp_dic:
        if ap_index not in recon_ap_list:
            continue
        if ap_index not in posterior_byAP_rp_dic:
            continue
        f=open(folder+'/'+AP_dic[ap_index]+'.txt','w')
        for rp_xy in prior_byAP_rp_dic[ap_index]:
            if rp_xy not in posterior_byAP_rp_dic[ap_index]:
                continue
            line=str(rp_xy).strip('()''').replace(' ','').replace("'",'')
            # cal prior rssi mean
            prior_rssi=[];posterior_rssi=[]
            for i in range(len(prior_byAP_rp_dic[ap_index][rp_xy])):
                prior_rssi.append(prior_byAP_rp_dic[ap_index][rp_xy][i][0])
            for j in range(len(posterior_byAP_rp_dic[ap_index][rp_xy])):
                posterior_rssi.append(posterior_byAP_rp_dic[ap_index][rp_xy][j][0])
            line=line+','+ str(round(rssi_positive_2_negative(np.mean(prior_rssi)),1))+','+\
                str(round(rssi_positive_2_negative(np.mean(posterior_rssi)),1))
            f.write(line+'\n')
        f.close()
    ##### to plot
    if os.path.exists(folder+'/heatmap'):
            shutil.rmtree(folder+'/heatmap');os.makedirs(folder+'/heatmap')
    else:
        os.makedirs(folder+'/heatmap')
    files=os.listdir(folder)
    for file in files:
        if file[len(file)-4:]!='.txt':
            continue
        print('generating heatmap:'+file+'\n')
        map_file=A0.global_settings['background_floor_map']
        plot_heatmap(map_file,folder+'/'+file,folder+'/heatmap/'+file[:(len(file)-4)])
    return 0
################################################################################################ core functions, Part 2


################################################################################################ API functions
# input: crowd_fingerprint_file: the file name of crowdsource signal data
# output: an updated version of fingerprint file, which is replacement of previous fingerprint file
# this function is to update fingerprint data of a selected number of APs(changed APs),
# several visualization moudels can be disabled and enabled in this function
def updating_by_BatchVectorTxtFile(crowd_fingerprint_file):
    prior_fingerprint_file = get_newest_txt_file(A0.global_settings['fingerprint_txt_file_prefix'])
    if not prior_fingerprint_file:
        print('Fingerprint file is missing...\n')
        return -1
    else:
        print('Fingerprint file is: '+prior_fingerprint_file+'\n')
    creat_folders()
    print('Processing fingerprint...\n')
    AP_dic={}; RP_dic={}
    prior_byRP_3D_array, AP_dic, RP_dic  = fingerprints_byRP_txt_2_3D_array(prior_fingerprint_file, AP_dic,RP_dic)
    prior_byAP_3D_array = fingerprints_byRP_2_byAP(prior_byRP_3D_array)

    ################################################ to continuous reconstruct
    while 1:
        newest_crowd_file = get_newest_txt_file(crowd_fingerprint_file)
        if not newest_crowd_file:
            print('Waiting for new crowdsource file...\n')
            continue
        else:
            print('The new crowdsouce file is:'+newest_crowd_file+'\n')
        print('Processing crowd signals...\n')
        crowd_byRP_3D_array,_,crowd_fresh_ap_list,AP_dic_fresh_included = crowd_byRP_txt_2_3D_array(newest_crowd_file,RP_dic,AP_dic)

        ################################################ signal analysis for visualization
        if crowd_byRP_3D_array.shape[0]==0:
            print('crowdsource signals are useless..\n')
            continue
        crowd_changed_ap_list,AP_prior_missing_byAP_dic=crowd_prior_signal_analysis(crowd_byRP_3D_array,prior_byRP_3D_array,RP_dic,AP_dic)

        ################################################ to process crowd signals
        crowd_byRP_3D_array = crowd_byRP_3D_array_cal_mean(crowd_byRP_3D_array,RP_dic)
        crowd_byAP_3D_array = fingerprints_byRP_2_byAP( crowd_byRP_3D_array )
        crowd_changed_byAP_3D_array,_ = byAP_3D_array_split(crowd_byAP_3D_array,crowd_changed_ap_list)
        crowd_fresh_byAP_3D_array,_ = byAP_3D_array_split(crowd_byAP_3D_array,crowd_fresh_ap_list)
        prior_changed_byAP_3D_array,prior_old_byAP_3D_array = byAP_3D_array_split(prior_byAP_3D_array,crowd_changed_ap_list)

        ################################################ to reconstruct
        recon_ap_list=crowd_changed_ap_list   #recon_ap_list=crowd_shared_ap_list
        posterior_changed_byAP_3D_array = reconstruction(crowd_changed_byAP_3D_array,prior_changed_byAP_3D_array,AP_dic,RP_dic,recon_ap_list)

        ################################################ updated prior signals
        prior_byAP_3D_array = byAP_3D_array_combine(posterior_changed_byAP_3D_array,prior_old_byAP_3D_array)
        if A0.global_settings['crowd_freshed_ap_included_for_updating']:
            AP_dic=AP_dic_fresh_included
            prior_byAP_3D_array = byAP_3D_array_combine(prior_byAP_3D_array,crowd_fresh_byAP_3D_array)
        if A0.global_settings['prior_missing_ap_excluded_for_updating']:
            prior_byAP_3D_array=prior_crowd_missing_ap_excluded_4_updating(prior_byAP_3D_array,AP_prior_missing_byAP_dic,RP_dic)

        prior_byRP_3D_array=fingerprints_byAP_2_byRP(prior_byAP_3D_array)

        ################################################ output
        fingerprints_byAP_3D_array_2_txt(prior_byAP_3D_array,AP_dic,RP_dic,A0.global_settings['fingerprint_txt_file_prefix'])

        #now=dt.datetime.now()
        #try:
        #    move(newest_crowd_file, 'Archive_crowd_signals/'+crowd_fingerprint_file[0:(len(crowd_fingerprint_file)-4)] + '_'+str(now)[0:10]+'-'+str(now.time())[0:8].replace(':','-')+'.txt')
        #except:
        #    print('move file failure..\n')
        break
    return 0

# input: SingleVectorString: a single crowdsource signal vector which has location (x,y)
# output: an updated version of fingerprint file, which is replacement of previous fingerprint file
# this function is to update fingerprint data of a selected number of APs(changed APs)
def updating_by_SingleVectorString(SingleVectorString):
    prior_fingerprint_file = get_newest_txt_file(A0.global_settings['fingerprint_txt_file_prefix'])
    if not prior_fingerprint_file:
        print('Fingerprint file is missing...\n')
        return -1
    f = open('temp_singlevector', 'w')
    f.write( SingleVectorString + '\n')
    f.close()
    updating_by_BatchVectorTxtFile('temp_singlevector')
    return 0
################################################################################################ API functions


################################################################################################ Demo
if __name__ == '__main__':
    ### the following lines are for function flow visulization
    #config = Config(max_depth=5)
    #config.trace_filter = GlobbingFilter(exclude=['pycallgraph.*','_*.*',])
    #graphviz = GraphvizOutput(output_file='A3_module_function_flow.png')
    #with PyCallGraph(output=graphviz, config=config):
    ### the above lines are for function flow visulization

        # Demo to show how to call API function
        # the input file is from output of module: A2_subset_localization.py
        # the output file is like: 'p_all_2016-07xxxx.txt',which is updated version of 'p_all.txt'
        #crowd_fingerprint_file ='crowd_targets_true.txt'
        crowd_fingerprint_file ='crowd_targets_raw_calibrated_localized.txt'
        updating_by_BatchVectorTxtFile(crowd_fingerprint_file)

        #SingleVectorString='3705,1080,1,1 2c5d93087159:-74,0.000000,1.00 2c5d93c87158:-72,0.000000,1.00 0228c8d8dcd2:-77,0.000000,1.00 '
        #updating_by_SingleVectorString(SingleVectorString)
