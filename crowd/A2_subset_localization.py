import re
import math
import os.path, time
import datetime as dt
from datetime import datetime
from sklearn.cluster import AffinityPropagation
import numpy as np
from scipy import spatial
from itertools import cycle
import random
from copy import deepcopy
from  shutil import move
import shutil
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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
    else:
        return rssi-A0.global_settings['rssi_negative_2_positive_shift']

# input: (1) a list of real values 'a'; (2) the number of top K (descending order)
# output: a list of topK index, note: it's not the topK values,but index
# this function is to select topK items of a list
def topK_index_array(a, topK): return np.argsort(a)[::-1][:topK]

# this function is to generate 0 or 1 based on a predefined probability
def randomNum_generator(sample_rate):
    assert(sample_rate>0.0 and sample_rate<1.0)
    r = random.random()
    assert(r>0.0 and r<1.0)
    if r<sample_rate:
        return 1.0
    else:
        return 0.0

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

# input: a subset vector of RSSI
# output: vector length
# to check the length of subset vector
def subset_vector_length_checker(subset_vector):
    length=0
    for i in range(subset_vector.shape[0]):
        if subset_vector[i,1]!=0:
            length+=1
    return length
################################################################################################ utilities

################################################################################################ core functions
# input: a black list of mac address( which will be ignored in processing)
# output: (1)a 3D arrary which is fingerprint database,(2) a dictionary of reference points,(3) a dic of AP mac addresss
# this function is to load reference points from txt file

"""
rp_num_count, ap_num_count, rp_index, ap_index all start from 0

ref_AP_dic
{0: '7072cf05ec97', 1: '002584231d90', 2: '0023eb3a1090', 3: '7072cf05ec98', 4: '0023eb3a0fb0', 5: '34bdc8dbc414', '002584231d90': 1, 7: '34bdc8dbc413', 8: '34bdc8dbc411', 9: '00258422fb30', 10: '0023eb3a1220', 11: '002584231c40', 12: '002584231c41', 13: '00258422f630', 14: 'b4750e4fc895', 15: '0023eb3a1050', 16: '4ce67645bd1d', 17: '20aa4bcbb9c8', 18: '0025842317b0', 19: '0019be000550', 20: 'f4ec38b451a4', 21: '002584231e80', 22: '002584231db2', 23: '588d09e2e1ba', 24: '002584231db0', 25: '002584231bb1', '00258422fb30': 9, 27: '002584231630', 28: '3037a6a54202', 29: '3037a6a54203', 30: '001d73053465', 31: '3037a6a54201', 32: '002584231c71', 33: '3037a6a54204', 34: '002584231c70', 35: '3037a6a54205', 36: '00258422fed1', 6: '34bdc8dbc415', 38: '002584231bb0', 39: '00258422fed0', 40: '00258422fa80', 41: '002584231db3', 42: '00258422fb31', 43: '002584231db1', '0023eb3a1050': 15, 45: '3037a6a54200', 46: '002584231631', 47: '0023eb0b5180', 48: '0017dfaa9ba1', '0019be000550': 19, '002584231db3': 41, '002584231db2': 22, '002584231db1': 43, '002584231db0': 24, '002584231bb1': 25, '002584231bb0': 38, '002584231c71': 32, '3037a6a54205': 35, '0023eb0b5180': 47, '0023eb3a0fb0': 4, '0023eb3a1220': 10, 26: '002584231dd0', '00258422fb31': 42, '002584231e80': 21, '002584231630': 27, '002584231631': 46, 37: '00116b47d836', '001d73053465': 30, '0023eb3a1090': 2, 44: '00258422ff90', '00258422f630': 13, '4ce67645bd1d': 16, '20aa4bcbb9c8': 17, '3037a6a54202': 28, '3037a6a54203': 29, '3037a6a54200': 45, '3037a6a54201': 31, '3037a6a54204': 33, '002584231c70': 34, '00258422fed0': 39, '00258422fed1': 36, '0017dfaa9ba1': 48, '7072cf05ec97': 0, '7072cf05ec98': 3, '00258422ff90': 44, '34bdc8dbc414': 5, '34bdc8dbc415': 6, '34bdc8dbc413': 7, '34bdc8dbc411': 8, '002584231c40': 11, '002584231c41': 12, 'b4750e4fc895': 14, '00258422fa80': 40, '0025842317b0': 18, 'f4ec38b451a4': 20, '588d09e2e1ba': 23, '002584231dd0': 26, '00116b47d836': 37}

ref_RP_dic
{0: ('3641', '1113', '0'), 1: ('3641', '1113', '1'), 2: ('3681', '1113', '0'), 3: ('3681', '1113', '1'), ('3641', '1113', '0'): 0, ('3681', '1113', '1'): 3, ('3641', '1113', '1'): 1, ('3681', '1113', '0'): 2, 4: ('3721', '1113', '0'), ('3721', '1113', '0'): 4}

"""

def load_reference_points (ap_black_list=[]):
    ref_file = get_newest_txt_file(A0.global_settings['fingerprint_txt_file_prefix'])
    if not ref_file:
        print('Reference file is missing...\n')
        return -1
    print('loading reference points:'+ref_file+'\n')
    AP_dic={}; RP_dic={}
    f = open(ref_file, 'r')
    total_RP_num = sum(1 for line in f)
    f.seek(0)
    estimated_max_AP_num_in_each_RP = A0.global_settings['estimated_max_AP_num_in_each_RP']
    data_num_in_each_AP = 3  # AP_index + RSSI+sd
    ref_byRP_3D_array = np.zeros((total_RP_num,  estimated_max_AP_num_in_each_RP,  data_num_in_each_AP))

    rp_num_count = -1
    for line in f:
        rp_num_count +=1
        #print('rp_num_count:'+str(rp_num_count)+'\n')
        temp=re.split(',|:|\s',line)
        # put (x, y, d) into dic
        k=0
        while ((temp[0], temp[1],str(k)) in RP_dic):
            k+=1
        temp[2]=str(k)

        RP_dic [(temp[0], temp[1], temp[2])] =int( len(RP_dic)/2)
        RP_dic [ RP_dic[(temp[0], temp[1], temp[2])] ] = (temp[0], temp[1], temp[2])

        # to read mac+rssi in each line
        ap_num_count = -1
        for t in range(4,len(temp) +1,4):
            rp_index = RP_dic[(temp[0], temp[1], temp[2])]
            ref_byRP_3D_array[rp_num_count, 0, 0] = rp_index
            if len(temp)<=4:
                continue
            if t>=len(temp):
                break
            if temp[t]=='':
                break
            if ap_num_count >= estimated_max_AP_num_in_each_RP:
                raise Exception('fingerprints_byRP_txt_2_3D_array()', 'fingerprint_byRP_3D_array is too small\n')
            if temp[t] in ap_black_list:
                continue
            if temp[t] not in AP_dic:  # mac -> index
                AP_dic[temp[t]] = int( len(AP_dic)/2)
                AP_dic[ AP_dic[temp[t]] ] = temp[t]
            ap_num_count+=1
            ap_index = AP_dic[temp[t]]
            #print "rp_num_count, ap_num_count, ap_index = ",(rp_num_count, ap_num_count, ap_index)
            ref_byRP_3D_array[rp_num_count, ap_num_count, 0] = ap_index
            ref_byRP_3D_array[rp_num_count, ap_num_count, 1] = rssi_negative_2_positive(float(temp[t+1]))
            if temp[t+2]=='':
                temp[t+2]='0.0'
            ref_byRP_3D_array[rp_num_count, ap_num_count, 2] = round(float(temp[t+2]),1)

    if rp_num_count+1 < total_RP_num: # have duplicated RP
        ref_byRP_3D_array=ref_byRP_3D_array[:rp_num_count+1]
    print('loading reference points completed!\n')
    return ref_byRP_3D_array, AP_dic, RP_dic

# input: (1) a target vector of RSSI,(2) fingerprint database and the associated RP/AP dictionary
# output: esitimated location: (x, y) and estimated direction: d (in fact, such direction is simply for data format)
# this funciton uses: random sampling + naive localization(cosine_simlarity) + clustering, to estimate location
def subset_loc(target_vector, ref_byRP_3D_array,ref_AP_dic,ref_RP_dic):
    # 1. generate subsets
    subset_num = A0.global_settings['subset_localization_subset_num']
    subset_loc=np.zeros((subset_num,3))

    for i in range(0, subset_num):
        #print ('subset localization cycle:'+str(i+1)+'\n')
        subset_vector=deepcopy(target_vector)
        for ap_num in range(target_vector.shape[0]):
            #subset_vector[ap_num,1]=target_vector[ap_num,1]*random.randint(0,1)
            subset_vector[ap_num,1]=target_vector[ap_num,1]*randomNum_generator(A0.global_settings['subset_sample_rate'])

        x,y,d = cossim_loc(subset_vector,ref_byRP_3D_array,ref_AP_dic,ref_RP_dic)

        assert not math.isnan(x)
        assert not math.isnan(y)

        subset_loc[i,0]=x
        subset_loc[i,1]=y
        subset_loc[i,2]=d

    subset_loc=subset_loc[~(subset_loc==0).all(1)] # remove all-zero rows
    if subset_loc.shape[0]==0:
        return (-1,-1,-1)
    if subset_loc.shape[0]==1:
        return (subset_loc[0,0],subset_loc[0,1],subset_loc[0,2])

    # 2. location Affinity Propagation clustering
    x,y,d = AP_clustering(subset_loc)
    if A0.global_settings['Subset_clustering_visualization']:
        AP_cluster_show_plot(subset_loc)
    return (x,y,d)

# input: (1) a target vector of RSSI, (2) fingerprint database and the associated RP/AP dictionary
# output: esitimated location: (x, y) and estimated direction: d (in fact, such direction is simply for data format)
# this function use: naive localization (cosine_similarity), to estimate location
def cossim_loc(target_vector, ref_byRP_3D_array,ref_AP_dic,ref_RP_dic):
    cossim_raw=[]
    target_vector_tmp=np.zeros((int(len(ref_AP_dic)/2),1))
    for i in range(target_vector.shape[0]):
        target_vector_tmp[target_vector[i,0]]=target_vector[i,1]
    for rp_num in range(ref_byRP_3D_array.shape[0]):
        ref_vector_tmp=np.zeros((int(len(ref_AP_dic)/2),1))
        for ap_num in range(1,ref_byRP_3D_array.shape[1],1):
            if ref_byRP_3D_array[rp_num,ap_num,0]==0:
                break
            ref_vector_tmp[ref_byRP_3D_array[rp_num,ap_num,0]]=ref_byRP_3D_array[rp_num,ap_num,1]
        cossim_raw.append(1 - spatial.distance.cosine(target_vector_tmp[1:], ref_vector_tmp[1:]))
    topK=A0.global_settings['subset_localization_topK']
    topK_index = topK_index_array(cossim_raw,topK)
    summ=0
    for k in range(topK):
        summ+=cossim_raw[topK_index[k]]
    [x, y, d]=[0, 0, 0]
    d = ref_RP_dic[ ref_byRP_3D_array[topK_index[0],0,0] ][2]
    for k in range(topK):
        x+= ( cossim_raw[topK_index[k]]/summ ) * float(ref_RP_dic[ref_byRP_3D_array[topK_index[k],0,0]][0])  # x
        y+= ( cossim_raw[topK_index[k]]/summ ) * float(ref_RP_dic[ref_byRP_3D_array[topK_index[k],0,0]][1])  # y

    return (round(x,1),round(y,1),int(d))

# input: a list of locations (x,y,d) which are estimated from: random sampling + naive localization
# output: esitimated location: (x, y) and estimated direction: d (in fact, such direction is simply for data format)
# this function is to find the final location from dense cluster
def AP_clustering(X):
    # Compute Affinity Propagation
    af = AffinityPropagation().fit(X[:,0:2])
    #AP_cluster_show_plot(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    # to find the dentest cluster and return its centroid
    max=0
    max_index=0
    for k in range(n_clusters_):
        temp=0
        for kk in range(len(labels)):
            if labels[kk]==k:
                temp+=1
        if temp>max:
            max=temp
            max_index=k
    cluster_center = cluster_centers_indices[max_index]
    return X[cluster_center,0],X[cluster_center,1],X[cluster_center,2]

# input: a list of locations (x,y,d) which are estimated from: random sampling + naive localization
# output: a plot to show the clusters of the location list
# this function is to visualize clusters,
# it can be disabled by setting: global_settings['Subset_clustering_visualization']=False
def AP_cluster_show_plot(X):
    af = AffinityPropagation().fit(X[:,0:2])
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    plt.close('all')
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()
    now=dt.datetime.now()
    plt.savefig('Visualization_SubsetClustering/'+str(now)[0:10]+'-'+str(now.time())[0:8].replace(':','-')+'.png')
    return 0

# input: a list of estimated locations, a list of true locations(if there are)
# output: a plot to show the accaracy of localization
# this function is to visualize localization results,
# it can be disabled by setting: global_settings['Subset_localization_visualization']=False
def subset_loc_show_plot(crowd_tar_file,estimated_xy,true_xy):
    if len(estimated_xy)!=len(true_xy):
        print('localization show plot error..\n');return -1
    # to save as txt
    crowd_file_trueLoc=open('Visualization_SubsetLocalization/'+crowd_tar_file[0:len(crowd_tar_file)-4]+'_loc_true.txt','w')
    crowd_file_estLoc=open('Visualization_SubsetLocalization/'+crowd_tar_file[0:len(crowd_tar_file)-4]+'_loc_esti.txt','w')
    for i in range(len(estimated_xy)):
        crowd_file_trueLoc.write(str(true_xy[i][0])+','+str(true_xy[i][1])+'\n')
        crowd_file_estLoc.write(str(estimated_xy[i][0])+','+str(estimated_xy[i][1])+'\n')
    crowd_file_trueLoc.close()
    crowd_file_estLoc.close()

    # to plot
    map_file=A0.global_settings['background_floor_map']
    if not os.path.exists(map_file):
        print('cannot find background map..\n'); return -1
    im = Image.open(map_file)
    draw = ImageDraw.Draw(im)
    # draw.line(zip(trueLocs,estLocs), "green", width=1)
    for i in range(len(true_xy)):
        if true_xy[i][0]!=-1 and true_xy[i][1]!=-1 and estimated_xy[i][0]!=-1 and estimated_xy[i][1]!=-1:
            draw.line([(true_xy[i][0],true_xy[i][1]),(estimated_xy[i][0],estimated_xy[i][1])], "red", width=2)
    now=dt.datetime.now()
    im.save('Visualization_SubsetLocalization/'+crowd_tar_file[0:len(crowd_tar_file)-4]+'_'+str(now)[0:10]+'-'+str(now.time())[0:8].replace(':','-')+'.png',"PNG")
    return 0
################################################################################################ core functions


################################################################################################ API functions
# input: (1)crowd_tar_file: the txt file which contains multiple crowdsource RSSI vectors,
#        (2)ref_byRP_3D_array: the fingerprint database
#        (3)ref_AP_dic,ref_RP_dic: the associated RP/AP dictionary of fingerprint database
# output: a txt file, each line(i.e.each RSSI vector) has a location (x,y,d)
# this API function is to estimate locations of batch crowdsource RSSI vectors using advanced algorithm
# several visualization modules can be enable/disable in this function
def localization_byBatchVectorTxtFile(crowd_tar_file,ref_byRP_3D_array,ref_AP_dic,ref_RP_dic):
    if A0.global_settings['Subset_clustering_visualization']:
        if not os.path.exists('Visualization_SubsetClustering'):
            os.makedirs('Visualization_SubsetClustering')

    estimated_xy=[];true_xy=[]
    crowd_file_localized=open(A0.global_settings['fingerprint_txt_file_folder']+crowd_tar_file[0:len(crowd_tar_file)-4]+'_localized.txt','w')
    crowd_file = open(A0.global_settings['fingerprint_txt_file_folder']+crowd_tar_file, 'r')
    k=0
    for line in crowd_file:
        if not line or line=='\n':
            continue
        k+=1; print('subset localization of vector:'+str(k)+'\n')
        new_line,(e_x,e_y,_),(t_x,t_y,_)=localization_bySingleVectorString(line,ref_byRP_3D_array,ref_AP_dic,ref_RP_dic)
        crowd_file_localized.write(new_line+'\n')
        estimated_xy.append([e_x,e_y])
        true_xy.append([t_x,t_y])
    crowd_file.close()
    crowd_file_localized.close()

    if A0.global_settings['Subset_localization_visualization']:
        map_file=A0.global_settings['background_floor_map']
        if not os.path.exists(map_file):
            print('cannot find background map..\n')
        if not os.path.exists('Visualization_SubsetLocalization'):
            os.makedirs('Visualization_SubsetLocalization')
        subset_loc_show_plot(crowd_tar_file,estimated_xy,true_xy)
    return 0

# input: (1)SingleVectorString: a string which contatins a vector of RSSI values
#        (2)ref_byRP_3D_array: the fingerprint database
#        (3)ref_AP_dic,ref_RP_dic: the associated RP/AP dictionary of fingerprint database
# output:(1)new_vector_string: a string which contains estimated location (x,y,d) and RSSI vector
#        (2)(estimated_x,estimated_y,estimated_d): estimated location using advanced algorithm
#        (3)(true_x,true_y,true_d): true location if there is, otherwise (true_x,true_y,true_d)==(-1,-1,-1)
# this API function is to estimate location of a single RSSI vector using advanced algorithm
def localization_bySingleVectorString(SingleVectorString,ref_byRP_3D_array,ref_AP_dic,ref_RP_dic):
    new_vector_string=''
    true_x=-1; true_y=-1;true_d=-1
    estimated_x=-1;estimated_y=-1;estimated_d=-1
    temp=re.split(',|:|\s',SingleVectorString)
    if temp=='':
        print('vector string is empty...\n')
        return -1
    crowd_vector=np.zeros((A0.global_settings['estimated_max_AP_num_in_each_RP'], 2))  # mac_index + rssi(0)
    ap_no=-1
    for t in range(0,len(temp),4):
        if temp[t]=='':
            break
        if t==0 and len(temp[t]) < A0.global_settings['ap_mac_address_string_length']: # not mac address
            continue
        new_vector_string+=temp[t]+':'+temp[t+1]+','+temp[t+2]+','+temp[t+3]+' '
        if temp[t] not in ref_AP_dic: # ignore other APs
            continue
        ap_no+=1
        if ap_no>A0.global_settings['estimated_max_AP_num_in_each_RP']:
            raise Exception('localization_bySingleVectorString()', 'global_settings[\'estimated_max_AP_num_in_each_RP\'] is too small\n')
        crowd_vector[ap_no,0] = ref_AP_dic[temp[t]]
        crowd_vector[ap_no,1] = rssi_negative_2_positive(float(temp[t+1])) # mac_index

    if str(temp[0]).isdigit()and len(temp[0])!=A0.global_settings['ap_mac_address_string_length']: # has location
        true_x=int(temp[0]);true_y=int(temp[1]);true_d=int(temp[2])

    if (not str(temp[0]).isdigit()) or len(temp[0])==A0.global_settings['ap_mac_address_string_length'] or A0.global_settings['To_localize_even_with_locations']: # does not have location
        if ap_no==-1: # cannot localization because there is no AP
            new_vector_string=str(estimated_x)+','+str(estimated_y)+','+str(int(estimated_d))+',0 '+new_vector_string
            return new_vector_string,(estimated_x,estimated_y,estimated_d),(true_x,true_y,true_d)
        if ap_no+1 < A0.global_settings['estimated_max_AP_num_in_each_RP']:
            crowd_vector=crowd_vector[:ap_no+1]

        #print "crowd_vector = ", crowd_vector
        estimated_x, estimated_y, estimated_d = subset_loc(crowd_vector,ref_byRP_3D_array,ref_AP_dic,ref_RP_dic)
        new_vector_string=str(estimated_x)+','+str(estimated_y)+','+str(int(estimated_d))+','+'0 '+new_vector_string
    else: # has location and no need to localize
        true_x=int(temp[0]);true_y=int(temp[1]);true_d=int(temp[2])
        new_vector_string=temp[0]+','+temp[1]+','+temp[2]+','+temp[3]+' '+new_vector_string

    return new_vector_string,(estimated_x,estimated_y,estimated_d),(true_x,true_y,true_d)
################################################################################################ API functions



################################################################################################ Demo
if __name__ == '__main__':
    # the following lines are for function flow visulization
    #config = Config(max_depth=5)
    #config.trace_filter = GlobbingFilter(exclude=['pycallgraph.*'])
    #graphviz = GraphvizOutput(output_file='A2_module_function_flow.png')
    #with PyCallGraph(output=graphviz, config=config):
    # the above lines are for function flow visulization

        # load reference points
        random.seed(100)
        ref_byRP_3D_array, ref_AP_dic, ref_RP_dic=load_reference_points()
        #print "ref_byRP_3D_array"
        #print ref_byRP_3D_array
        #print "ref_AP_dic"
        #print ref_AP_dic
        #print "ref_RP_dic"
        #print ref_RP_dic

        # Demo to show how to call API function
        # output file: 'crowd_targets_raw_localized.txt',which is the input of module: 'A3_fingerprint_updating.py'
        crowd_tar_file='crowd_targets_raw_calibrated.txt'
        localization_byBatchVectorTxtFile(crowd_tar_file,ref_byRP_3D_array,ref_AP_dic,ref_RP_dic)

        # Demo to show how to call API function
        #crowd_tar_vector='2c5d93087159:-74,0,1 2c5d93c87158:-72,0,1 0228c8d8dcd2:-77,0,1 64a0e790f0d3:-69,0,1 64a0e7895892:-57,0,1 8843e113c872:-46,0,1'
        #crowd_tar_vector_localized,est_xyd,true_xyd=localization_bySingleVectorString(crowd_tar_vector,ref_byRP_3D_array,ref_AP_dic,ref_RP_dic)
