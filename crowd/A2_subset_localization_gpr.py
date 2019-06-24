import numpy as np
import re
import random
from scipy import spatial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import GP
import math

SAMPLING_PERCENTAGE = 0.8
NUM_OF_SAMPLINGS = 10
SIMILARITY_TOPK = 10
POWER_LOWER_BOUND = -1000.0

SIMILARITY_THRESHOLD = 0.85
MAX_NUM_CLUSTERS = 4
DBM_DIFF = 30

# RP_FILE = "Data/1002.txt"
# CR_FILE = "Data/crowd_targets_raw_calibrated.txt"
# IMG_FILE = "Data/1002.jpg"
RP_FILE = '/Users/kris/Downloads/floor_test/hc/newref9002/p_all.txt'
CR_FILE = '/Users/kris/Downloads/floor_test/hc/tar9002/p_all copy.txt'
IMG_FILE = "/Users/kris/Downloads/floor_test/map/9002.png"
####################################################################

def rssi2power(rssi):
    return math.pow(2.0, rssi/10.0)

def power2rssi(power):
    return 10.0*math.log(power, 2.0)

class Observation():
    def __init__(self, mac = None, rssi = None):
        if mac and rssi:
            self.mac = mac
            self.rssi = rssi
            self.power = rssi2power(rssi)
            self.power2 = self.power*self.power
        else:
            self.mac = None
            self.rssi = POWER_LOWER_BOUND
            self.power = 0.0
            self.power2 = 0.0

class ObservationVector():
    def __init__(self, observations = None):
        self._v = []
        self._mac2rssi = {}
        self._mac2power = {}
        self._mac2power2 = {}

        if observations:
            for o in observations:
                self._mac2rssi[o.mac] = o.rssi
                self._mac2power[o.mac] = o.power
                self._mac2power2[o.mac] = o.power2
                self._v.append(o)

    def load(self, element_list):
        for element in element_list:
            # element example: "002584231db0:-85.000000,0,0.200000"
            mac = element.split(':')[0]
            rssi = float(element.split(':')[1:][0].split(','[0])[0])
            o = Observation(mac, rssi)
            self._mac2rssi[mac] = rssi
            self._mac2power[o.mac] = o.power
            self._mac2power2[o.mac] = o.power2
            self._v.append(o)

    def has_mac(self, mac):
        try:
            self._mac2rssi[mac]
            return True
        except:
            return False

    def __len__(self):
        return len(self._v)

    def __getitem__(self, i):
        return self._v[i]

    def __setitem__(self, i, o):
        self._v[i] = o

    def get_macs(self):
        return [o.mac for o in self._v]

    def get_rssi_by_mac(self, mac):
        return self._mac2rssi[mac]

    def get_power_by_mac(self, mac):
        return self._mac2power[mac]

    def get_power2_by_mac(self, mac):
        return self._mac2power2[mac]

    def cosine_sim(self, observation):

        macs1 = self.get_macs()
        macs2 = observation.get_macs()

        dot = 0.0
        sqsum1 = 0.0
        sqsum2 = 0.0

        for mac in macs1:
            if mac in macs2:
                dot += self.get_power_by_mac(mac)*observation.get_power_by_mac(mac)

        if dot == 0.0:
            return 0.0

        for mac in macs1:
            sqsum1 += self.get_power2_by_mac(mac)

        for mac in macs2:
            if mac in macs1:
                sqsum2 += observation.get_power2_by_mac(mac)

        return dot/( math.sqrt(sqsum1*sqsum2) )

    def get_random_sample(self, sampling_rate):
        sampling_size = int(len(self._v) * sampling_rate)
        return ObservationVector(random.sample(self._v, sampling_size))

class RPVector():
    def __init__(self):
        self.x = -1
        self.y = -1
        self.dir = -1
        self.rp_index = -1 # the line number (minus 1) of this vector in input file
        self.observation_vector = ObservationVector()
        self.accurate = True

    def __len__(self):
        return len(self.observation_vector)

    def len(self):
        return len(self.observation_vector)

    def get_macs(self):
        return self.observation_vector.get_macs()

    def has_mac(self, mac):
        return self.observation_vector.has_mac(mac)

    def get_rssi_by_mac(self, mac):
        return self.observation_vector.get_rssi_by_mac(mac)

class CRVector():
    def __init__(self):
        self.est_x = -1.0 # estimated x
        self.est_y = -1.0 # estimated y
        self.observation_vector = ObservationVector()
        self.cr_index = -1 # the line number (minus 1) of this vector in input file

    def __len__(self):
        return len(self.observation_vector)

    def len(self):
        return len(self.observation_vector)

    def __getitem__(self, i):
        return self.observation_vector[i]

    def get_rssi_by_mac(self, mac):
        return self.observation_vector.get_rssi_by_mac(mac)

    def get_macs(self):
        return self.observation_vector.get_macs()

    def get_random_sample(self, sampling_rate):
        return self.observation_vector.get_random_sample(sampling_rate)

    def has_mac(self, mac):
        return self.observation_vector.has_mac(mac)

def load_rp_vector(str):
    items = str.split(' ')
    rpv = RPVector()
    t = items[0].split(',')
    rpv.x, rpv.y, rpv.dir, rpv.index = float(t[0]), float(t[1]), int(t[2]), t[3]
    rpv.observation_vector = ObservationVector()
    rpv.observation_vector.load(items[1:])
    return rpv

def load_rp_vectors(file):
    v = []
    with open(file) as f:
        lines = f.readlines()
    count = 0
    for line in lines:
        rpv = load_rp_vector(line.rstrip())
        rpv.rp_index = count
        v.append(rpv)
        count += 1
    return v

def load_cr_vector(str):
    crv = CRVector()
    crv.observation_vector = ObservationVector()
    crv.observation_vector.load(str.split(' '))
    return crv

def load_cr_vectors(file):
    v = []
    with open(file) as f:
        lines = f.readlines()
    count = 0
    for line in lines:
        crv = load_cr_vector(line.rstrip())
        crv.cr_index = count
        v.append(crv)
        count += 1
    return v

def dump_rp_vectors(rp_vetors):
    for rp_vector in rp_vectors:
        print "rp_vector(x {0}, y {1}, dir {2}, index {3}, ap_vector_length {4}): ".format(rp_vector.x, rp_vector.y, rp_vector.dir, rp_vector.index, len(rp_vector))

def dump_cr_vectors(cr_vectors):
    for cr_vector in cr_vectors:
        print "cr_vector(x {0}, y {1}, ap_vector_length {2}): ".format(cr_vector.x, cr_vector.y, len(cr_vector))

def max_cosine_sim(v1, rp_vectors):
    maxcs = 0.0
    max_rp_vector = None
    for v2 in rp_vectors:
        cs = v1.cosine_sim(v2.observation_vector)
        if cs>maxcs:
            maxcs = cs
            max_rp_vector = v2
    return ((max_rp_vector.x, max_rp_vector.y), maxcs)

# for each crowd vector, find estimated x, y
def estimate_location(cr_vector, rp_vectors):
    # compute a list of (RP, cs)'s with highest similarity to
    # list length less or equal to SIMILARITY_TOPK
    topk = min(SIMILARITY_TOPK, len(rp_vectors))
    similarity_list = []
    subset = []

    for i in range(0, NUM_OF_SAMPLINGS):
        subset.append(cr_vector.get_random_sample(SAMPLING_PERCENTAGE))

    for v in subset:
        similarity_list.append(max_cosine_sim(v, rp_vectors))

    # find the set of RPs with highest similarity
    topk = sorted(similarity_list, key=lambda tup: -tup[1])
    # print "vec", subset
    print "top1 = ", topk[0]
    # exit(0)

    cs_set = [t[1] for t in topk]
    if np.max(cs_set) < SIMILARITY_THRESHOLD:
        return -1.0, -1.0

    # clustering
    rp_index_set = sorted([t[0] for t in topk])
    cluster_dict = {}
    for e in rp_index_set:
        try:
            cluster_dict[e] = cluster_dict[e] + 1
        except:
            cluster_dict[e] = 1
    # sort the clusters in descending order according to cluster size
    # cluster_set[0] is the biggest cluster
    cluster_set = sorted([(k, cluster_dict[k]) for k in cluster_dict.keys()], key = lambda tup: -tup[1])
    # print "Clusters: ",cluster_set

    # We use two criteria to determine if a crowd vector is "trust worthy"
    # 1. number of total clusters < MAX_NUM_CLUSTERS
    # 2. the size of the largest cluster is greater than half of sum of all the clusers' size'
    estx, esty = -1.0, -1.0
    if (len(cluster_set)<MAX_NUM_CLUSTERS):
        num_of_points = np.sum([t[1] for t in cluster_set])
        if cluster_set[0][1] > num_of_points/2:
            estx, esty = cluster_set[0][0][0], cluster_set[0][0][1]

    return estx, esty

if __name__ == "__main__":

    random.seed(100)

    rp_vectors = [] # a list of RP (reference point) vectors
    cr_vectors = [] # a list of crowd source vectors
    union_mac_list = []
    pair_count_dict = {}

    rp_vectors = load_rp_vectors(RP_FILE)
    #dump_rp_vectors(rp_vectors)
    print "Reference Points:"
    print "Total references taken: ",len(rp_vectors)
    lens = [len(v) for v in rp_vectors]
    print "Observations per reference (max, mean, min, std): ",np.max(lens),np.mean(lens),np.min(lens),np.std(lens)
    rf_points = list(set([ (v.x, v.y) for v in rp_vectors ]))
    print "Num of reference points (x,y): ",len(rf_points)

    for v in rp_vectors:
        t = (v.x, v.y)
        try:
            pair_count_dict[t] += 1
        except:
            pair_count_dict[t] = 1
    pair_count_tuples = [ (k, pair_count_dict[k]) for k in pair_count_dict.keys()]
    sorted_pair_count_tuples = sorted(pair_count_tuples, key = lambda tuple: -tuple[1])
    #print sorted_pair_count_tuples
    counts = [pair_count_dict[k] for k in pair_count_dict.keys()]
    print "References taken at each reference point (max, mean, min, std): ",np.max(counts),np.mean(counts),np.min(counts),np.std(counts)

    print "Crowd Vectors:"
    cr_vectors = load_cr_vectors(CR_FILE)
    #dump_cr_vectors(cr_vectors)
    print "Num of crowd vectors: ",len(cr_vectors)
    lens = [len(v) for v in cr_vectors]
    print "Observations per crowd vector (max, mean, min, std): ", np.max(lens), np.mean(lens), np.min(lens), np.std(lens)

    mac_list = []
    for v in rp_vectors + cr_vectors:
        mac_list = mac_list + v.get_macs()

    union_mac_list = list(set(mac_list))

    num_of_aps = len(union_mac_list)
    print
    print "num of total APs = ",num_of_aps

    ####################################################
    # Estimate location for each crowd vector
    estimate_locations = []
    good_quality_cr_vectors = []
    for v in cr_vectors:
        print "Estimating location for crowd vector {0} ...".format(v.cr_index)
        est_x, est_y = estimate_location(v, rp_vectors)
        v.est_x, v.est_y = est_x, est_y
        if est_x>0.0 and est_y>0.0:
            good_quality_cr_vectors.append(v)
        print "est_x, est_y = ",est_x, est_y

    print "Num of good quality crowd vectors: ", len(good_quality_cr_vectors)

    ####################################################
    # Use GPR to reconstruct signal map and flag those RPs needed to be resureyed
    # 1. Extract all the APs from good_quality_cr_vectors
    ap_macs = []

    for v in good_quality_cr_vectors:
        for i in range(0, len(v)):
            ap_macs.append(v[i].mac)

    ap_macs = list(set(ap_macs)) # remove redundant aps
    print "Num of APs in high quality crowd vectors: ", len(ap_macs)

    xy_flagged = []
    for mac in ap_macs:
        # 2. For each AP, get the (x,y) and rssi from high quality cr vectors and train GP
        # using the (x,y) and rssi
        # print "Training GPR ..."
        xy_train = []
        rssi_train = []
        count = 0
        for v in good_quality_cr_vectors:
            if v.has_mac(mac):
                xy_train.append((float(v.est_x), float(v.est_y)))
                rssi_train.append(float(v.get_rssi_by_mac(mac)))
        assert(xy_train)
        assert(rssi_train)
        try:
            GP.GP(xy_train, rssi_train)
        except:
            print "Training Exception"
            continue
        # 3. For every AP, go through those RPs that has the AP
        # and flag it if there is a huge difference in rssi
        # print "Identifying locations that need new survey ..."
        for v in rp_vectors:
            if v.accurate and v.has_mac(mac): # skip if it has already been flagged as inaccurate
                rssi_estimate = GP.EstimateGP(float(v.x), float(v.y))
                if math.fabs(v.get_rssi_by_mac(mac) - rssi_estimate[0][0]) > DBM_DIFF:
                    v.accurate = False
                    xy_flagged.append((int(v.x), int(v.y)))

    num_rp_flagged = len(set(xy_flagged))
    print "Num reference points flagged = ", num_rp_flagged
    print "Percentage of reference points flagged = ", float(num_rp_flagged)/float(len(rf_points))

    ####################################################
    img = plt.imread(IMG_FILE)

    # plot original reference points
    x = []
    y = []
    for v in rp_vectors:
        x.append(v.x)
        y.append(v.y)

    plt.scatter(x, y, s=80, marker=".")
    plt.imshow(img)
    plt.savefig("rp.jpg")

    # plot reference points required resurveyed
    x = []
    y = []
    for v in rp_vectors:
        if not v.accurate:
            x.append(v.x)
            y.append(v.y)

    plt.scatter(x, y, s=30, color="r", marker=".")
    plt.imshow(img)
    plt.savefig("rp2.jpg")
