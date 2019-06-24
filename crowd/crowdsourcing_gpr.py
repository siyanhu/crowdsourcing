from __future__ import print_function
import re
from itertools import count, izip
from collections import Counter

import numpy as np
from numpy import linalg as LA
from sklearn.cluster import AffinityPropagation
from sklearn.gaussian_process import GaussianProcessRegressor


REF = '/Users/kris/Downloads/floor_test/hc/ref9002/p_all.txt'
TAR = '/Users/kris/Downloads/floor_test/hc/tar9002/p_all.txt'
imFile = '/Users/kris/Downloads/floor_test/map/9002.png'

KNN = 20
SAMPLE_RATE = 0.8
SAMPLE_NO = 20
SHORT_VEC_AP_NUM_THRES = 10
BIG_CHANGE_DB_DIFF = 30
GP_AP_NO_THRES = 10


def load_ref_data(ref):
    content = open(ref).read()
    macs = set(int(s, 16) for s in re.findall(r'(\w{9,12}):', content))
    assert len(macs) != 0
    macIds = dict(izip(macs, count()))
    idMacs = dict(izip(count(), macs))
    lines = [l.strip() for l in content.splitlines() if ',' in l]
    matrix = np.zeros((len(lines), len(macIds)), dtype=np.float32)
    print('REF\t%s: %d points found, %d macs found' % (ref, len(lines), len(macIds)))
    ptMat = np.zeros((len(lines), 2), dtype=np.float32)
    for i, line in enumerate(lines):
        items = line.split(' ')
        loc = [float(j) for j in items[0].split(',')]
        try:
            for item in items[1:]:
                tmp = re.split(r':|,', item)
                tmp[0] = int(tmp[0], 16)
                matrix[i, macIds[tmp[0]]] = float(tmp[1])
        except Exception as e:
            print(e)
            print(line)
            exit(-1)
        ptMat[i, 0] = loc[0]
        ptMat[i, 1] = loc[1]
    matrix = np.power(2, matrix / 10.0)
    matrix[matrix >= 1] = 0
    return macIds, idMacs, ptMat, matrix


def load_target_data(macIds, tar):
    lines = [l.strip() for l in open(tar).xreadlines() if ',' in l]
    matrix = np.zeros((len(lines), len(macIds)), dtype=np.float32)
    print('TAR\t%s: %d points found' % (tar, len(lines)))

    ptMat = np.zeros((len(lines), 2), dtype=np.float32)
    for i, line in enumerate(lines):
        items = line.split(' ')
        loc = [float(j) for j in items[0].split(',')]
        for item in items[1:]:
            tmp = re.split(r':|,', item)
            tmp[0] = int(tmp[0], 16)
            if tmp[0] not in macIds:
                continue
            matrix[i, macIds[tmp[0]]] = float(tmp[1])
        ptMat[i, 0] = loc[0]
        ptMat[i, 1] = loc[1]
    matrix = np.power(2, matrix / 10.0)
    matrix[matrix >= 1] = 0
    # print ptMat
    # np.savetxt('c2.txt', ptMat, fmt='%f',delimiter=',')
    # exit(-1)
    return ptMat, matrix


def get_cosine_ests(refMatrix, tarMatrix, ptsRef):
    assert refMatrix.shape[1] == tarMatrix.shape[1]
    ptsEst = np.zeros((tarMatrix.shape[0], 2), dtype=np.float32)
    for i in xrange(tarMatrix.shape[0]):
        tarVec = tarMatrix[i]
        selRefMat = refMatrix[:, tarVec != 0]
        tarVec = tarVec[tarVec != 0]
        sims = np.dot(selRefMat, tarVec) / LA.norm(selRefMat, axis=1) / LA.norm(tarVec)
        sims[np.isnan(sims)] = 0
        idx = np.argsort(sims)
        finalGroup = idx[-1 * KNN:]
        selectedSims = sims[finalGroup]
        selectedSims[selectedSims >= 1] = 0.9999
        selectedPts = ptsRef[finalGroup]
        wm = 1.0 / (1 - selectedSims) ** 2
        normalizer = np.sum(wm)
        if normalizer < 10e-3:
            ptsEst[i] = np.array([0, 0])
        else:
            wm /= normalizer
            ptsEst[i] = np.sum((wm, wm) * selectedPts.T, axis=1)
    return ptsEst


def localise(refMat, tarMat, ptsRef, sampleRate, sampleNo, apNoThres):
    ptsEst = np.zeros((tarMat.shape[0], 2), dtype=np.float32)
    for i in xrange(tarMat.shape[0]):
        if (tarMat[i]!=0).sum() < apNoThres:
            tmpMat = tarMat[i].reshape(1, tarMat.shape[1])
            ptsEst[i] = get_cosine_ests(refMat, tmpMat, ptsRef)[0]
        else:
            tmpMat = np.repeat(tarMat[i].reshape(1, tarMat.shape[1]), sampleNo, axis=0)
            for j in xrange(sampleNo):
                indices = tmpMat[j].nonzero()[0]
                np.random.shuffle(indices)
                tmpMat[j, indices[:int(len(indices)*(1-sampleRate))]] = 0
            pts = get_cosine_ests(refMat, tmpMat, ptsRef)
            model = AffinityPropagation().fit(pts)
            centers = model.cluster_centers_
            mostFre = Counter(model.labels_).most_common(1)[0][0]
            ptsEst[i] = centers[mostFre][0:2]
    return ptsEst


def GPR(refMat, tarMat, ptsRef, ptsEst, dbThres, apNoThres):
    refMat[refMat==0] = 1
    refMat = 10*np.log2(refMat)
    tarMat[tarMat==0] = 1
    tarMat = 10*np.log2(tarMat)
    ret = []
    for i in xrange(tarMat.shape[1]):
        apTarVec = tarMat[:, i]
        idxTar = apTarVec != 0
        if idxTar.sum() >= apNoThres:
            model = GaussianProcessRegressor(normalize_y=True).fit(ptsEst[idxTar], apTarVec[idxTar])
            apRefVec = refMat[:, i]
            idxRef = apRefVec != 0
            estRssis = model.predict(ptsRef[idxRef])
            idxRet = np.where(np.abs(apRefVec[idxRef]-estRssis) > dbThres)[0]
            for j in idxRet:
                ret.append((int(ptsRef[idxRef][j][0]), int(ptsRef[idxRef][j][1])))
    return ret


def plot_pts(imFile, pts):
    import matplotlib.pyplot as plt
    pts = np.array(list(pts))
    print(pts.shape)
    im = plt.imread(imFile)
    plt.imshow(im, alpha=0.2, zorder=0)
    plt.plot(pts[:, 0], pts[:, 1], 'ro')
    plt.xlim([0, im.shape[1]])
    plt.ylim([im.shape[0], 0])
    plt.savefig('plot.png')


def plot_single_heatmap(imFile, pts):
    import matplotlib.pyplot as plt
    import matplotlib.mlab as ml
    from matplotlib import cm
    occ = Counter(pts)
    mat = np.zeros((len(occ), 3), dtype=np.int32)
    for i, (xy, cnt) in enumerate(occ.iteritems()):
        mat[i] = [xy[0], xy[1], cnt]
    im = plt.imread(imFile)
    image_height, image_width = im.shape[:2]
    num_x = image_width / 5
    num_y = num_x / (image_width / image_height)
    x = np.linspace(0, image_width, num_x)
    y = np.linspace(0, image_height, num_y)
    figure = plt.figure()
    ax = plt.gca()
    z = ml.griddata(mat[:, 0], mat[:, 1], mat[:, 2], x, y, interp='linear')
    cs = ax.contourf(x, y, z, alpha=0.6, zorder=2, cmap=cm.jet)
    plt.colorbar(cs)
    plt.plot(mat[:, 0], mat[:, 1], '+', alpha=0.6, markersize=1.5, zorder=3)
    ax.imshow(im, alpha=0.3, zorder=0)
    plt.savefig('heatmap.png')
    plt.clf()
    plt.close(figure)


def visualize(imFile, ptsTar, ptsEst):
    import matplotlib.pyplot as plt
    # from matplotlib.patches import Rectangle
    im = plt.imread(imFile)
    plt.imshow(im, alpha=0.2, zorder=0)
    plt.plot(ptsTar[:, 0], ptsTar[:, 1], 'o')
    plt.plot(ptsEst[:, 0], ptsEst[:, 1], '+')
    # plt.plot(ptsRef[:, 0], ptsRef[:, 1], 'bo', zorder=1, alpha=0.2)
    # plt.gca().add_patch(Rectangle((3400, 860), 560, 1060, facecolor="red", alpha=0.7, zorder=2))
    plt.xlim([0, im.shape[1]])
    plt.ylim([im.shape[0], 0])
    plt.savefig('result.png')


macIds, idMacs, ptsRef, refMat = load_ref_data(REF)
print(refMat.shape)
ptsTar, tarMat = load_target_data(macIds, TAR)
print(tarMat.shape)

ptsEst = ptsTar
# ptsEst = localise(refMat, tarMat, ptsRef, SAMPLE_RATE, SAMPLE_NO, SHORT_VEC_AP_NUM_THRES)
# # visualize(imFile, ptsTar, ptsEst)
# errs = LA.norm(ptsTar - ptsEst, axis=1)
# print(np.mean(errs), np.std(errs))

problemPts = GPR(refMat, tarMat, ptsRef, ptsEst, BIG_CHANGE_DB_DIFF, GP_AP_NO_THRES)
print(len(problemPts))
plot_pts(imFile, problemPts)
plot_single_heatmap(imFile, problemPts)
print('done')
