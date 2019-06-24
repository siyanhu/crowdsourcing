import numpy as np
import scipy.optimize
import sys
import random
from math import *
import scipy.linalg

gN = 0                              #size of training data
gKy = np.array([[]])                #invese of Ky, or temp (xp - xq)T(xp - xq), N*N
gT = np.array([[]])                 #temp matrix, N*N
gX = np.array([[]])                 #training input data X, N*2
gY = np.array([[]])                 #training input data Y, or temp (Ky-1)(Y - m(X)), N*1
g_pGP = np.array([])                #GP parameters: sigma_n, sigma_f, l
g_pLDPL = np.array([])              #LDPL parameters: A, B, x_ap, y_ap, (z_ap)

DBL_MAX=sys.float_info.max
DBL_MIN=sys.float_info.min
withZ=False

def GP(position,rssi):
    gN = 0                              #size of training data
    gKy = np.array([[]])                #invese of Ky, or temp (xp - xq)T(xp - xq), N*N
    gT = np.array([[]])                 #temp matrix, N*N
    gX = np.array([[]])                 #training input data X, N*2
    gY = np.array([[]])                 #training input data Y, or temp (Ky-1)(Y - m(X)), N*1
    g_pGP = np.array([])                #GP parameters: sigma_n, sigma_f, l
    g_pLDPL = np.array([])              #LDPL parameters: A, B, x_ap, y_ap, (z_ap)

    withZ=False
    LoadData(position,rssi)
    TrainLDPL() #Log distance path loss
    TrainGP()

def LoadData(position,rssi):
    global gN, gX, gY, gT, gKy
    gN = len(rssi)
    gY.resize(gN,1)
    gX.resize(gN,2)
    gT.resize(gN,gN)
    gKy.resize(gN,gN)

    for i in range(gN):
        gY[i,0] = rssi[i]
        gX[i,0] = position[i][0]
        gX[i,1] = position[i][1]

def TrainLDPL():
    global g_pLDPL
    maxN = 10

    bv = -100.0
    bx = 0.0
    by = 0.0
    for i in range(gN):
        if gY[i,0] > bv:
            bv = gY[i,0]
            bx = gX[i,0]
            by = gX[i,1]

    bestObj = DBL_MAX
    for i in range(maxN):
        pLDPL = np.array([0.0,0.0,0.0,0.0])
        if withZ:
            pLDPL.resize(5)
            #randomly initialize
            pLDPL[4] = 0.1 + 0.1 * random.randrange(10)

        pLDPL[0] = -20.0 - 0.1 * random.randrange(100)
        pLDPL[1] = -20.0 + 0.1 * random.randrange(100)
        pLDPL[2] = bx + 0.1 * random.randrange(80) - 4.0
        pLDPL[3] = by + 0.1 * random.randrange(80) - 4.0

        x, f, d = scipy.optimize.fmin_l_bfgs_b(ObjLDPL,x0=pLDPL,fprime=DerivLDPL,factr=1e7)   # m = 10
        #see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

        if f < bestObj:
            bestObj = f
            g_pLDPL = x

    #print('LDPL parameters: ',g_pLDPL)

def ObjLDPL(ldpl):
    #minimize E = sum (m(xi) - yi)^2
    global gT
    ans = 0.0
    for r in range(gN):
        temp = EstimateLDPL(ldpl,gX[r,0],gX[r,1]) - gY[r,0]
        gT[r,0] = temp
        ans += temp * temp

    return ans

def EstimateLDPL(ldpl,x,y):    # return m(x)
    temp = SqdSum(ldpl[2],ldpl[3],x,y)
    if withZ:
        temp += ldpl[4] * ldpl[4]
    return ldpl[0] + ldpl[1] * log10(sqrt(temp) + DBL_MIN)

def DerivLDPL(ldpl):
    #equation (43) (44)
    ans = np.array([0.0,0.0,0.0,0.0])
    if withZ:
        ans.resize(5)
        ans[4] = 0.0
    for r in range(gN):
        sqdsum = SqdSum(ldpl[2],ldpl[3],gX[r,0],gX[r,1])
        if withZ:
            sqdsum += ldpl[4] * ldpl[4]
        dfB = log10(sqrt(sqdsum) + DBL_MIN)
        temp = ldpl[1] / (sqdsum + DBL_MIN)
        dfx = (ldpl[2] - gX[r,0]) * temp
        dfy = (ldpl[3] - gX[r,1]) * temp
        f = gT[r,0]
        ans[0] += f
        ans[1] += f * dfB
        ans[2] += f * dfx
        ans[3] += f * dfy
        if withZ:
            ans[4] += f * temp
    return  2.0 * ans

def TrainGP():
    global gY, gKy, g_pGP
    maxN = 5

    #gY = z = v - m(x)
    for r in range(gN):
        gY[r,0] -= EstimateLDPL(g_pLDPL,gX[r,0],gX[r,1])

    #gKy = tpq = (xp - xq)T (xp - xq)
    for p in range(gN):
        for q in range(p+1,gN):
            gKy[p,q] = SqdSum(gX[p,0],gX[p,1],gX[q,0],gX[q,1])

    bestObj = DBL_MAX
    for i in range(maxN):
        pGP = np.array([0.0, 0.0, 0.0])

        pGP[0] = 1.5 + 0.1 * random.randrange(10)
        pGP[1] = 2.0 + 0.1 * random.randrange(10)
        pGP[2] = 16 + 0.5 * random.randrange(20)
        """
        print('fmin_l_bfgs_b is running, i = ',end='')
        print(i)
        """
        x, f, d = scipy.optimize.fmin_l_bfgs_b(ObjGP,x0=pGP,fprime=DerivGP,factr=1e7)

        if f < bestObj:
            bestObj = f
            g_pGP = x
    #print('GP parameters: ', g_pGP)

    # preprocess gKy = Ky-1
    EstimateKy(g_pGP,gKy)
    gKy = scipy.linalg.cholesky(gKy,lower=True)
    gKy = scipy.linalg.inv(gKy)
    gKy = np.dot(gKy.transpose(),gKy)

    # gY = (Ky-1) (y-m(x))
    gY = np.dot(gKy,gY)

    gT.resize(0,0)


def ObjGP(gp):
    global gT

    # minimize -p = 0.5 * (zT (Ky-1) z + 2 * sum log(Lii))
    # gT = Ky
    EstimateKy(gp,gT)

    #do cholesky decomp
    gT = scipy.linalg.cholesky(gT,lower=True)

    # log|Ky| = 2 * sum log(Lii)
    ans = 0.0
    for i in range(gN):
        ans += 2 * log(gT[i,i])

    gT = scipy.linalg.inv(gT)
    gT = np.dot(gT.transpose(),gT)   # gT = Ky-1 = (L-1)T (L-1)

    # ans += zT (Ky-1) z
    ans += np.dot(np.dot(gY.transpose(),gT),gY)
    #print(ans)
    return ans * 0.5

def DerivGP(gp):
    # -dp = 0.5 * tr{((Ky-1) - [Ky-1 * z * zT * (Ky-1)T]) * dKy}
    # gKy = tpq
    global gT
    dKf = np.array([[]])
    dKd = np.array([[]])
    dKf.resize(gN,gN)
    dKd.resize(gN,gN)
    for p in range(gN):
        dKf[p,p] = 1.0
        dKd[p,p] = 0.0
        for q in range(p+1,gN):
            temp = exp(-0.5 * gKy[p,q] / (gp[2] * gp[2]))
            dKf[p,q] = temp
            dKf[q,p] = temp
            temp *= gKy[p,q]
            dKd[p,q] = temp
            dKd[q,p] = temp

    # gT = (Ky-1)  gY = z
    # compute (Ky-1) - [Ky-1 * z * zT * (Ky-1)T]
    gT -= np.dot(np.dot(gT,gY),np.dot(gT,gY).transpose())

    ans = np.array([0.0,0.0,0.0])
    ans[0] = np.trace(gT) * gp[0]
    ans[1] = np.trace(np.dot(gT,dKf)) * gp[1]
    ans[2] = 0.5 * np.trace(np.dot(gT,dKd)) * gp[1] * gp[1] / (gp[2] * gp[2] * gp[2])
    return ans

def EstimateKy(gp,Ky):
    for p in range(gN):
        Ky[p,p] = gp[0] * gp[0] + gp[1] * gp[1]
        for q in range(p+1,gN):
            temp = kfunc(gp[1],gp[2],gKy[p,q])
            Ky[p,q] = temp
            Ky[q,p] = temp

def kfunc(sigma_f,d,sqd_sum):
    return sigma_f * sigma_f * exp(-sqd_sum / (2 * d * d))

def SqdSum(x1,y1,x2,y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

#prediction
def EstimateGP(x,y):
    # f* = m(x*) + k(x*, X)T (Ky-1) (Y - m(X))
    # compute m(x*)
    ans = EstimateLDPL(g_pLDPL,x,y)

    # compute k(x*, X)T
    kT = np.array([[]])
    kT.resize(1,gN)
    for i in range(gN):
        kT[0,i] = kfunc(g_pGP[1],g_pGP[2],SqdSum(x,y,gX[i,0],gX[i,1]))

    return ans + np.dot(kT,gY)

