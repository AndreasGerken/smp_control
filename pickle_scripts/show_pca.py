# WORK IN PROGRESS

from igmm_cond import IGMM_COND
import cPickle as pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

motors = None
sensors = None

def prepareData():
    #trainingData = pickle.load(open(""))
    variableDict = pickle.load(open("../pickles/newest.pickle", "rb" ))
    trainingData = np.hstack((variableDict['y'],variableDict['x']))
    print trainingData.shape
    print variableDict['y']


    trainingData -= np.mean(trainingData, axis=0)
    trainingData /= np.max(np.abs(trainingData), axis=0)

    print "100th row", trainingData[100,:]

    delete_rows = []

    # delete some crazy datapoints
    for i in range(1000):
        for m in range(trainingData.shape[1]):
            if np.abs(trainingData[i,m]) > 0.9:
                if not i in delete_rows:
                    delete_rows.append(i)

    print len(delete_rows)

    trainingData = np.delete(trainingData,delete_rows, axis=0)

    print trainingData.shape

    # delete rows for speed tests
    scale_down = 1
    trainingData = trainingData[::scale_down,:]

    trainingData -= np.mean(trainingData, axis=0)
    trainingData /= np.max(np.abs(trainingData), axis=0)

    return trainingData

def showPCA():
    motors = trainingData[:,:4]
    sensors = trainingData[:,4:]

    print("fitting pca")
    pca = PCA(n_components=2)
    sensor_pca = pca.fit_transform(sensors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #scale_down_display = 1
    #sensor_pca = sensor_pca[::scale_down_display,:]
    #print variableDict['y'][::scale_down_display,0].shape
    #print sensor_pca[:,0].shape

    #print "y0"
    #print np.sum(variableDict['y'][:,0])
    #print "s0"
    #print sensor_pca[:,0]
    #print "1"
    #print sensor_pca[:,1]
    #ax.plot(motors[:,0],sensors[:,0], np.zeros_like(sensors[:,0]),"ok", alpha=0.1)
    ax.plot(motors[:,0],sensor_pca[:,0], sensor_pca[:,1], "ok", alpha=0.1)

    plt.show()
    #ax.plot(t_means[:,0], t_means[:,1],t_means[:,2], "or")


    """
    for x in np.linspace(0,1,10):
        for y in np.linspace(0,1,10):
            samples = gmm.sample_cond_dist([x,x,y,y,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 2)
            t_samples = pca.transform(samples)
            ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet((x + y) / 2), alpha = 0.8, markersize=10)
    """
    """
    points_per_dim = 5
    for x in np.linspace(0,1,points_per_dim):
        for y in np.linspace(0,1,points_per_dim):
            for z in np.linspace(0,1,points_per_dim):
                for w in np.linspace(0,1,points_per_dim):
                    samples = gmm.sample_cond_dist([x,y,z,w,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 1)
                    t_samples = pca.transform(samples)

                    #print samples
                    ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet((x + y + z + w )/4), alpha = 0.9, markersize=10)



    for x in np.linspace(0,1,10):
        samples = gmm.sample_cond_dist([x,x,x,x,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 40)

        t_samples = pca.transform(samples)

        ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet(0.2), alpha = 0.8, markersize=10)

    for x in np.linspace(0,1,10):
        samples = gmm.sample_cond_dist([x,0,0,0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 2)
        t_samples = pca.transform(samples)
        ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet(0.4), alpha = 0.8, markersize=10)


    for x in np.linspace(0,1,10):
        samples = gmm.sample_cond_dist([0,x,0,0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 2)
        t_samples = pca.transform(samples)
        ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet(0.4), alpha = 0.8, markersize=10)


    for x in np.linspace(0,1,10):
        samples = gmm.sample_cond_dist([0,0,x,0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 2)
        t_samples = pca.transform(samples)
        ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet(0.4), alpha = 0.8, markersize=10)


    for x in np.linspace(0,1,10):
        samples = gmm.sample_cond_dist([0,0,0,x,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 2)
        t_samples = pca.transform(samples)
        ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet(0.4), alpha = 0.8, markersize=10)

    plt.show()
    """

    """
    for i in range(10):
        print("new batch")
        a = i * 10
        gmm.train(trainingData[a:a+100,:])
    """

def showSensorMotor(trainingData):
    motors = trainingData[:,:4]
    sensors = trainingData[:,4:]


    print "starting plot"
    fig = plt.figure()

    for i in range(sensors.shape[1]):
        print i
        ax = fig.add_subplot(3,np.ceil(sensors.shape[1]/3.),1 + i)
        ax.scatter(motors[:,0], sensors[:,i], lw=0, alpha=0.04)
    plt.show()

def showSensorPCA2D(trainingData):
    motors = trainingData[:,:4]
    sensors = trainingData[:,4:]

    print("fitting pca")
    pca = PCA(n_components=2)
    sensor_pca = pca.fit_transform(sensors)

    fig = plt.figure()
    for i in range(2):
        #   print pca.components_[i]
        ax = fig.add_subplot(221 + (i * 2))
        ax.plot(motors[:,0],sensor_pca[:,i],"ok", alpha=0.4, lw=0)
        ax = fig.add_subplot(222 + (i * 2))
        ax.plot(pca.components_[i])

    plt.show()

def showAllChannels(trainingData):
    plt.subplot(121)
    plt.plot(trainingData[:,:4])
    plt.subplot(122)
    plt.plot(trainingData[:,4:])
    plt.show()

def showVar(trainingData):
    windows = linspace(-1, 1, 10)

    return

if __name__ == "__main__":
    trainingData = prepareData()
    #showAllChannels(trainingData)
    showSensorMotor( trainingData)
    #showPCA()
    #showSensorPCA2D(trainingData)
