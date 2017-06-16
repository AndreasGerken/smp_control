# WORK IN PROGRESS

from igmm_cond import IGMM_COND
import cPickle as pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


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
for i in range(10000):
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

# calculate difference between motor signals and shift sensor data with lag
#motors = np.diff(trainingData[:len(trainingData)-3,:4], axis = 0)
#sensors = trainingData[4:,4:]
#trainingData = np.hstack((motors,sensors))

# whiten
variances = np.var(trainingData, axis=0)
means = np.mean(trainingData, axis=0)
trainingData -= means
trainingData /= variances

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
ax.plot(motors[:,0],sensors[:,0], np.zeros_like(sensors[:,0]),"ok", alpha=0.1)
#ax.plot(motors[:0],sensor_pca[:,0], np.zeros_like(sensor_pca[:,0]), "ok", alpha=0.1)

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
