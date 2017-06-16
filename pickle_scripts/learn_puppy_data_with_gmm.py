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
trainingData = pickle.load(open("../pickles/newest.pickle", "rb" ))
trainingData = np.hstack((trainingData['y'], trainingData['x']))
print trainingData.shape


variances = np.var(trainingData, axis=0)
means = np.mean(trainingData, axis=0)
delete_rows = []

# delete some crazy datapoints
for i in range(1000):
    for m in range(10):
        if np.abs(trainingData[i,m] - means[m]) > variances[m] * 9:
            delete_rows.append(i)

trainingData = np.delete(trainingData,delete_rows, axis=0)

# delete rows for speed tests
scale_down = 10
trainingData = trainingData[::scale_down,:]

# calculate difference between motor signals and shift sensor data with lag
motors = np.diff(trainingData[:len(trainingData)-3,:4], axis = 0)
sensors = trainingData[4:,4:]
trainingData = np.hstack((motors,sensors))

# whiten
variances = np.var(trainingData, axis=0)
means = np.mean(trainingData, axis=0)
trainingData -= means
trainingData /= variances

print trainingData.shape

trainingData = np.hstack((motors, sensors))

#trainingData = np.average(trainingData.reshape((-1,10,50)), axis = 2)
print trainingData.shape

gmm = None
if(os.path.exists("gmm_sv.pickle")):
    inp = raw_input("old gmm found use it? (y/n)")

    if(inp.lower() == "yes" or inp.lower() == "y"):
        gmm = pickle.load(open("gmm_sv.pickle","rb"))

if gmm == None:
    gmm = IGMM_COND(min_components=10, max_components=20)
    print("training")
    gmm.train(trainingData)
    print("training finished")

pickle.dump(gmm, open("gmm_sv.pickle","wb"))


print("fitting pca")
pca = PCA(n_components=3)
t_data = pca.fit_transform(trainingData)
print("pca finished")

t_means = pca.transform(gmm.means_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scale_down_display = 5
t_data = t_data[::scale_down_display,:]

ax.plot(t_data[:,0],t_data[:,1], t_data[:,2],"ok", alpha=0.1)
ax.plot(t_means[:,0], t_means[:,1],t_means[:,2], "or")


"""
for x in np.linspace(0,1,10):
    for y in np.linspace(0,1,10):
        samples = gmm.sample_cond_dist([x,x,y,y,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 2)
        t_samples = pca.transform(samples)
        ax.plot(t_samples[:,0],t_samples[:,1], t_samples[:,2], linestyle = 'none', marker='.', c = cm.jet((x + y) / 2), alpha = 0.8, markersize=10)
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
for i in range(10):
    print("new batch")
    a = i * 10
    gmm.train(trainingData[a:a+100,:])
"""
