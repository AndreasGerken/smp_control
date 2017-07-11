import cPickle as pickle
import warnings
import time
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import datasets, linear_model
from sklearn import kernel_ridge
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neural_network import MLPRegressor


pickleFolder = '../pickles_new_body/'
#pickleFolder = '../goodPickles/'

A = None
C = None
b = None
h = None
i = 0

motorGlobal = None
sensorGlobal = None


class Analyzer():
    def __init__(self, args):
        self.args = args

        if args.randomFile:
            files = [f for f in self.all_files(pickleFolder)
                     if f.endswith('.pickle') and 'recording' in f]
            print len(files)
            self.filename = files[np.random.randint(len(files))]
            print "random File: %s" % (self.filename)
        else:
            self.filename = self.args.filename

        try:
            self.variableDict = pickle.load(open(self.filename, "rb"))
        except Exception:
            raise Exception("File not found, use -f and the filepath")

        # if self.variableDict["dataversion"] == 1:
        #    warnings.warn("Pickle from V1, xsi might not be correct")

        # if self.variableDict["dataversion"] >= 9:
        #    self.hz = self.variableDict["hz"]
        self.motorCommands = self.variableDict["y"]
        self.sensorValues = self.variableDict["x"]
        self.timesteps = self.motorCommands.shape[0]
        self.nummot = self.motorCommands.shape[1]

        self.windowsize = self.args.windowsize
        self.embsize = self.args.embsize
        self.hamming = self.args.hamming

    def all_files(self, directory):
        for path, dirs, files in os.walk(directory):
            for f in files:
                yield os.path.join(path, f)

    def get_triu_of_matrix(self, matrix):
        if matrix.shape[0] != matrix.shape[1]:
            return None

        dim = matrix.shape[0]
        triu = np.triu_indices(dim, k=1)
        return matrix[triu]

    def correlation_func(self):
        loopsteps = self.timesteps - self.windowsize

        #self.motorCommands += np.random.normal(size=self.motorCommands.shape) * 0.1

        # testdata
        # self.motorCommands = np.arange(20).reshape(10,2)
        # print self.motorCommands

        # print(len(self.motorCommands))

        allCorrelations = np.zeros((loopsteps, self.nummot, self.nummot))
        allAverageSquaredCorrelations = np.zeros((loopsteps, 6))

        correlationsGlobal = np.cov(self.motorCommands.T)
        print correlationsGlobal

        for i in range(len(self.motorCommands) - self.windowsize):
            # use only the latest step of the buffer but all motor commands
            # shape = (timestep, motorChannel, buffer)

            window = self.motorCommands[i:i + self.windowsize, :]
            # print "windooriginal", window

            windowfunction = np.hamming(self.windowsize)
            if(self.hamming):
                window = (window.T * windowfunction).T
            # print "windowshapehamming", window

            correlations = np.cov(window.T)

            # normalize
            for x in range(4):
                for j in range(4):
                    correlations[x, j] = correlations[x, j] / \
                        np.sqrt(correlationsGlobal[x, x]
                                * correlationsGlobal[j, j])

            if self.hamming:
                correlations[:, :] *= self.windowsize / np.sum(windowfunction)
            allCorrelations[i, :, :] = correlations[:, :]

            # save average of the squared upper triangle of the correlations

            # allAverageSquaredCorrelations[i,:] = np.triu(correlations,k=1).flatten() ** 2
            #allAverageSquaredCorrelations[i,:] = np.triu(correlations,k=1).flatten()

            #allAverageSquaredCorrelations[i,0] = np.sum(np.triu(correlations,k=1).flatten() ** 2)
            #allAverageSquaredCorrelations[i,:] = np.abs(np.triu(correlations,k=1).flatten())
            allAverageSquaredCorrelations[i, :] = self.get_triu_of_matrix(
                correlations)

        corrCombinations = allAverageSquaredCorrelations.shape[1]
        # print "corrCombinations", allAverageSquaredCorrelations[0,:]

        combinationNames = ["rb-lb", "rb-rf",
                            "rb-lf", "lb-rf", "lb-lf", "rf-lf"]
        numtimesteps = allAverageSquaredCorrelations.shape[0]

        colors = cm.jet(np.linspace(0, 1, corrCombinations))
        colors = ["#FFFF00", "#FF0000", "#FF00FF",
                  "#0000FF", "#00FFFF", "#00FF00"]

        fig = plt.figure(figsize=(15, 5))

        xValues = (np.arange(
            len(allAverageSquaredCorrelations[:, j])) + (self.windowsize // 2)) / 20.

        plt.plot(xValues, [0] * len(xValues), 'k', alpha=0.3)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        print np.average(allAverageSquaredCorrelations[:, 0])

        for j in range(corrCombinations):
            plt.plot(
                xValues, allAverageSquaredCorrelations[:, j], alpha=1, c=colors[j], label=combinationNames[j])

        #plt.plot(xValues, np.average(np.abs(allAverageSquaredCorrelations[:,:]), axis=1), c='k', lw = 2, label="average")
        # plt.legend()

        """
        print maximum
        t= 0
        tstart = 0
        printingChannel = -1
        while t < numtimesteps:
            maxChannel = np.argmax(allAverageSquaredCorrelations[t,:])
            if maxChannel != printingChannel or t == numtimesteps - 1:
                if printingChannel != -1:
                    # plot old channel
                    if tstart < 0:
                        tstart = 0
                    plt.plot(np.arange(tstart + 50, t + 50), allAverageSquaredCorrelations[tstart:t ,printingChannel], c = colors[printingChannel], lw=3)

                # change to new printingChannel
                printingChannel = maxChannel
                tstart = t - 1
            t+=1
        """

        # plt.plot(allAverageSquaredCorrelations[:,0])
        #plt.title("Correlation from a episode with")
        plt.xlabel("time [s]", fontsize=28)
        plt.ylabel("correlation", fontsize=28)
        # plt.ylim((-0.5,0.5))
        plt.xlim(0, self.timesteps / 20)
        plt.xticks(np.linspace(0, 50, 11))
        plt.tight_layout()
        fig.savefig('correlations.png')
        plt.show()
        # print allAverageSquaredCorrelations

    def prepare_data_for_learning(self):
        testDataLimit = 4 * self.timesteps / 5

        motoremb = np.array([self.motorCommands[i:i + self.embsize].flatten()
                             for i in range(0, testDataLimit - self.embsize)])
        motorembtest = np.array([self.motorCommands[i:i + self.embsize].flatten()
                                 for i in range(testDataLimit, self.timesteps - self.embsize)])
        #self.sensorValues = self.sensorValues[:,3:]

        # motoremb = self.motorCommands[:testDataLimit]

        self.trainingData = {
            "motor": motoremb, "sensor": self.sensorValues[self.embsize:testDataLimit]}
        self.testData = {"motor": motorembtest,
                         "sensor": self.sensorValues[testDataLimit + self.embsize:]}

    def learn_motor_sensor_gmm(self):
        """
        This mode learns the sensor responses from the motor commands with
        gaussian mixture models and tests the result
        """
        # TODO: implement

        for emb in range(1, 100):
            self.embsize = emb

            x = []
            score = []
            avg_x = []
            avg_sc = []

            self.prepare_data_for_learning()
            for k in range(5, 6):
                sum_sc = 0
                for i in range(0, 10):
                    gmm = GMM(n_components=k, covariance_type='full')
                    gmm.fit(self.trainingData["motor"],
                            self.trainingData["sensor"])
                    sc = gmm.score(
                        self.testData["motor"], self.testData["sensor"])
                    print "%d score %.2f" % (k, sc)
                    x.append(k)
                    score.append(sc)
                    sum_sc += sc
                avg_x.append(k)
                avg_sc.append(sum_sc / 10.)

            plt.plot(x, score, 'o', label=emb)
            plt.plot(avg_x, avg_sc)
        plt.legend()
        plt.show()

        return 0

    def learn_motor_sensor_mlp_cross(self):
        fig = plt.figure()
        for lag in range(1, 6):

            mot_lag = self.motorCommands[0:-1 * lag]
            sen_lag = self.sensorValues[lag:]
            n = mot_lag.shape[0]
            nummot = mot_lag.shape[1]
            numsen = sen_lag.shape[1]
            split = int(np.floor(n / 5.))

            print split
            print mot_lag.shape
            print sen_lag.shape

            sen_pred = np.zeros((0, numsen))

            for i in range(5):
                regr = MLPRegressor(solver='lbfgs', alpha=1e-4,
                                    hidden_layer_sizes=(10, 1), random_state=1)
                start = i * split
                end = (i + 1) * split

                mot_train = np.concatenate((mot_lag[0:start], mot_lag[end:]))
                mot_test = mot_lag[start:end]

                sen_train = np.concatenate((sen_lag[0:start], sen_lag[end:]))
                sen_test = sen_lag[start:end]

                regr.fit(mot_train, sen_train)
                sen_pred = np.vstack((sen_pred, regr.predict(mot_test)))

            for i in range(numsen):
                ax = fig.add_subplot(numsen, 5, (i * 5) + lag)
                plt.plot(sen_lag[:, i], c='k')
                plt.plot(sen_pred[:, i], c='r', alpha=0.7)
        plt.show()

    def learn_motor_sensor_linear_cross(self):
        fig = plt.figure()
        for lag in range(1, 6):

            mot_lag = self.motorCommands[0:-1 * lag]
            sen_lag = self.sensorValues[lag:]
            n = mot_lag.shape[0]
            nummot = mot_lag.shape[1]
            numsen = sen_lag.shape[1]
            split = int(np.floor(n / 5.))

            print split
            print mot_lag.shape
            print sen_lag.shape

            regr = linear_model.Ridge(alpha=lag)

            sen_pred = np.zeros((0, numsen))

            for i in range(5):
                start = i * split
                end = (i + 1) * split

                mot_train = np.concatenate((mot_lag[0:start], mot_lag[end:]))
                mot_test = mot_lag[start:end]

                sen_train = np.concatenate((sen_lag[0:start], sen_lag[end:]))
                sen_test = sen_lag[start:end]

                regr.fit(mot_train, sen_train)
                sen_pred = np.vstack((sen_pred, regr.predict(mot_test)))

            for i in range(numsen):
                ax = fig.add_subplot(numsen, 5, (i * 5) + lag)
                plt.plot(sen_lag[:, i], c='k')
                plt.plot(sen_pred[:, i], c='r', alpha=0.7)
        plt.show()

    def learn_motor_sensor_linear(self):
        """
        This mode learns the sensor responses from the motor commands with
        linear regression and tests the result
        """

        self.prepare_data_for_learning()

        # regr = linear_model.LinearRegression()
        regr = linear_model.Ridge(alpha=10.0)
        #regr = kernel_ridge.KernelRidge(alpha = 0.5, kernel="rbf")
        regr.fit(self.trainingData["motor"], self.trainingData["sensor"])

        # A,b Matrices from online learner for
        # python analyze_pickle.py -f ../goodPickles/recording_eC0.70_eA0.01_c0.50_n1000_id0.pickle -m lin -es 1

        """
        regr.coef_ = np.array([[ 0.03122893, -0.01962895, -0.08606354, -0.17203238],
                         [-0.01257594, -0.19804437, -0.05068203,  0.03037255],
                         [-0.02674586,  0.01104205,  0.02475127,  0.09052899],
                         [ 0.25470938,  0.15598333,  0.1278085,  0.20791332],
                         [ 0.15407881, -0.09799728, -0.07484991, -0.04812915],
                         [-0.36372194,  0.40417843, -0.22249272, -0.01279415]])

        regr.intersept_ = np.array([0.23388626,-0.46302744, -3.22338203, 0.00920198, -0.07409651, -0.03185829])
        """

        # predict the sensor data from the motor data
        predTest = regr.predict(self.testData["motor"])

        print "coeffs: " + str(regr.coef_) + "\n"

        # find the absolute sum of the coeffitients corresponding to the lag coefs[max] is zero lag
        coefs = np.sum(np.sum(np.abs(regr.coef_), axis=0).reshape(
            (4, self.embsize)), axis=0)
        print coefs
        print np.argmax(coefs)

        # calculate mean squared error of the test data
        mse = np.mean((predTest - self.testData["sensor"]) ** 2)

        #print("trainingError: %.2f" %np.mean((regr.predict(trainingData["motor"]) - trainingData["sensor"]) ** 2))
        print("Mean squared error: %.2f, s = %s" %
              (mse, np.var(self.testData["sensor"], axis=0)))

        # plot the coefficient sum
        plt.plot(coefs, label="coefefs over lag")

        fig, axs = plt.subplots(nrows=3, ncols=4)

        bins = np.linspace(-2, 2, 15)
        for j, ax in enumerate(axs.reshape(-1)):
            i = j / 2

            predError = predTest[:, i] - self.testData["sensor"][:, i]

            # get the sensor values without mean
            zmData = self.testData["sensor"][:, i] - \
                np.mean(self.testData["sensor"][:, i])

            # plot time series
            if j % 2 == 0:
                ax.plot(self.testData["sensor"][:, i])
                ax.plot(predTest[:, i])
            else:
                """
                plot a histogram of the prediction error and the zero mean
                data, the prediction error should be sharper than the signal
                distribution
                """
                data = np.vstack([predError, zmData]).T
                ax.hist(data, bins=bins, alpha=1, label=[
                        "predictionError", "meanedData"])
                ax.legend()

        plt.show()

    def find_best_embsize(self):
        mse = np.zeros(39)
        regr = linear_model.Ridge(alpha=10.0)
        for emb in range(1, 40):
            self.embsize = emb
            self.prepare_data_for_learning()
            regr.fit(self.trainingData["motor"], self.trainingData["sensor"])
            predTest = regr.predict(self.testData["motor"])
            mse[emb - 1] = np.mean((predTest - self.testData["sensor"]) ** 2)

        plt.plot(mse)
        plt.show()

    def scattermatrix_func(self):
        """
        This function creates a scatterplot of the data
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from pandas.tools.plotting import scatter_matrix
        import matplotlib.cm as cm

        # settings, ammount of points in one graph and color mode
        points_per_plot = 1000  # self.timesteps
        color = False

        if color:
            c = np.linspace(0, 1, points)
        else:
            c = None

        # go through the data until the end
        plots = int(self.timesteps / points_per_plot)
        print "Warning: %d points not shown" % (self.timesteps % points_per_plot)

        for part in range(0, plots):
            partS = part * points_per_plot
            partE = (part + 1) * points_per_plot
            combinedData = np.hstack(
                (self.motorCommands[partS:partE, :], self.sensorValues[partS:partE, :]))

            df = pd.DataFrame(combinedData, columns=[
                              'm1', 'm2', 'm3', 'm4', 's1', 's2', 's3', 's4', 's5', 's6'])

            scatterm = scatter_matrix(df, alpha=0.4, s=25, figsize=(
                12, 12), diagonal='kde', c=c, edgecolors='none')

            for x in range(10):
                for y in range(10):
                    scatterm[x, y].set_ylim(-1.5, 1.5)
                    scatterm[x, y].set_xlim(-1.5, 1.5)

            # plt.draw()
            # plt.show()
            #plt.savefig("scattermatrix%d.jpg" %(part))
            #print("scatter %d saved" %(part))
            plt.show()

    def spectogram_func(self):
        loopsteps = self.timesteps - self.windowsize

        sensors = self.sensorValues[...]
        self.numsen = sensors.shape[1]
        s = sensors

        print sensors.shape

        import scipy.signal as sig

        m = self.motorCommands
        s = self.sensorValues

        Mspecs = [sig.spectrogram(m[:, i], fs=20.0, nperseg=32, nfft=32)
                  for i in range(self.nummot)]
        Sspecs = [sig.spectrogram(s[:, i], fs=20.0, nperseg=32, nfft=32)
                  for i in range(self.numsen)]

        allSpecs = np.zeros((35, 0))

        from matplotlib import gridspec
        gs = gridspec.GridSpec(3, max(self.nummot, self.numsen) + 1)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(sensors)
        for i in range(self.numsen):
            ax = fig.add_subplot(gs[0, 1 + i])

            Mspec = Sspecs[i]
            ax.pcolormesh(Mspec[1], Mspec[0], Mspec[2])

            allSpecs = np.concatenate((allSpecs, Mspec[2].T), axis=1)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(m)
        for i in range(self.nummot):
            ax = fig.add_subplot(gs[1, 1 + i])

            Mspec = Mspecs[i]
            ax.pcolormesh(Mspec[1], Mspec[0], Mspec[2])

            allSpecs = np.concatenate((allSpecs, Mspec[2].T), axis=1)

        # do k-means
        from sklearn.cluster import KMeans
        import matplotlib.cm

        kmeans = KMeans(n_clusters=4, random_state=1)
        kmeans.fit(allSpecs)
        ax3 = fig.add_subplot(gs[2, 1])

        ax3.scatter(range(len(allSpecs)), kmeans.predict(
            allSpecs), c=np.linspace(0, 255, 35))

        from sklearn import decomposition

        pca = decomposition.PCA(n_components=2)
        pca.fit(allSpecs)
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.scatter(pca.transform(allSpecs)[:, 0], pca.transform(
            allSpecs)[:, 1], c=np.linspace(0, 255, 35))

        print(allSpecs.shape)
        plt.show()

        print sensors.shape

    def activity_plot(self):
        if self.windowsize % 2 != 0:
            print("please choose a even windowsize")
            return

        distance_array = np.zeros((self.timesteps, 6))
        windowfunction = np.hamming(self.windowsize)

        self.sensorValues -= np.mean(self.sensorValues, axis=0)
        for i in range(0, self.timesteps - self.windowsize):
            start = i
            end = i + self.windowsize

            min_x = np.min(self.sensorValues[start:end, :], axis=0)
            max_x = np.max(self.sensorValues[start:end, :], axis=0)
            distance_array[i, :] = max_x - min_x
        # plt.plot(self.sensorValues[:,1])
        # plt.plot(self.sensorValues[:,1])
        #plt.plot(distance_array, self.hz[self.windowsize/2: self.timesteps - self.windowsize/2])
        dist_avg = [np.average(distance_array[i:i + 60, :], axis=0)
                    for i in range(self.timesteps - 60)]
        lines = plt.plot(np.linspace(
            0, 6.34, self.timesteps - 70), dist_avg[10:])
        plt.legend(lines, ("ax", "ay", "az", "gx", "gy", "gz"))
        plt.show()

    def activity_plot2(self):

        windowfunction = np.hamming(self.windowsize)
        sv_abs = np.sum(np.abs(self.sensorValues), axis=1)
        sv_abs[0] = sv_abs[1]
        x = np.linspace(0, 1, self.timesteps)

        z = np.polyfit(x, sv_abs, 10)
        p = np.poly1d(z)

        plt.plot(x, sv_abs, 'o', alpha=0.2, linewidth=0)
        plt.plot(x, p(x), linewidth=2)
        plt.show()

        sv_smooth = np.zeros(self.timesteps - self.windowsize)

        for i in range(0, self.timesteps - self.windowsize):
            sv_smooth[i] = np.sum((sv_abs[i:i + self.windowsize] *
                                   windowfunction)) / np.sum(windowfunction) - (i / 10000000.)

        plt.plot(sv_smooth)
        plt.show()

    def animate_scatter(self):
        import matplotlib.animation as animation
        global motorGlobal, sensorGlobal, i
        i = 0
        fig = plt.figure(self.filename)

        motorGlobal = self.motorCommands[:, 0]
        sensorGlobal = self.sensorValues

        datax = motorGlobal[i:i + 100]
        datay1 = sensorGlobal[i:i + 100, 0]
        datay2 = sensorGlobal[i:i + 100, 2]
        datay3 = sensorGlobal[i:i + 100, 3]

        ax1 = plt.subplot(131)
        scat1 = plt.scatter(datax, datay1, animated=True)
        # plt.set_xlim(1000,2000)
        # plt.set_ylim(1000,2000)

        ax2 = plt.subplot(132)
        scat2 = plt.scatter(datax, datay2, animated=True)

        ax3 = plt.subplot(133)
        scat3 = plt.scatter(datax, datay3, animated=True)

        ax1.set_xlim(np.min(motorGlobal), np.max(motorGlobal))
        ax2.set_xlim(np.min(motorGlobal), np.max(motorGlobal))
        ax3.set_xlim(np.min(motorGlobal), np.max(motorGlobal))

        ax1.set_ylim(np.min(sensorGlobal[:, 0]), np.max(sensorGlobal[:, 0]))
        ax2.set_ylim(np.min(sensorGlobal[:, 1]), np.max(sensorGlobal[:, 1]))
        ax3.set_ylim(np.min(sensorGlobal[:, 2]), np.max(sensorGlobal[:, 2]))

        def updatefig(*args):
            global motorGlobal, sensorGlobal, i
            if(i < self.timesteps - 101):

                pts = 100
                data1 = np.vstack(
                    (motorGlobal[i:i + pts], sensorGlobal[i:i + pts, 0])).T
                data2 = np.vstack(
                    (motorGlobal[i:i + pts], sensorGlobal[i:i + pts, 1])).T
                data3 = np.vstack(
                    (motorGlobal[i:i + pts], sensorGlobal[i:i + pts, 2])).T

                scat1.set_offsets(data1)
                scat2.set_offsets(data2)
                scat3.set_offsets(data3)

                # scat1.set_ydata()

                # scat2.set_xdata(motorGlobal[i:i+100])
                # scat2.set_ydata(sensorGlobal[i:i+100,1])

                # scat3.set_xdata(motorGlobal[i:i+100])
                # scat3.set_ydata(sensorGlobal[i:i+100,2])

                print "i = %d" % (i)
                i += pts / 4

            return scat1, scat2, scat3,
        ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
        plt.show()

    def time_series_smoothing(self):
        x = self.variableDict["x"]
        y = self.variableDict["y"]
        sensors = 3
        fig = plt.figure()
        for i in range(sensors):
            fig.add_subplot(sensors, 1, i + 1)
            plt.plot(x[:, i::sensors])

        plt.show()

    def details(self):
        epsA = self.variableDict["epsA"]
        epsC = self.variableDict["epsC"]
        creativity = self.variableDict["creativity"]
        x = self.variableDict["x"]
        y = self.variableDict["y"]
        xsi = self.variableDict["xsi"]
        ee = self.variableDict["EE"]
        print "epsA\t", epsA
        print "epsC\t", epsC
        print "Creativity\t", creativity

    def step_sweep(self):
        x = self.variableDict["x"]
        y = self.variableDict["y"]
        x -= np.mean(x, axis=0)
        x /= np.std(x, axis=0)
        x = np.abs(x)
        f, axarr = plt.subplots(1, 1)
        colors = cm.Greys(np.linspace(0, 1, 19))
        for i in range(19):
            axarr.plot(np.average(x[i * 50:i * 50 + 25], axis=1), c=colors[i])

            plt.xticks(np.arange(0, 25, 1.0))
            #axarr[0].plot(y[i*50:i*50 + 20], c= colors[i])
            #plt.xticks(np.arange(0, 20, 1.0))
        plt.show()

    def time_series(self):
        cut = 2000
        x = self.variableDict["x"][:cut]
        y = self.variableDict["y"][:cut]
        xsi = self.variableDict["xsi"][:cut]
        ee = self.variableDict["EE"][:cut]
        # print self.variableDict["robot.sensor_dimensions"]

        f, axarr = plt.subplots(4, 1)

        sensor_names = ["acc_x", "acc_y",
                        "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        for sen in range(x.shape[1]):
            axarr[0].plot(x[:, sen], label=sensor_names[sen])
        axarr[0].set_title("Sensors")
        axarr[0].legend()

        axarr[1].plot(y)

        for i in range(xsi.shape[1]):
            axarr[2].plot(xsi[:, i, 0])
        axarr[2].set_title("xsi")

        # smoothing_window_C = 500
        # #for sen in range(xsi.shape[1]):
        # tmp = np.average(np.abs([np.average(xsi[i+50:i+50+smoothing_window_C,:], axis=0) for i in range(xsi.shape[0]-smoothing_window_C-50)]),axis=1)
        # axarr[2].plot(np.abs(tmp))
        # axarr[2].set_title("xsi")

        axarr[3].plot(ee)

        f.subplots_adjust(hspace=0.3)
        plt.legend()
        plt.show()

    def split_sensors_effects_smooth(self):
        # TODO: not functional
        # C=self.variableDict["C"]
        # x=self.variableDict["x"]
        # h=self.variableDict["h"]
        # sensors=3
        # yp = np.zeros((x.shape[0],sensors + 1))
        #
        # print C.shape
        # print x.shape
        # print h.shape
        #
        #
        # for j in range(sensors):
        #     Cp = C[:,0,j::sensors]
        #     hp = h[:,:]
        #     xp = x[:,j::sensors]
        #
        #     for i in range(x.shape[0]):
        #         yp[i,j] = np.dot(Cp[i], xp[i].T) + hp[i]
        #         if(i>0):
        #             yp[i,j] = yp[i,j] * 0.2 + yp[i-1,j] * 0.8
        #
        # yp[:,3] = yp[:,0] + yp[:,1] + yp[:,2]
        #
        # plt.plot(yp)
        # plt.show()
        return

    def model_matrix_plot_smooth(self):
        C = self.variableDict["C"]
        A = self.variableDict["A"]
        sensors = 3

        fig = plt.figure()
        ax = None
        for j in range(sensors):
            abs_max = np.std(self.variableDict["x"][:, j + sensors * 2])
            #abs_max = np.max(np.abs(self.variableDict["x"][:,j+sensors*2]))
            ax = fig.add_subplot(1, sensors, j + 1, sharey=ax)
            plt.plot(C[:, 0, j::sensors] * abs_max)
            # plt.legend()
        # plt.subplot(122)
        # for i in range(A.shape[1]):
        #    for j in range(A.shape[2]):
        #        plt.plot(A[:,i,j], label="A "+ str(i) + " " + str(j))
        # plt.legend()
        plt.show()

    def model_matrix_plot(self):
        onePlot = False

        C = self.variableDict["C"]
        A = self.variableDict["A"]
        x = self.variableDict["x"]
        y = self.variableDict["y"]
        x_std = np.std(x, axis=0)
        y_std = np.std(y, axis=0)

        use_sensors = self.variableDict["robot.use_sensors"]
        sensor_dimensions = self.variableDict["robot.sensor_dimensions"]

        # create list of sensor names
        sensor_names = []
        for sensor in use_sensors:
            # repeat the sensor name with an identifier as often as the sensor has dimensions
            sensor_names.extend([sensor + str(j)
                                 for j in range(sensor_dimensions[sensor])])

        colors = cm.jet(np.linspace(0, 1, len(sensor_names)))
        #colors = ['r','r','r','b','b','b','k','k','k','k']

        cut_s = 0
        cut_e = C.shape[0]
        smoothing_window_C = 1
        alpha_C = 0.5

        if onePlot:
            plt.subplot(121)
        for mot in range(C.shape[1]):
            for sen in range(C.shape[2]):
                if not onePlot:
                    plt.subplot(C.shape[1], 2, mot * 2 + 1)

                tmp = [np.average(C[i:i + smoothing_window_C, mot, sen])
                       for i in range(C.shape[0] - smoothing_window_C)]
                tmp = np.array(tmp[cut_s:cut_e]) / y_std[mot]
                if(mot == 0):
                    plt.plot(tmp, label="C " +
                             sensor_names[sen], c=colors[sen], alpha=alpha_C)
                else:
                    plt.plot(tmp, c=colors[sen], alpha=alpha_C)

        plt.legend()
        smoothing_window_A = 1
        alpha_A = 0.5
        if onePlot:
            plt.subplot(122)
        for mot in range(A.shape[2]):
            for sen in range(A.shape[1]):
                if not onePlot:
                    plt.subplot(C.shape[1], 2, mot * 2 + 2)
                tmp = [np.average(A[i:i + smoothing_window_A, sen, mot])
                       for i in range(C.shape[0] - smoothing_window_A)]
                tmp = np.array(tmp[cut_s:cut_e]) / x_std[sen]
                if(mot == 0):
                    plt.plot(tmp, label="A " +
                             sensor_names[sen], c=colors[sen], alpha=alpha_A)
                else:
                    plt.plot(tmp, c=colors[sen], alpha=alpha_A)
                plt.legend()
        plt.show()

    def model_matrix_animate(self):
        global A, C, b, h, i
        import matplotlib.animation as animation
        fig = plt.figure(self.filename)

        A = self.variableDict["A"]
        C = self.variableDict["C"]
        b = self.variableDict["b"]
        h = self.variableDict["h"]
        print C.shape
        i = 0
        plt.subplot(121)
        im = plt.imshow(A[0, :, :], interpolation='none',
                        animated=True, vmin=np.min(A), vmax=np.max(A))
        plt.title('A')

        plt.subplot(122)
        im2 = plt.imshow(C[0, :, :].T, interpolation='none',
                         animated=True, vmin=np.min(C), vmax=np.max(C))
        plt.title('C')

        print A[self.timesteps - 1, :, :]
        print b[self.timesteps - 1, :]

        def updatefig(*args):
            global A, C, b, h, i

            if(i < self.timesteps - 1):
                Ai = A[i, :, :]
                Ci = C[i, :, :]
                bi = b[i, :]
                hi = h[i, :]
                modelData = np.hstack((Ai, np.zeros_like(bi), bi))
                controlerData = np.hstack((Ci, np.zeros_like(hi), hi))

                im.set_array(modelData)
                im2.set_array(controlerData.T)
                print "i = %d" % (i)

                if i % 100 == 0:
                    print np.round(Ai / np.max(np.abs(Ai)), 1)
                    print np.round(Ci / np.max(np.abs(Ci)), 1)

                i += 1
            return im, im2,

        # interval 42 results in about 50s for 1000 timesteps on my computer -> nearly realtime
        ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
        plt.show()


if __name__ == "__main__":
    function_dict = {
        'ts': Analyzer.time_series,
        'split_sensors': Analyzer.split_sensors_effects_smooth,
        'cor': Analyzer.correlation_func,
        'scat': Analyzer.scattermatrix_func,
        'anim_scat': Analyzer.animate_scatter,
        'spect': Analyzer.spectogram_func,
        'lin': Analyzer.learn_motor_sensor_linear,
        'lin_cross': Analyzer.learn_motor_sensor_linear_cross,
        'mlp_cross': Analyzer.learn_motor_sensor_mlp_cross,
        'gmm': Analyzer.learn_motor_sensor_gmm,
        'model_matrix_plot': Analyzer.model_matrix_plot,
        'model_matrix_animate': Analyzer.model_matrix_animate,
        'model_matrix_plot_smooth': Analyzer.model_matrix_plot_smooth,
        'find_emb': Analyzer.find_best_embsize,
        'activity': Analyzer.activity_plot2,
        'details': Analyzer.details,
        'step_sweep': Analyzer.step_sweep
    }

    parser = argparse.ArgumentParser(
        description="lpzrobots ROS controller: test homeostatic/kinetic learning")
    parser.add_argument("-m", "--mode", type=str,
                        help=str(function_dict.keys()))
    parser.add_argument("-f", "--filename", type=str,
                        help="filename (no default)", nargs='?')
    parser.add_argument("-r", "--randomFile", action="store_true")
    parser.add_argument("-ham", "--hamming", action="store_true")
    parser.add_argument("-w", "--windowsize", type=int,
                        help="correlation window size", default=100)
    parser.add_argument("-es", "--embsize", type=int,
                        help="history time steps for learning", default=15)
    args = parser.parse_args()

    if(args.filename == None and args.randomFile == False):
        print("Please select file with\n-f ../foo.pickle or -r for random file\n")

    if(args.mode in function_dict.keys()):
        print "Mode %s selected...\n" % (args.mode)

        analyzer = Analyzer(args)
        function_dict[args.mode](analyzer)
    else:
        if args.mode == None:
            args.mode = ""

        print("Mode '" + args.mode +
              "' not found,\nplease select a mode with -m " + str(function_dict.keys()))
