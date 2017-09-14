def learn_motor_sensor_gmm(self):
        """
        This mode learns the sensor responses from the motor commands with
        gaussian mixture models and tests the result
        """

        # TODO Does it work? got stuck

        for emb in range(1, 100):
            self.embsize = emb

            x = []
            score = []
            avg_x = []
            avg_sc = []

            self._prepare_data_for_learning()
            for k in range(5, 6):
                sum_sc = 0
                for i in range(0, 5):
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
                avg_sc.append(sum_sc / 5.)

            plt.plot(x, score, 'o', label=emb)
            plt.plot(avg_x, avg_sc)
        plt.legend()
        plt.show()

        return 0

    def time_series_smoothing(self):
        """ This is a special function for a tested feature of smoothing. It was used with the pendulum. """
        x = self.variable_dict["x"]
        y = self.variable_dict["y"]
        sensors = 3
        fig = plt.figure()
        for i in range(sensors):
            fig.add_subplot(sensors, 1, i + 1)
            plt.plot(x[:, i::sensors])

        plt.show()

    def split_sensors_effects_smooth(self):
        # TODO: not functional
        # C=self.variable_dict["C"]
        # x=self.variable_dict["x"]
        # h=self.variable_dict["h"]
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




    def spectogram_func(self):
        """ This function shows the spectral composition of motor and sensor signals."""

        # TODO: It only works with cut = 1000, since nperseg and overlap don't work elsewise
        
        loopsteps = self.timesteps - self.windowsize

        #sensors = self.sensor_values[...]
        #s = sensors

        print sensors.shape

        import scipy.signal as sig

        m = self.motor_commands
        s = self.sensor_values

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


    def animate_scatter(self):
        import matplotlib.animation as animation
        global motorGlobal, sensorGlobal, i

        # TODO: Fix for all axis or remove

        i = 0
        fig = plt.figure(self.filename)

        motorGlobal = self.motor_commands
        sensorGlobal = self.sensor_values

        data_m1 = motorGlobal[i:i + 100, 0]
        data_m2 = motorGlobal[i:i + 100, 1]
        data_m3 = motorGlobal[i:i + 100, 2]
        data_m4 = motorGlobal[i:i + 100, 3]
        datay1 = sensorGlobal[i:i + 100, 0]
        datay2 = sensorGlobal[i:i + 100, 1]
        datay3 = sensorGlobal[i:i + 100, 2]



        ax1 = plt.subplot(331)
        scat1 = plt.scatter(data_m1, datay1, animated=True)

        ax2 = plt.subplot(332)
        scat2 = plt.scatter(data_m1, datay2, animated=True)

        ax3 = plt.subplot(333)
        scat3 = plt.scatter(data_m1, datay3, animated=True)

        ax4 = plt.subplot(334)
        scat4 = plt.scatter(data_m2, datay1, animated=True)

        ax5 = plt.subplot(335)
        scat5 = plt.scatter(data_m2, datay2, animated=True)

        ax6 = plt.subplot(336)
        scat6 = plt.scatter(data_m2, datay3, animated=True)

        ax7 = plt.subplot(337)
        scat7 = plt.scatter(data_m3, datay1, animated=True)

        ax8 = plt.subplot(338)
        scat8 = plt.scatter(data_m3, datay2, animated=True)

        ax9 = plt.subplot(339)
        scat9 = plt.scatter(data_m3, datay3, animated=True)


        ax1.set_xlim(np.min(motorGlobal[:,0]), np.max(motorGlobal[:,0]))
        ax2.set_xlim(np.min(motorGlobal[:,0]), np.max(motorGlobal[:,0]))
        ax3.set_xlim(np.min(motorGlobal[:,0]), np.max(motorGlobal[:,0]))
        ax4.set_xlim(np.min(motorGlobal[:,1]), np.max(motorGlobal[:,1]))
        ax5.set_xlim(np.min(motorGlobal[:,1]), np.max(motorGlobal[:,1]))
        ax6.set_xlim(np.min(motorGlobal[:,1]), np.max(motorGlobal[:,1]))
        ax7.set_xlim(np.min(motorGlobal[:,2]), np.max(motorGlobal[:,2]))
        ax8.set_xlim(np.min(motorGlobal[:,2]), np.max(motorGlobal[:,2]))
        ax9.set_xlim(np.min(motorGlobal[:,2]), np.max(motorGlobal[:,2]))

        ax1.set_ylim(np.min(sensorGlobal[:, 0]), np.max(sensorGlobal[:, 0]))
        ax2.set_ylim(np.min(sensorGlobal[:, 1]), np.max(sensorGlobal[:, 1]))
        ax3.set_ylim(np.min(sensorGlobal[:, 2]), np.max(sensorGlobal[:, 2]))
        ax4.set_ylim(np.min(sensorGlobal[:, 0]), np.max(sensorGlobal[:, 0]))
        ax5.set_ylim(np.min(sensorGlobal[:, 1]), np.max(sensorGlobal[:, 1]))
        ax6.set_ylim(np.min(sensorGlobal[:, 2]), np.max(sensorGlobal[:, 2]))
        ax7.set_ylim(np.min(sensorGlobal[:, 0]), np.max(sensorGlobal[:, 0]))
        ax8.set_ylim(np.min(sensorGlobal[:, 1]), np.max(sensorGlobal[:, 1]))
        ax9.set_ylim(np.min(sensorGlobal[:, 2]), np.max(sensorGlobal[:, 2]))

        def updatefig(*args):
            global motorGlobal, sensorGlobal, i
            if(i < self.timesteps - 101):

                pts = 100
                data1 = np.vstack(
                    (motorGlobal[i:i + pts,0], sensorGlobal[i:i + pts, 0])).T
                data2 = np.vstack(
                    (motorGlobal[i:i + pts,0], sensorGlobal[i:i + pts, 1])).T
                data3 = np.vstack(
                    (motorGlobal[i:i + pts,0], sensorGlobal[i:i + pts, 2])).T

                data4 = np.vstack(
                    (motorGlobal[i:i + pts,1], sensorGlobal[i:i + pts, 0])).T
                data5 = np.vstack(
                    (motorGlobal[i:i + pts,1], sensorGlobal[i:i + pts, 1])).T
                data6 = np.vstack(
                    (motorGlobal[i:i + pts,1], sensorGlobal[i:i + pts, 2])).T

                data7 = np.vstack(
                    (motorGlobal[i:i + pts,2], sensorGlobal[i:i + pts, 0])).T
                data8 = np.vstack(
                    (motorGlobal[i:i + pts,2], sensorGlobal[i:i + pts, 1])).T
                data9 = np.vstack(
                    (motorGlobal[i:i + pts,2], sensorGlobal[i:i + pts, 2])).T

                scat1.set_offsets(data1)
                scat2.set_offsets(data2)
                scat3.set_offsets(data3)
                scat4.set_offsets(data4)
                scat5.set_offsets(data5)
                scat6.set_offsets(data6)
                scat7.set_offsets(data7)
                scat8.set_offsets(data8)
                scat9.set_offsets(data9)

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

