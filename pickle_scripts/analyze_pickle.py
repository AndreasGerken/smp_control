import cPickle as pickle
import warnings
import time
import argparse
import sys
import os
import numpy as np
import matplotlib as mpl
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
            self.variable_dict = pickle.load(open(self.filename, "rb"))
        except Exception:
            raise Exception("File not found, use -f and the filepath")

        self._extract_all_variables()
        self._prepare_sensor_names()


    """ HELPER FUNCTIONS """

    def _extract_all_variables(self):
        self.numtimesteps = self.variable_dict["numtimesteps"]

        self.cut = args.cut
        if self.cut <= 0 or self.cut > self.numtimesteps:
            self.cut = self.numtimesteps

        self.loop_time = None
        if "loop_time" in self.variable_dict:
            self.loop_time = self.variable_dict['loop_time']

        self.lag = self.variable_dict['lag']

        self.motor_commands = self.variable_dict["y"][:self.cut]
        self.sensor_values = self.variable_dict["x"][:self.cut]

        self.x_pred = None
        if "x_pred" in self.variable_dict:
            self.sensor_prediction = self.variable_dict["x_pred"][:self.cut]
        if "x_pred_coefficients" in self.variable_dict:
            self.x_pred_coefficients = self.variable_dict["x_pred_coefficients"][:self.cut]

        self.sensor_variance_each = np.var(self.sensor_values, axis=0)

        # how to normalize? OSWALD
        self.sensor_variance_each[:3] /= np.var(self.sensor_values[:,:3])
        self.sensor_variance_each[3:] /= np.var(self.sensor_values[:,3:])

        self.timesteps = self.motor_commands.shape[0]
        self.nummot = self.motor_commands.shape[1]
        self.numsen = self.sensor_values.shape[1]

        self.windowsize = self.args.windowsize
        self.embsize = self.args.embsize
        self.hamming = self.args.hamming
        self.extended = self.args.extended

        self.use_sensors = self.variable_dict["robot.use_sensors"]
        self.sensor_dimensions = self.variable_dict["robot.sensor_dimensions"]

        self.epsA = self.variable_dict["epsA"]
        self.epsC = self.variable_dict["epsC"]
        self.creativity = self.variable_dict["creativity"]
        self.xsi = self.variable_dict["xsi"]
        self.ee = self.variable_dict["EE"]

    def _save_image(self, fig, name):
        fig.savefig(os.path.dirname(__file__) + '/' + name)

    def _prepare_sensor_names(self):
        xyz = ["x", "y","z"]
        xyzw = ["x", "y", "z","w"]
        self.sensor_name_extensions = {"acc" : xyz, "gyr" : xyz, "orient" : xyzw, "euler:" : xyz}
        self.sensor_name_long = {"acc": "Accelerometer", "gyr": "Gyroscope", "orient": "Orientation", "rot": "Rotation"}
        self.sensor_names_with_dimensions = self._get_sensor_names_with_dimensions()

    def _get_sensor_names_with_dimensions(self):
        """ This function reads the used sensors from the variable dict and
        gives back an array of all names of the individual sensor dimensions.
        Its size should be matching the second dimension of x """

        # create list of sensor names
        sensor_names = []
        for sensor in self.use_sensors:
            # repeat the sensor name with an identifier as often as the sensor has dimensions
            for sensor_dimension in range(self.sensor_dimensions[sensor]):
                if sensor in self.sensor_name_extensions:
                    name = sensor + " " + self.sensor_name_extensions[sensor][sensor_dimension]
                else:
                    name = sensor + " " + str(sensor_dimension)
                sensor_names.extend([name])
            #sensor_names.extend([sensor + " "+ str(j)
            #                     for j in range(self.sensor_dimensions[sensor])])

        return sensor_names

    def _all_files(self, directory):
        for path, dirs, files in os.walk(directory):
            for f in files:
                yield os.path.join(path, f)

    def _get_triu_of_matrix(self, matrix):
        if matrix.shape[0] != matrix.shape[1]:
            return None

        dim = matrix.shape[0]
        triu = np.triu_indices(dim, k=1)
        return matrix[triu]

    def _get_cumsum_sensor_dimensions(self):
        """ This functions returns an array of the cumulative sum of the dimensions of each sensor.
        With acc 3dim and gyro 3 dim it returns [3, 6]."""

        use_sensor_dimensions = [self.sensor_dimensions[sensor] for sensor in self.use_sensors]
        return np.cumsum(use_sensor_dimensions)

    def _prepare_data_for_learning(self):
        testDataLimit = 4 * self.timesteps / 5

        motoremb = np.array([self.motor_commands[i:i + self.embsize].flatten()
                             for i in range(0, testDataLimit - self.embsize)])
        motorembtest = np.array([self.motor_commands[i:i + self.embsize].flatten()
                                 for i in range(testDataLimit, self.timesteps - self.embsize)])

        self.trainingData = {
            "motor": motoremb, "sensor": self.sensor_values[self.embsize:testDataLimit]}
        self.testData = {"motor": motorembtest,
                         "sensor": self.sensor_values[testDataLimit + self.embsize:]}

    """ ANALYZING FUNCTIONS """

    def details(self):
        print "--- Episode Details ---\n"
        print "timesteps:\t", self.timesteps
        print "looptime:\t", self.loop_time

        print "--- Robot ---\n"
        print "nummot\t:", self.nummot
        print "numsen\t", self.numsen
        print "sensors:\t%s" % ([name + ":[" + str(self.sensor_dimensions[name]) + "]" for name in self.use_sensors])

        print "--- Learning variables ---\n"
        print "epsA\t", self.epsA
        print "epsC\t", self.epsC
        print "Creativity\t", self.creativity


    def time_series_motors_sensors(self):
        """ This function can be used to show the time series of data """
        # THIS IS USED AND TESTED FOR EXPERIMENT 1

        print("The variance of the sensors = %s" % (str(self.sensor_variance_each)))

        f, axarr = plt.subplots(len(self.use_sensors) + 1, 1, figsize=(20,12))

        for motor in range(self.nummot):
            axarr[0].plot(self.motor_commands[:, motor])
            axarr[0].set_ylim([-1.1, 1.1])
        axarr[0].set_title("Motor Commands")

        sensor_index = 0
        for sensor in range(len(self.use_sensors)):
            sensor_name = self.use_sensors[sensor]
            for dim in range(self.sensor_dimensions[self.use_sensors[sensor]]):
                if sensor_name in self.sensor_name_extensions:
                    _label = self.sensor_name_extensions[sensor_name][dim]
                else:
                    _label = str(dim)
                axarr[sensor + 1].plot(self.sensor_values[:, sensor_index], label= _label)

                if sensor_name in self.sensor_name_long:
                    title = self.sensor_name_long[sensor_name]
                else:
                    title = sensor_name


                axarr[sensor + 1].set_title(title)
                sensor_index += 1
            axarr[sensor +1 ].legend()
        f.tight_layout()
        self._save_image(f, 'img/time_series_motor_sensors.png')
        plt.show()

    def time_series_sensor_prediction(self):
        """ This function can be used to show the prediction of the sensor data through the model """
        # THIS IS USED AND TESTED FOR EXPERIMENT 3

        parser = argparse.ArgumentParser()
        parser.add_argument("-dimensions", "--show_dimensions", type=int, default = 0)
        args, unknown_args = parser.parse_known_args()

        #print self.sensor_prediction.shape

        if args.show_dimensions == 0 or args.show_dimensions > self.numsen:
            show_sensors = self.numsen
        else:
            show_sensors = args.show_dimensions

        f, axarr = plt.subplots(show_sensors + 1, figsize=(20,10))

        for motor in range(self.nummot):
            axarr[0].plot(self.motor_commands[self.lag:, motor], label="motor " + str(motor + 1))
            axarr[0].set_ylim([-1.1, 1.1])
            axarr[0].legend()
        axarr[0].set_title("Motor Commands")

        error = self.sensor_values[:-self.lag] - self.sensor_prediction[:-self.lag,:,0]

        i = 0

        #print self.x_pred_coefficients[:-self.lag, :, 0]
        #print self.sensor_prediction[:-self.lag,0]
        for sensor in range(len(self.use_sensors)):
            sensor_name = self.use_sensors[sensor]
            for dim in range(self.sensor_dimensions[self.use_sensors[sensor]]):
                pred = self.sensor_prediction[:-self.lag,i,0]

                # normal plot
                axarr[i + 1].plot(self.sensor_values[:-self.lag,i], label="sensor measurement")
                axarr[i + 1].plot(pred, label="prediction")

                # cut between bias and motor pred
                if hasattr(self, 'x_pred_coefficients'):
                    motor_pred = np.sum(self.x_pred_coefficients[:-self.lag, :self.nummot, i], axis = 1)
                    bias_pred = self.x_pred_coefficients[:-self.lag, self.nummot, i]

                    pred[np.where(pred == 0)] = 1
                    motor_pred[np.where(motor_pred == 0)] = 1
                    bias_pred[np.where(bias_pred == 0)] = 1
                    print pred
                    print motor_pred
                    print bias_pred

                    axarr[i + 1].plot(motor_pred, label="motor_prediction")
                    axarr[i + 1].plot(bias_pred , label="bias")



                # title
                title = self.sensor_names_with_dimensions[i]

                axarr[i + 1].set_title(title)
                i+=1

                if i >= show_sensors:
                    break
            if i >= show_sensors:
                break
        plt.legend()
        f.tight_layout()
        self._save_image(f, '/img/time_series_pred.png')

        #plt.figure()


        #plt.plot(error)
        plt.show()

    def learn_motor_sensor_linear_new(self):
        """
        This mode learns the sensor responses from the motor commands with
        linear regression and tests the result.
        """
        # THIS IS USED AND TESTED FOR EXPERIMENT 2

        parser = argparse.ArgumentParser()
        parser.add_argument("-extended", "--extended", type=bool, default = False)
        parser.add_argument("-lag", "--lag", type=int, default = 1)
        args, unknown_args = parser.parse_known_args()

        self.sensor_values -= np.mean(self.sensor_values, axis=0)
        self._prepare_data_for_learning()

        if args.extended:
            self.trainingData["motor"] = np.zeros_like(self.trainingData["motor"])
            self.testData["motor"] = np.zeros_like(self.testData["motor"])

            self.trainingData["motor"] = np.hstack((self.trainingData["motor"][:-args.lag], self.trainingData["sensor"][args.lag:]))
            self.testData["motor"] = np.hstack((self.testData["motor"][:-args.lag], self.testData["sensor"][args.lag:]))

            self.trainingData["sensor"] = self.trainingData["sensor"][:-args.lag]
            self.testData["sensor"] = self.testData["sensor"][:-args.lag]


        regr = linear_model.Ridge(alpha=5.)
        regr.fit(self.trainingData["motor"], self.trainingData["sensor"])
        predTest = regr.predict(self.testData["motor"])

        mse = np.mean((predTest - self.testData["sensor"]) ** 2)
        print mse

        f, ax = plt.subplots(2,3, sharey=True, sharex=True, figsize=(20,12))
        plt.rc('font', family='serif', size=30)


        for i in range(self.numsen):
            ax[i/3, i%3].plot(predTest[:,i])
            ax[i/3, i%3].plot(self.testData["sensor"][:,i])
            plt.xticks(np.arange(0,400,100))
        f.tight_layout()
        self._save_image(f, 'img/sensor_motor_linear.png')

        plt.show()


    def step_sweep(self):
        if not 'robot.step_length' in self.variable_dict:
            warnings.warn("this pickle file was not recorded in the step sweep mode")

        reset_length = self.variable_dict['robot.reset_length']
        step_length = self.variable_dict['robot.step_length']
        repeat_step = self.variable_dict['robot.repeat_step']
        step_size = self.variable_dict['robot.step_size']

        cut_response = 50

        cycle_length = step_length + reset_length
        cycle_max = self.numtimesteps / cycle_length
        sweep_angle_total = 2. - step_size
        angle_per_step = sweep_angle_total / cycle_max
        steps_max = cycle_max / repeat_step

        x = self.variable_dict["x"]
        y = self.variable_dict["y"]
        x -= np.mean(x, axis=0)
        #x /= np.std(x, axis=all)
        #x = np.abs(x)

        responses = np.zeros((cycle_max, cut_response, x.shape[1]))
        avg_responses = np.zeros((steps_max, cut_response, x.shape[1]))
        std_responses = np.zeros((steps_max, cut_response, x.shape[1]))

        colors = cm.Greys(np.linspace(0, 1, cycle_max))
        colors = cm.viridis(np.linspace(0, 1, cycle_max))
        colors = cm.cool(np.linspace(0, 1, steps_max))

        #mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])

        # Using contourf to provide my colorbar info, then clearing the figure
        Z = [[0,0],[0,0]]
        levels = range(-180,int(sweep_angle_total * 90. + steps_max),int(steps_max))
        CS3 = plt.contourf(Z, levels, cmap=cm.cool)
        plt.clf()

        for i in range(cycle_max):
            responses[i] = x[i * cycle_length + reset_length: i * cycle_length + reset_length + cut_response]
            responses[i] -= np.average(responses[i,:5]) # bring the signal of the first timesteps to the center

        for i in range(steps_max):
            avg_responses[i] = np.average(responses[i * repeat_step: (i+1) * repeat_step], axis = 0)
            std_responses[i] = np.std(responses[i * repeat_step:(i+1)*repeat_step], axis = 0)
            avg_responses[i] -= np.average(avg_responses[i,:5], axis=0) # bring the signal of the first timesteps to the center


        # plotting
        f, axarr = plt.subplots(3, 2)

        line_alpha = 0.5
        line_offset = 3.

        line_offset = 0.
        for i in range(steps_max):
            j = steps_max - i - 1

            if j == 1:
                text = "-180"
            elif j == steps_max - 1:
                text = "180"
            else:
                text = None

            for dim in range(6):
                x = range(cut_response)
                y_avg = avg_responses[j,:,dim]
                y_std = std_responses[j,:,dim]
                y_offset = y_avg + j * line_offset
                y_low = y_offset - y_std
                y_high = y_offset + y_std
                ax = axarr[dim / 2, dim % 2]

                #ax.plot(x, y_low, c=colors[i], lw=1, alpha=line_alpha, label=text)
                #ax.plot(x, y_high, c=colors[i], lw=1, alpha=line_alpha, label=text)

                im = ax.fill_between(x, y_low, y_high, facecolor=colors[j], lw=1, alpha=line_alpha, label=text)
                ax.set_ylabel(self.sensor_names_with_dimension[dim])
                ax.set_xlabel("ms")
                #plt.xticks(np.arange(0, cut_response, 1.0))

                #axarr[1].plot(np.average(np.abs(responses[i,:,3:]), axis = 1)  + (i / repeat_step) * line_offset, c=colors[i], lw=1.5, alpha=line_alpha, label=text)
                #plt.xticks(np.arange(0, cut_response, 1.0))

                axarr[dim/2, dim%2].legend()

        f.colorbar(CS3)

        # Single plot

        f, ax = plt.subplots(1,1, figsize=(20,15))
        plt.rc('font', family='serif', size=30)
        line_alpha = 0.
        fill_alpha = 0.4
        for i in range(steps_max-2):
            j = steps_max - i - 3

            if j == 1:
                text = "180"
            elif j == steps_max - 1:
                text = "-180"
            else:
                text = None

            dim = 1
            x = range(cut_response)
            y_avg = avg_responses[j,:,dim]
            y_std = std_responses[j,:,dim]
            y_offset = y_avg + j * line_offset
            y_low = y_offset - y_std
            y_high = y_offset + y_std

            ax.plot(x, y_low, c=colors[j], lw=1, alpha=line_alpha)
            ax.plot(x, y_high, c=colors[j], lw=1, alpha=line_alpha)

            ax.fill_between(x, y_low, y_high, facecolor=colors[j], lw=1, alpha=fill_alpha, label=text)

            ax.set_ylabel("$m/s^2$", fontsize=40)
            ax.set_xlabel("$ms$", fontsize=40)
        ax.set_title("acceleration - y", fontsize=40)

        cbar = plt.colorbar(CS3)
        cbar.set_label("angle before step")
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        f.tight_layout()
        self._save_image(f, 'img/stepResponseAccelY.png')
        plt.show()


    def time_series(self):
        """ This function can be used to show the time series of data """
        cut = self.variable_dict["numtimesteps"]
        x = self.sensor_values
        y = self.motor_commands
        xsi = self.variable_dict["xsi"][:cut]
        ee = self.variable_dict["EE"][:cut]

        f, axarr = plt.subplots(4, 1)
        plt.rc('font', family='serif', size=30)

        for sen in range(x.shape[1]):
            axarr[0].plot(x[:, sen], label=self.sensor_names_with_dimensions[sen])
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



    def correlation_func(self):
        loopsteps = self.timesteps - self.windowsize

        #self.motor_commands += np.random.normal(size=self.motor_commands.shape) * 0.1

        # testdata
        # self.motor_commands = np.arange(20).reshape(10,2)
        # print self.motor_commands

        # print(len(self.motor_commands))

        allCorrelations = np.zeros((loopsteps, self.nummot, self.nummot))
        allAverageSquaredCorrelations = np.zeros((loopsteps, 6))

        correlationsGlobal = np.cov(self.motor_commands.T)
        print correlationsGlobal

        for i in range(len(self.motor_commands) - self.windowsize):
            # use only the latest step of the buffer but all motor commands
            # shape = (timestep, motorChannel, buffer)

            window = self.motor_commands[i:i + self.windowsize, :]
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
            allAverageSquaredCorrelations[i, :] = self._get_triu_of_matrix(
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
        self._save_image(fig,'img/correlations.png')
        plt.show()
        # print allAverageSquaredCorrelations

    def learn_motor_sensor_gmm(self):
        """
        This mode learns the sensor responses from the motor commands with
        gaussian mixture models and tests the result
        """

        for emb in range(1, 100):
            self.embsize = emb

            x = []
            score = []
            avg_x = []
            avg_sc = []

            self._prepare_data_for_learning()
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

            mot_lag = self.motor_commands[0:-1 * lag]
            sen_lag = self.sensor_values[lag:]
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

            mot_lag = self.motor_commands[0:-1 * lag]
            sen_lag = self.sensor_values[lag:]
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


    def find_best_embsize(self):
        max_embsize = 50
        mse = np.zeros(max_embsize - 1 )
        regr = linear_model.Ridge(alpha=10.0)
        for emb in range(1, max_embsize):
            self.embsize = emb
            self._prepare_data_for_learning()
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
                (self.motor_commands[partS:partE, :], self.sensor_values[partS:partE, :]))

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

            #print("scatter %d saved" %(part))
            plt.show()

    def spectogram_func(self):
        loopsteps = self.timesteps - self.windowsize

        sensors = self.sensor_values[...]
        self.numsen = sensors.shape[1]
        s = sensors

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

    def activity_plot(self):
        # TODO: erase or modify

        if self.windowsize % 2 != 0:
            print("please choose a even windowsize")
            return

        distance_array = np.zeros((self.timesteps, 6))
        windowfunction = np.hamming(self.windowsize)

        self.sensor_values -= np.mean(self.sensor_values, axis=0)
        for i in range(0, self.timesteps - self.windowsize):
            start = i
            end = i + self.windowsize

            min_x = np.min(self.sensor_values[start:end, :], axis=0)
            max_x = np.max(self.sensor_values[start:end, :], axis=0)
            distance_array[i, :] = max_x - min_x
        # plt.plot(self.sensor_values[:,1])
        # plt.plot(self.sensor_values[:,1])
        #plt.plot(distance_array, self.hz[self.windowsize/2: self.timesteps - self.windowsize/2])
        dist_avg = [np.average(distance_array[i:i + 60, :], axis=0)
                    for i in range(self.timesteps - 60)]
        lines = plt.plot(np.linspace(
            0, 6.34, self.timesteps - 70), dist_avg[10:])
        plt.legend(lines, ("ax", "ay", "az", "gx", "gy", "gz"))
        plt.show()

    def activity_plot2(self):
        windowfunction = np.hamming(self.windowsize)
        sv_abs = np.sum(np.abs(self.sensor_values), axis=1)
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

        motorGlobal = self.motor_commands[:, 0]
        sensorGlobal = self.sensor_values

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

    def model_matrix_plot_smooth(self):
        C = self.variable_dict["C"]
        A = self.variable_dict["A"]
        sensors = self.numsen

        fig = plt.figure()
        ax = None
        for j in range(sensors):
            abs_max = np.std(self.variable_dict["x"][:, j + sensors * 2])
            #abs_max = np.max(np.abs(self.variable_dict["x"][:,j+sensors*2]))
            ax = fig.add_subplot(1, sensors, j + 1, sharey=ax)
            plt.plot(C[:, 0, j::sensors] * abs_max)
            # plt.legend()
        # plt.subplot(122)
        # for i in range(A.shape[1]):
        #    for j in range(A.shape[2]):
        #        plt.plot(A[:,i,j], label="A "+ str(i) + " " + str(j))
        # plt.legend()
        plt.show()

    def matrices_coefficients(self):
        # USED FOR EXPERIMENT 5

        onePlot = True
        oneColorPerSensor = False
        labelForEachCoefficient = True

        C = self.variable_dict["C"]
        A = self.variable_dict["A"]
        x = self.variable_dict["x"]
        y = self.variable_dict["y"]
        embedding = self.variable_dict["embedding"]
        x_std = np.std(x, axis=0)
        y_std = np.std(y, axis=0)

        colors = cm.jet(np.linspace(0, 1, len(self.sensor_names_with_dimensions) * embedding))
        #colors = ['r','r','r','b','b','b','k','k','k','k']


        cut_s = 0
        cut_e = C.shape[0]
        smoothing_window_C = 1
        alpha_C = 1

        if onePlot:
            plt.subplot(121)
        for mot in range(C.shape[1]):
            for sen in range(C.shape[2]):
                print mot, sen
                if not onePlot:
                    plt.subplot(C.shape[1], 2, mot * 2 + 1)

                tmp = [np.average(C[i:i + smoothing_window_C, mot, sen])
                       for i in range(C.shape[0] - smoothing_window_C)]
                tmp = np.array(tmp[cut_s:cut_e]) / y_std[mot]
                if labelForEachCoefficient:
                    text = "C [sensor %d, motor %d]" % (sen, mot)

                elif mot == 0 and embedding == 1:
                    text = "C" + self.sensor_names_with_dimensions[sen]
                else:
                    text = None

                if oneColorPerSensor:
                    _color = colors[sen]
                else:
                    _color = None

                plt.plot(tmp, label=text, c=_color, alpha=alpha_C)

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
                if mot == 0 and embedding == 1:
                    text = "A " + self.sensor_names_with_dimensions[sen]
                else:
                    text = None
                plt.plot(tmp, label=text, c=colors[sen], alpha=alpha_A)
                plt.legend()
        plt.show()

    def model_matrix_animate(self):
        global A, C, b, h, i
        import matplotlib.animation as animation
        fig = plt.figure(self.filename)

        A = self.variable_dict["A"]
        C = self.variable_dict["C"]
        b = self.variable_dict["b"]
        h = self.variable_dict["h"]
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
        'ts_ms': Analyzer.time_series_motors_sensors,
        'ts_pred': Analyzer.time_series_sensor_prediction,
        'split_sensors': Analyzer.split_sensors_effects_smooth,
        'cor': Analyzer.correlation_func,
        'scat': Analyzer.scattermatrix_func,
        'anim_scat': Analyzer.animate_scatter,
        'spect': Analyzer.spectogram_func,
        'lin_cross': Analyzer.learn_motor_sensor_linear_cross,
        'lin_new': Analyzer.learn_motor_sensor_linear_new,
        'mlp_cross': Analyzer.learn_motor_sensor_mlp_cross,
        'gmm': Analyzer.learn_motor_sensor_gmm,
        'matrices_coefficients': Analyzer.matrices_coefficients,
        'model_matrix_animate': Analyzer.model_matrix_animate,
        'model_matrix_plot_smooth': Analyzer.model_matrix_plot_smooth,
        'find_emb': Analyzer.find_best_embsize,
        'activity': Analyzer.activity_plot2,
        'details': Analyzer.details,
        'step_sweep': Analyzer.step_sweep
    }

    parser = argparse.ArgumentParser(
        description="lpzrobots ROS controller: test homeostatic/kinetic learning")

    parser.add_argument("filename")

    parser.add_argument("-m", "--mode", type=str,
                        help=str(function_dict.keys()))

    parser.add_argument("-r", "--randomFile", action="store_true")
    parser.add_argument("-ham", "--hamming", action="store_true")
    parser.add_argument("-w", "--windowsize", type=int,
                        help="correlation window size", default=100)
    parser.add_argument("-es", "--embsize", type=int,
                        help="history time steps for learning", default=1)
    parser.add_argument("-extended", "--extended", type=int,
                        help="toggle extended model", default=1)
    parser.add_argument("-cut", "--cut", type=int, help="Cut of the graph after 'cut' timesteps.", default=0)
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        warnings.warn("Unknown arguments were parsed in smp_control. They will be parsed by the robot configuration again:")
        print "unknown_args: %s" % str(unknown_args)

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
