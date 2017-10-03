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
from igmm_cond import IGMM_COND
from sklearn.neural_network import MLPRegressor
from scipy import signal

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

        if 'sensor_prediction' in self.variable_dict:
            self._sliding_window_variance()

        params = {'legend.fontsize': 'large',
          'figure.figsize': (20, 10),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
        plt.rcParams.update(params)


    """ HELPER FUNCTIONS """

    def _extract_all_variables(self):
        self.numtimesteps = self.variable_dict["numtimesteps"]

        self.cut = args.cut
        if self.cut <= 0 or self.cut > self.numtimesteps:
            self.cut = self.numtimesteps

        self.lag = self.variable_dict['lag']
        self.motor_commands = self.variable_dict["y"][:self.cut]
        self.sensor_values = self.variable_dict["x"][:self.cut]

        self.loop_time = None
        if "loop_time" in self.variable_dict:
            self.loop_time = self.variable_dict['loop_time']

        self.x_pred = None
        if "x_pred" in self.variable_dict:
            self.sensor_prediction = self.variable_dict["x_pred"][:self.cut][:,:,0]
            self.sensor_prediction_error = self.sensor_values - self.sensor_prediction

        if "x_pred_coefficients" in self.variable_dict:
            self.x_pred_coefficients = self.variable_dict["x_pred_coefficients"][:self.cut]

        self.use_sensors = self.variable_dict["robot.use_sensors"]
        self.sensor_dimensions = self.variable_dict["robot.sensor_dimensions"]
        self.classname = self.variable_dict["robot.classname"]
        self.sensor_variance_each = np.var(self.sensor_values, axis=0)

        # how to normalize? OSWALD
        #self.sensor_variance_each[:3] /= np.var(self.sensor_values[:,:3])
        #self.sensor_variance_each[3:] /= np.var(self.sensor_values[:,3:])

        self.timesteps = self.motor_commands.shape[0]
        self.nummot = self.motor_commands.shape[1]
        self.numsen = self.sensor_values.shape[1]

        self.windowsize = self.args.windowsize
        self.windowsize_half = self.windowsize / 2
        self.embsize = self.args.embsize
        self.hamming = self.args.hamming
        self.extended = self.args.extended


        self.epsA = self.variable_dict["epsA"]
        self.epsC = self.variable_dict["epsC"]
        self.creativity = self.variable_dict["creativity"]
        self.xsi = self.variable_dict["xsi"]
        self.ee = self.variable_dict["EE"]

    def _save_image(self, fig, name, tight=None):
        if tight:
            fig.savefig(os.path.dirname(__file__) + '/' + name, bbox_inches='tight')
        else:
            fig.savefig(os.path.dirname(__file__) + '/' + name)

    def _prepare_sensor_names(self):
        xyz = ["x", "y","z"]
        xyzw = ["x", "y", "z","w"]
        self.sensor_name_extensions = {"acc" : xyz, "gyr" : xyz, "orient" : xyzw, "euler:" : xyz, "poti_integral":[""]}
        self.sensor_name_long = {"acc": "Accelerometer", "gyr": "Gyroscope", "orient": "Orientation", "rot": "Rotation", "poti_integral": "Inegral of the Potentiometer"}
        self.sensor_units = {"acc":"m/s^2", "gyr":"rad/s"}
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

    def _prepare_data_for_learning(self, normalizeByStd=False):
        testDataLimit = 4 * self.timesteps / 5

        motoremb = np.array([self.motor_commands[i:i + self.embsize].flatten()
                             for i in range(0, testDataLimit - self.embsize)])
        motorembtest = np.array([self.motor_commands[i:i + self.embsize].flatten()
                                 for i in range(testDataLimit, self.timesteps - self.embsize)])

        if normalizeByStd:
            self.sensor_values /= np.std(self.sensor_values, axis = 0)

        self.trainingData = {
            "motor": motoremb, "sensor": self.sensor_values[self.embsize:testDataLimit]}
        self.testData = {"motor": motorembtest,
                         "sensor": self.sensor_values[testDataLimit + self.embsize:]}

    def _pointwise_variance(self, x):
        return (x - np.mean(x, axis = 0)) ** 2

    def _pointwise_hamming_variance(self, data):
        # calculate the pointwise variance
        x = self._pointwise_variance(data)

        # create hamming window and normalize it
        windowfunction = np.hamming(self.windowsize)
        #windowfunction /= np.sum(windowfunction)

        result = np.zeros((self.numtimesteps-self.windowsize, self.numsen))

        # apply the window function and calculate the mean
        for i in range(self.numtimesteps - self.windowsize):
            window = x[i: i + self.windowsize]
            window_ham = (window.T * windowfunction).T

            result[i,:] = np.mean(window_ham, axis = 0).flatten()

        return result


    def _sliding_window_variance(self):
        self.sensor_value_variance_ham = self._pointwise_hamming_variance(self.sensor_values)
        self.sensor_prediction_variance_ham = self._pointwise_hamming_variance(self.sensor_prediction)
        self.sensor_prediction_error_variance_ham = self._pointwise_hamming_variance(self.sensor_prediction_error - np.mean(self.sensor_prediction_error, axis = 0))

    """ ANALYZING FUNCTIONS """

    def details(self):
        print "--- Episode Details ---\n"
        print "timesteps:\t", self.timesteps
        print "looptime:\t", self.loop_time

        print "--- Robot ---\n"
        print "class\t:", self.classname
        print "nummot\t:", self.nummot
        print "numsen\t", self.numsen
        print "sensors:\t%s" % ([name + ":[" + str(self.sensor_dimensions[name]) + "]" for name in self.use_sensors])

        print "--- Learning variables ---\n"
        print "epsA\t", self.epsA
        print "epsC\t", self.epsC
        print "Creativity\t", self.creativity


    def time_series(self):
        """ This function can be used to show the time series of data """

        # TODO REPAIR?

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

        axarr[3].plot(ee)

        f.subplots_adjust(hspace=0.3)
        plt.legend()

        plt.show()

    def time_series_motors_sensors(self):
        """ This function can be used to show the time series of data """
        # THIS IS USED AND TESTED FOR EXPERIMENT 1

        parser = argparse.ArgumentParser()
        parser.add_argument("-sync", "--synchronous_front_back", help="Argument to have front and hind legs synchronous", default = False, action='store_true')
        args, unknown_args = parser.parse_known_args()

        print("The variance of the sensors = %s" % (str(self.sensor_variance_each)))

        f, axarr = plt.subplots(len(self.use_sensors) + 1, 1, figsize=(20,12), sharex=True)

        for motor in range(self.nummot):
            if args.synchronous_front_back and motor % 2 != 0:
                axarr[0].plot(self.motor_commands[:, motor])
        axarr[0].set_ylim([-1.1, 1.1])
        axarr[0].set_title("Motor Commands")

        if args.synchronous_front_back:
            axarr[0].legend(["front", "hind"])

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
                axarr[sensor + 1].set_ylabel("$" + self.sensor_units[sensor_name] + "$")
                sensor_index += 1

            axarr[sensor +1 ].legend()

        axarr[sensor + 1].set_xlabel("timesteps ($10ms$)")

        f.tight_layout()
        self._save_image(f, 'img/time_series_motor_sensors.png')
        plt.show()

    def hist(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--source", type=str, help="[m = motor, s= sensor]", default = "m")
        args, unknown_args = parser.parse_known_args()

        if args.source == "m":
            source = self.motor_commands
        else:
            source = self.sensor_values

        f = plt.figure()
        plt.hist(source, label=["M1", "M2", "M3", "M4"], bins= 10)
        plt.ylabel=""
        plt.legend()
        self._save_image(f, 'img/hist.png')
        plt.show()

    def time_series_sensor_prediction(self):
        """ This function can be used to show the prediction of the sensor data through the model """
        # THIS IS USED AND TESTED FOR EXPERIMENT 3

        parser = argparse.ArgumentParser()
        parser.add_argument("-dimensions", "--show_dimensions", type=int, default = 0)
        parser.add_argument("-filter", "--filter_cutoff", type=float, default = 0.5)
        parser.add_argument("-xs", '--xlim_start', type= int, default = 0)
        parser.add_argument('-xe', '--xlim_end', type=int, default = self.numtimesteps)
        parser.add_argument('-pMotor', '--plotMotor', type=bool, default = False)
        parser.add_argument('-pSensors', '--plotSensors', type=bool, default = True)
        parser.add_argument('-pSensorsCut', '--plotSensorsCut', type=bool, default = False)
        parser.add_argument('-pMse', '--plotMse', type=bool, default = False)
        parser.add_argument('-pVar', '--plotVar', type=bool, default = False)
        parser.add_argument('-pMseNorm', '--plotMseNorm', type=bool, default = False)
        # TODO Make the bool args setable without writing the 1

        args, unknown_args = parser.parse_known_args()

        cut_begin = 1000
        cut_end = 2000

        b, a = signal.butter(8, args.filter_cutoff)
        self.sensor_values = signal.filtfilt(b, a, self.sensor_values, padlen=150, axis =0 )
        self.sensor_prediction = signal.filtfilt(b, a, self.sensor_prediction, padlen=150, axis = 0)

        # calculate the number of subplots
        num_subplots = 0
        if args.plotMotor:
            num_subplots += 1

        if args.plotSensors:
            if args.show_dimensions == 0 or args.show_dimensions > self.numsen:
                show_sensors = self.numsen
            else:
                show_sensors = args.show_dimensions

            num_subplots += show_sensors

        if args.plotMse:
            num_subplots += 1

        if args.plotVar:
            num_subplots += 1

        if args.plotMseNorm:
            num_subplots += 1

        params = {'legend.fontsize': 'large',
          'figure.figsize': (20, 10),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
        plt.rcParams.update(params)
        f, axarr = plt.subplots(num_subplots, sharex=True)

        cnt_subplot = 0

        # Motors
        if args.plotMotor:
            for motor in range(self.nummot):
                axarr[cnt_subplot].plot(self.motor_commands[self.lag:, motor], label="motor " + str(motor + 1))
                axarr[cnt_subplot].set_ylim([-1.1, 1.1])
                axarr[cnt_subplot].legend()
            axarr[cnt_subplot].set_title("Motor Commands")

            # increment the subplot counter
            cnt_subplot+=1

        # Sensors with their prediction
        if args.plotSensors:
            error = self.sensor_values[:-self.lag] - self.sensor_prediction[:-self.lag]

            i = 0

            for sensor in range(len(self.use_sensors)):
                sensor_name = self.use_sensors[sensor]
                for dim in range(7):
                    pred = self.sensor_prediction[:-self.lag,i]
                    selected_ax = axarr[cnt_subplot + i]
                    # normal plot
                    selected_ax.plot(self.sensor_values[:-self.lag,i], label="measurement")
                    selected_ax.plot(pred, label="prediction")

                    # cut between bias and motor pred
                    if hasattr(self, 'x_pred_coefficients') and args.plotSensorsCut:
                        motor_pred = np.sum(self.x_pred_coefficients[:-self.lag, :self.nummot, i], axis = 1)
                        bias_pred = self.x_pred_coefficients[:-self.lag, self.nummot, i]

                        pred[np.where(pred == 0)] = 1
                        motor_pred[np.where(motor_pred == 0)] = 1
                        bias_pred[np.where(bias_pred == 0)] = 1

                        selected_ax.plot(motor_pred, label="motor_prediction")
                        selected_ax.plot(bias_pred , label="bias")

                    # title
                    #print self.sensor_name_long[sensor_name]
                    #print self.sensor_name_extensions[sensor_name]
                    #title = self.sensor_name_long[sensor_name] + " " + self.sensor_name_extensions[sensor_name][dim]

                    #if i == 0:
                    #    selected_ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          #ncol=2, fancybox=True, shadow=True)
                    #selected_ax.set_title(title)
                    #selected_ax.set_ylim([-2, 2.5])
                    #selected_ax.set_yticks(np.arange(-2, 2.5, 1))
                    #selected_ax.set_ylabel("$[rad/sec]$")
                    #selected_ax.grid(linestyle='--', linewidth=1, alpha = 0.2)
                    i+=1

                    if i >= show_sensors:
                        break
                if i >= show_sensors:
                    break

            # increment the subplot counter
            cnt_subplot += show_sensors

        x = np.arange(0, self.numtimesteps - self.windowsize, 1) + (self.windowsize/2)

        if args.plotMse:
            #axarr[cnt_subplot].plot(x, mse)
            axarr[cnt_subplot].plot(x, self.sensor_prediction_error_variance_ham)
            axarr[cnt_subplot].plot(x, np.mean(self.sensor_prediction_error_variance_ham, axis = 1), color='k', linewidth = 2, linestyle="--")
            axarr[cnt_subplot].grid(linestyle='--', linewidth=1, alpha = 0.2)
            axarr[cnt_subplot].set_title("Squared error through sliding window")
            axarr[cnt_subplot].legend(["x", "y", "z", "average"], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True)
            axarr[cnt_subplot].set_ylim([0,0.19])
            axarr[cnt_subplot].set_yticks(np.arange(0,0.19, 0.04))
            cnt_subplot += 1

        if args.plotVar:
            #axarr[cnt_subplot].plot(x, stddev)
            axarr[cnt_subplot].plot(x, self.sensor_value_variance_ham)
            axarr[cnt_subplot].set_title("Variance")

            selected_ax.legend()
            cnt_subplot += 1

        if args.plotMseNorm:
            #axarr[cnt_subplot].plot(x, mse/stddev)
            pe_norm = self.sensor_prediction_error_variance_ham / self.sensor_value_variance_ham
            #for i in range(self.numsen):
                #pe_norm[:,i] /= np.mean(self.sensor_value_variance_ham, axis = 1)

            axarr[cnt_subplot].plot(x, pe_norm)
            axarr[cnt_subplot].plot(x, np.mean(pe_norm, axis = 1), color='k', linewidth = 2, linestyle="--")
            axarr[cnt_subplot].grid(linestyle='--', linewidth=1, alpha = 0.2)
            axarr[cnt_subplot].set_title("Normalized squared error through sliding window")
            #axarr[cnt_subplot].legend(["x", "y", "z", "average"], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True)
            axarr[cnt_subplot].set_yticks(np.arange(0,1.6, 0.3))
            plt.annotate('local minimum', xy=(2150, 0.6), xytext=(2150, 1.2), arrowprops=dict(facecolor='black', shrink=0.05), fontsize='large')
            plt.annotate('global minimum', xy=(5520, 0.40), xytext=(5300, 1.0), arrowprops=dict(facecolor='black', shrink=0.05), fontsize='large')
            cnt_subplot += 1

        plt.xlim([args.xlim_start, args.xlim_end])
        plt.xticks(np.arange(args.xlim_start, args.xlim_end + 1, 500))
        plt.xlabel("timesteps")

        plt.legend()
        f.tight_layout()
        self._save_image(f, '/img/time_series_pred.png')

        #plt.figure()


        #plt.plot(error)
        plt.show()

    def time_series_motor_estimate(self):
        max_turning_speed_deg = 60 * 0.01 / 0.17
        max_turning_speed_our_scale = max_turning_speed_deg * 2. / 180.
        motor_estimate = np.zeros_like(self.motor_commands)

        print max_turning_speed_our_scale
        for i in range(self.numtimesteps):
            if i == 0:
                motor_estimate[0] = self.motor_commands[0]
            else:
                delta = self.motor_commands[i] - motor_estimate[i-1]

                for j in range(delta.shape[0]):

                    if np.abs(delta[j]) > max_turning_speed_our_scale:
                        delta[j] = np.sign(delta[j]) * max_turning_speed_our_scale

                motor_estimate[i] = motor_estimate[i - 1].copy() + delta

        f = plt.figure(figsize = (10,5))
        plt.title('Estimated motor position')
        for i in range(self.nummot):
            plt.plot(motor_estimate[:,i], label="M"+str(i) )
        plt.legend()
        self._save_image(f, '/img/time_series_motor_estimate.png')
        plt.show()


    def time_series_ham_var(self):
        f = plt.figure(figsize = (20,10))
        plt.plot(self.sensor_value_variance_ham)
        #plt.plot(self.sensor_prediction_variance_ham)
        #plt.plot(self.sensor_prediction_error_variance_ham)
        self._save_image(f, '/img/time_series_variance_ham.png')
        plt.show()

    def fft(self):
        from scipy.fftpack import fft, ifft
        n = self.numtimesteps / 2
        sampling_rate = 1. / 0.01

        frequencies_sensors = ifft(self.sensor_values, axis = 0)
        frequencies_predictions = ifft(self.sensor_prediction, axis = 0)

        x = np.linspace(0, sampling_rate/2, n, endpoint=True)

        plt.plot(x, np.mean(np.abs(frequencies_sensors[:n]), axis = 1))
        plt.plot(x, np.mean(np.abs(frequencies_predictions[:n]), axis = 1))

        plt.show()

    def learn_motor_sensor_linear(self):
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


        _var = np.std(self.sensor_values, axis=0)


        mse = np.mean((predTest - self.testData["sensor"]) ** 2)
        mse_var = np.mean(((predTest - self.testData["sensor"])/_var) ** 2)
        print mse
        print mse_var

        _y_min = np.ceil(np.min(self.testData["sensor"]))
        _y_max = np.floor(np.max(self.testData["sensor"]))
        _y_ticks = np.arange(_y_min, _y_max + 1, 1)

        f, ax = plt.subplots(3,2, sharey=True, sharex=True, figsize=(20,15))


        plt.rc('font', family='serif', size=30)

        for i in range(self.numsen):
            _x = i % 3
            _y = i / 3
            ax_selected = ax[_x, _y]
            selected_sensor = self.use_sensors[_y]

            ax_selected.plot(self.testData["sensor"][:,i], label='sensor measurements')
            ax_selected.plot(predTest[:,i], label='sensor prediction')

            ax_selected.grid(linestyle='--', linewidth=1, alpha = 0.2)
            plt.yticks(_y_ticks)
            plt.xticks(np.arange(0,predTest.shape[0],100))

            if _y == 0:
                ax_selected.set_ylabel(['x','y','z'][_x])

            if _x == 0:
                ax_selected.set_title(self.sensor_name_long[selected_sensor] + " in [$" + self.sensor_units[selected_sensor] + "$]", y = 1.08)
            if _x == 2:
                ax_selected.set_xlabel("timesteps")

        # Put a legend below current axis
        ax[2,0].legend(loc='lower left', bbox_to_anchor=(0, -0.7),
          fancybox=True, shadow=True, ncol=2)

        f.tight_layout(pad=3, h_pad=0.4, w_pad = 0.3)

        self._save_image(f, 'img/sensor_motor_linear.png', tight=True)

        plt.show()


    def learn_motor_sensor_igmm(self):
        # TODO: USE FOR EXPERIMENT
        parser = argparse.ArgumentParser()
        parser.add_argument("-dimensions", "--show_dimensions", type=int, default = 0)
        args, unknown_args = parser.parse_known_args()


        self._prepare_data_for_learning()

        trainingDataAll = np.hstack((self.trainingData["motor"], self.trainingData["sensor"]))
        testDataAll = np.hstack((self.testData["motor"], self.testData["sensor"]))
        testDataNan = np.hstack((self.testData["motor"], np.full_like(self.testData["sensor"], np.nan)))
        testDataSampled = np.zeros_like(testDataNan)

        print trainingDataAll.shape
        print testDataAll.shape

        gmm = IGMM_COND(min_components=3, max_components=60)
        print "training..."
        gmm.train(trainingDataAll)
        print "sampling..."
        for i in range(testDataNan.shape[0]):
            testDataSampled[i,:] = gmm.sample_cond_dist(testDataNan[i,:], n_samples=1)

        print "plotting"


        if args.show_dimensions == 0 or args.show_dimensions > self.numsen:
            show_sensors = self.numsen
        else:
            show_sensors = args.show_dimensions
        i = 0
        f, axarr = plt.subplots(show_sensors + 1, figsize=(20,10))


        #print self.x_pred_coefficients[:-self.lag, :, 0]
        #print self.sensor_prediction[:-self.lag,0]
        for sensor in range(len(self.use_sensors)):
            sensor_name = self.use_sensors[sensor]
            for dim in range(self.sensor_dimensions[self.use_sensors[sensor]]):
                pred = testDataSampled[:-self.lag,i]

                # normal plot
                axarr[i + 1].plot(testDataAll[:-self.lag, i + self.nummot], label="sensor measurement")
                axarr[i + 1].plot(testDataSampled[:-self.lag, i + self.nummot], label="prediction")

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
        self._save_image(f, 'img/igmm_pred.png')
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

        line_offset = 0.1
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
                ax.set_ylabel(self.sensor_names_with_dimensions[dim])
                ax.set_xlabel("timesteps")
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


            x = range(cut_response)
            y_avg = np.mean(np.abs(avg_responses[j,:,:]), axis=1)
            y_std = np.mean(np.abs(std_responses[j,:,:]), axis = 1)
            print y_avg.shape
            y_offset = y_avg + j * line_offset
            y_low = y_offset - y_std
            y_high = y_offset + y_std

            ax.plot(x, y_low, c=colors[j], lw=1, alpha=line_alpha)
            ax.plot(x, y_high, c=colors[j], lw=1, alpha=line_alpha)

            ax.fill_between(x, y_low, y_high, facecolor=colors[j], lw=1, alpha=fill_alpha, label=text)

            ax.set_ylabel("$m/s^2$", fontsize=40)
            ax.set_xlabel("$ms$", fontsize=40)
            ax.grid(linestyle='--', linewidth=1, alpha = 0.2)
        ax.set_title("acceleration - y", fontsize=40)

        cbar = plt.colorbar(CS3)
        cbar.set_label("angle before step")
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        f.tight_layout()
        self._save_image(f, 'img/stepResponseAccelY.png')
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
            var = np.var(window, axis=0)

            # print "windooriginal", window

            windowfunction = np.hamming(self.windowsize)
            if(self.hamming):
                window = (window.T * windowfunction).T
            # print "windowshapehamming", window

            correlations = np.cov(window.T)

            # normalize
            for x in range(4):
                for j in range(4):
                    #correlations[x, j] = correlations[x, j] / \
                    #    np.sqrt(correlationsGlobal[x, x]
                    #            * correlationsGlobal[j, j])

                    # normalize by the average variance of both
                    correlations[x, j] = correlations[x, j] / \
                        ((var[x] + var[j])/2)


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

        parser = argparse.ArgumentParser()
        parser.add_argument("-cuts", "--cuts", type=int, default = 1)
        args, unknown_args = parser.parse_known_args()


        # settings, ammount of points in one graph and color mode
        points_per_plot = (int)(np.floor(self.timesteps / args.cuts))
        print points_per_plot
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

            columns = np.concatenate((['m'+str(i) for i in range(self.nummot)], ['s'+str(i) for i in range(self.numsen)]))
            print columns
            df = pd.DataFrame(combinedData, columns=columns)

            scatterm = scatter_matrix(df, alpha=0.4, s=25, figsize=(
                12, 12), diagonal='kde', c=c, edgecolors='none')

            for x in range(self.nummot + self.numsen):
                for y in range(self.nummot + self.numsen):
                    scatterm[x, y].set_ylim(-1.5, 1.5)
                    scatterm[x, y].set_xlim(-1.5, 1.5)

            # plt.draw()
            # plt.show()

            #print("scatter %d saved" %(part))
            plt.show()

    def custom_scattermatrix(self):
        color_cuts = 100

        motor_commands_parts = np.split(self.motor_commands, color_cuts, axis=0)
        sensov_values_parts = np.split(self.sensor_values, color_cuts, axis=0)

        f, ax = plt.subplots(self.nummot, self.numsen)
        colors = cm.cool(np.linspace(0, 1, color_cuts))

        for m in range(self.nummot):
            for s in range(self.numsen):
                for p in range(color_cuts):
                    ax[m,s].scatter(motor_commands_parts[p][:,m], sensov_values_parts[p][:,s], color=colors[p], alpha = 0.3)
        plt.show()

    def position_to_sensors(self):
        """ This function should be used with episodes recorded with a very slow sin_sweep."""
        if not hasattr(self.variable_dict, 'robot.classname') or self.variable_dict['robot.classname'] != 'PuppyConfigSinSweep':
            warnings.warn('This analyzing mode is made specifically for the sin sweep configuration, which was not detected.')

        # Scatter for all sensors
        part = self.numtimesteps / 16

        f, ax = plt.subplots(3,2, sharex=True, figsize=(20,15))

        #self.sensor_values -= np.mean(self.sensor_values, axis = 0)
        #self.sensor_values /= np.std(self.sensor_values, axis = 0)

        _xticks = np.linspace(-90,90,5)
        _yticks_acc = np.linspace(-0.9,0.9,4)
        _yticks_gyr = np.linspace(-0.15,0.15,3)
        sensor_means = np.mean(self.sensor_values, axis = 0)

        plt.rc('font', family='serif', size=30)

        for i in range(self.numsen):
            _x = i % 3
            _y = i / 3
            ax_selected = ax[_x, _y]
            selected_sensor = self.use_sensors[_y]
            for j in range(16):
                if j%4 == 0 or j%4==3:
                    #forward moving
                    color = 'r'
                else:
                    #backward moving
                    color = 'b'

                ax_selected.scatter(self.motor_commands[part * j: part * (j + 1),0] * 90., self.sensor_values[part * j: part * (j + 1),i], alpha = 0.05, color=color)

            #fake_scatter for legend
            legend_scatter_forwards = ax_selected.scatter( np.NaN, np.NaN, marker = 'o', color='r', label='forwards' )
            legend_scatter_backwards = ax_selected.scatter( np.NaN, np.NaN, marker = 'o', color='b', label='backwards' )


            ax_selected.grid(linestyle='--', linewidth=1, alpha = 0.2)
            #plt.yticks(_y_ticks)
            plt.xticks(_xticks)

            if _y == 0:
                #accelerometer add xyz labels
                ax_selected.set_ylabel(['x','y','z'][_x])
                ax_selected.set_yticks(_yticks_acc + np.round(sensor_means[i], 1))
            else:
                #gyro
                ax_selected.set_yticks(_yticks_gyr + np.round(sensor_means[i], 1))

            if _x == 0:
                # first row add titles
                ax_selected.set_title(self.sensor_name_long[selected_sensor] + " in [$" + self.sensor_units[selected_sensor] + "$]", y = 1.08)
            if _x == 2:
                #last row add x units
                ax_selected.set_xlabel("degrees")

        # Put a legend below current axis
        lgnd = ax[2,0].legend(loc='lower left', bbox_to_anchor=(0, -0.7),
          fancybox=True, shadow=True, ncol=2)
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]


        f.tight_layout(pad=3, h_pad=0.4, w_pad = 0.3)
        self._save_image(f, 'img/position_to_sensor.png', tight=True)

        plt.show()


        # Variance of all
        # TODO: Not working
        # steps = np.arange(np.min(self.motor_commands), np.max(self.motor_commands), 0.1)
        # print steps
        #
        # variances = np.zeros((len(steps) - 1, self.numsen))
        # means = np.zeros_like(variances)
        # for i in range(len(steps) - 1):
        #     indices = np.where(self.motor_commands[:,0] >= steps[i]) and np.where(self.motor_commands[:,0] <= steps[i + 1])
        #     variances[i] = np.var(self.sensor_values[indices], axis = 0)
        #     means[i] = np.average(self.sensor_values[indices], axis = 0)
        #
        # variances /= np.var(self.sensor_values, axis = 0)
        # means -= np.average(self.sensor_values, axis = 0)
        #
        # plt.figure()
        # colors = cm.cool(np.linspace(0, 1, self.numsen))
        # x = np.arange(-1, 1, 2./variances.shape[0])
        # for i in range(self.numsen):
        #     print((means[:,i] - variances[:,i]).shape)
        #     print((means[:,i] + variances[:,i]).shape)
        #     print(x.shape)
        #     plt.fill_between(x, means[:,i] - variances[:,i], means[:,i] + variances[:,i], facecolor = colors[i], alpha = 0.2)
        #     #plt.plot(means)
        #     #plt.plot(means + variances)

        plt.show()

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
        'details': Analyzer.details,                                # Check
        'ts': Analyzer.time_series,                                 # Check
        'ts_ms': Analyzer.time_series_motors_sensors,               # Check
        'ts_pred': Analyzer.time_series_sensor_prediction,          # Check
        'learn_linear': Analyzer.learn_motor_sensor_linear,
        'cor': Analyzer.correlation_func,
        'pos_to_sen': Analyzer.position_to_sensors,
        'scat': Analyzer.scattermatrix_func,
        'scat_slim': Analyzer.custom_scattermatrix,
        'lin_cross': Analyzer.learn_motor_sensor_linear_cross,
        'mlp_cross': Analyzer.learn_motor_sensor_mlp_cross,
        'igmm': Analyzer.learn_motor_sensor_igmm,
        'matrices_coefficients': Analyzer.matrices_coefficients,
        'model_matrix_animate': Analyzer.model_matrix_animate,
        'model_matrix_plot_smooth': Analyzer.model_matrix_plot_smooth,
        'find_emb': Analyzer.find_best_embsize,
        'activity': Analyzer.activity_plot,
        'step_sweep': Analyzer.step_sweep,
        'fft': Analyzer.fft,
        'hist': Analyzer.hist,
        'ts_me': Analyzer.time_series_motor_estimate,
        'ham_var' : Analyzer.time_series_ham_var,
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
