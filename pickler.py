import cPickle as pickle
import numpy as np
import warnings

# TODO: Implement
class Pickler():
    def __init__(self, source_instance, numtimesteps):
        self.variableNamesOnce = []
        self.variableNamesFrequent = []

        self.numtimesteps = numtimesteps
        self.source_instance = source_instance
        self.source_dict = self.source_instance.__dict__

        self.frequentBuffer = None

    def addOnceVariables(self, variableNames):
        self._addToList(variableNames, self.variableNamesOnce)

    def addFrequentVariables(self, variableNames):
        self._addToList(variableNames, self.variableNamesFrequent)

    def initializeFrequentBuffer(self):
        self.frequentBuffer = {key: np.zeros((self.numtimesteps,) + np.array(self.source_dict[key]).shape) for key in self.variableNamesFrequent}

    def saveFrequentVariablesToBuffer(self, i):
        for key in self.variableNamesFrequent:
            self.frequentBuffer[key][i] = self.source_dict[key]

    def save_pickle(self, pickleName):
        # copy the requested items from the source_dict
        once_dict = {key:self.source_dict[key] for key in self.variableNamesOnce}
        frequent_dict = {key:self.frequentBuffer[key] for key in self.variableNamesFrequent}

        # merge the dicts and write them to the given file
        once_dict.update(frequent_dict)
        pickle.dump(once_dict, open(pickleName, 'wb'))

        print "Variables saved in pickle file once: [%s] freq: [%s] filename = %s" % (', '.join(self.variableNamesOnce), ', '.join(self.variableNamesFrequent), pickleName)

    def _addToList(self, variableNames, variableList):
        # convert scalar to list if necessary
        if not isinstance(variableNames, list):
            variableNames = variableNames.tolist()

        # check if all variableNames are available in source instance
        if not np.all([i in self.source_dict.keys() for i in variableList]):
            warnings.warn("Some variables which should be added to the pickler are not available.")

        # add the variableNames to the existing list
        variableList.extend(variableNames)
