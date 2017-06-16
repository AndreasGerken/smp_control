import cPickle as pickle
import numpy as np
import warnings

# TODO: Implement
class Pickler():
    def __init__(self, source_instance):
        self.variableNames = []
        self.source_instance = source_instance

    def addVariables(self, variableNames):
        # convert scalar to list if necessary
        if not isinstance(variableNames, list):
            variableNames = variableNames.tolist()

        # check if all variableNames are available in source instance
        if not np.all([i in self.source_instance.__dict__.keys() for i in variableNames]):
            warnings.warn("Some variables which should be added to the pickler are not available.")

        # add the variableNames to the existing list
        self.variableNames.extend(variableNames)

    def save_pickle(self, pickleName):
        # get the variable dict from the source instance
        source_dict = self.source_instance.__dict__

        # copy the requested items from the source_dict
        save_dict = {key:source_dict[key] for key in self.variableNames}

        # write the dictionary to the given file
        pickle.dump(save_dict, open(pickleName, 'wb'))

        print "Variables [%s] saved in pickle file: %s" % (", ".join(self.variableNames), pickleName)
