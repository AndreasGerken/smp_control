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

        self.frequent_buffer = None

    """ internal functions """

    def _resolve_key_recurrent(self, _dict, key):
        """ Resolve a key from a dictionary recurrent.
        Returns the value of the key if the key does not contain a point '.'
        elsewise the part of the key before the point is resolved and this function
        is called for the rest of the keyname recursively."""

        # split the key at each point
        splitted_key = key.split('.')

        if len(splitted_key) > 1:
            assert len(splitted_key) < 50, "The key has to many points"

            # The key included points and this function should be called once more
            key_front = splitted_key[0]
            key_back = '.'.join(splitted_key[1:])
            deeper_dict = _dict[key_front].__dict__

            return self._resolve_key_recurrent(deeper_dict, key_back)
        else:
            # The key does not include points and the value can be returned
            return _dict[key]

    def _add_to_list(self, variableNames, variableList):
        """ Extend a list of variable names with new names and check if they exist
        in the source dictionary"""

        # convert scalar to list if necessary
        if not isinstance(variableNames, list):
            variableNames = variableNames.tolist()

        # check if all variableNames are available in source instance
        for key in variableList:
            assert key in self.source_dict.keys(), "The key '%s' was not in the source dict" % (key)

        # add the variableNames to the existing list
        variableList.extend(variableNames)

    def _initialize_frequent_buffer(self):
        """ Initialize the frequent buffer to the shapes of the variables and add
        a time dimension with the size of numtimesteps """
        self.frequent_buffer = {key: np.zeros((self.numtimesteps,) + np.array(
            self.source_dict[key]).shape) for key in self.variableNamesFrequent}

    def add_once_variables(self, variable_names):
        """ This function adds a variable which will be saved once, it should not
        change."""
        self._add_to_list(variable_names, self.variableNamesOnce)

    def add_frequent_variables(self, variable_names):
        """ This function adds a variable which will be saved at every timestep."""
        self._add_to_list(variable_names, self.variableNamesFrequent)
        self._initialize_frequent_buffer()

    def save_frequent_variables_to_buffer(self, i):
        """ This method saves all frequent variables to the buffer at the given
        timestep."""
        for key in self.variableNamesFrequent:
            self.frequent_buffer[key][i] = self.source_dict[key]

    def save_pickle(self, pickle_file_name):
        """ This function collects the data from the instance and from the buffer
        and saves them to a pickle file. The saved variable is a dictionary."""

        # copy the variables from the source_dict
        once_dict = {key: self._resolve_key_recurrent(
            self.source_dict, key) for key in self.variableNamesOnce}

        # copy the frequent variables from the buffer
        frequent_dict = {key: self._resolve_key_recurrent(
            self.frequent_buffer, key) for key in self.variableNamesFrequent}

        # merge the dicts and write them to the given file
        once_dict.update(frequent_dict)
        pickle.dump(once_dict, open(pickle_file_name, 'wb'))

        print "Variables saved in pickle file once: [%s] freq: [%s] filename = %s" % (', '.join(self.variableNamesOnce), ', '.join(self.variableNamesFrequent), pickle_file_name)
