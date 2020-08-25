import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from datetime import datetime
import os
from tensorflow.python.keras import backend

pid = os.getpid()
SEED = 7                # universal seed for all
VERBOSE = 2             # verbosity for fitting function and +

def printt(text_to_print):
    now = datetime.now()
    print(f'({str(pid)}) {str(now)}: {text_to_print}')


class CustomCallback(keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        #print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        #print("...Training: end of batch {}; got log keys: {}".format(batch, keys))




class LstmEncoder:
    data = []                   # will be set in load_data
    batch_size = 1
    num_epochs = 1              # set in stone!
    #trunc_bp_len = 8
    total_series_len = -1       # will be set in load_data => len(data)
    time_steps = -1             # will be set in load_data => total_series_len//batch_size//num_features
    dictionary = []             # will be set in load_data => set(data)
    dictionary_size = -1        # will be set in load_data => len(dictionary)
    num_features = 1            # will be set after one hot encoding to dictionary_size

    def load_data(self):
        text = np.loadtxt(fname="data/space_separated/input501_4.txt", dtype=int, delimiter=" ")
        self.data.extend(text)
        self.total_series_len = len(self.data)
        self.dictionary = set(self.data)
        self.dictionary_size = len(self.dictionary)
        #self.num_features = len(self.dictionary)
        self.time_steps = (self.total_series_len) // self.batch_size // self.num_features
        printt(f'Data loaded.')

    def print_hyperparams_and_data_info(self):
        printt('Hyper-parameters and data info:')
        print(f'\tBatch size: {str(self.batch_size)}')
        print(f'\tNumber of epochs: {str(self.num_epochs)}')
        #print(f'\tTruncated backpropagation length: {str(self.trunc_bp_len)}')
        print(f'\tTotal data series length: {str(self.total_series_len)}')
        print(f'\tNumber of batches (time steps): {str(self.time_steps)}')
        print(f'\tDictionary entries: ', self.dictionary)
        print(f'\tDictionary size (later number of features): {str(self.dictionary_size)}')

    # NOT USED !
    # simple implementation for binary one hot encoding
    def one_hot_encode_test(self, array_of_data):
        new_array = [[1, 0] if x == 0 else [0, 1] for x in array_of_data]
        return new_array

    def one_hot_encode(self, array_of_data, num_of_classes):
        new_array = tf.one_hot(array_of_data, num_of_classes, dtype=tf.dtypes.int32)
        return new_array

    def build_model(self):

        #inputs_array = self.data[:-1]
        #targets_array = self.data[1:]
        #inputs_encoded = self.one_hot_encode(inputs_array, self.dictionary_size)
        #targets_encoded = self.one_hot_encode(targets_array, self.dictionary_size)

        data_array = self.data
        data_encoded = self.one_hot_encode(data_array, self.dictionary_size)

        self.num_features = self.dictionary_size

        #inputs = np.array(inputs_encoded)
        #targets = np.array(targets_encoded)
        #inputs_final = inputs.reshape((self.batch_size, self.time_steps, self.num_features))
        #targets_final = targets.reshape((self.batch_size, self.time_steps, self.num_features))

        data_final = np.array(data_encoded).reshape((self.batch_size, self.time_steps, self.num_features))

        print(data_array, '\r\n', data_encoded, '\r\n', data_final)

        model = keras.Sequential(name='eNcoder-network')
        model.add(layers.LSTM(self.num_features,
                              name='eNcoder-lstm-1',
                              stateful=True,
                              return_sequences=True,
                              batch_input_shape=(self.batch_size, None, self.num_features),
                              kernel_initializer=keras.initializers.RandomNormal(seed=SEED),
                              #bias_constraint=keras.initializers.Zeros()
                              )
                  )

        model.summary()
        print("Inputs: {}".format(model.input_shape))
        print("Outputs: {}".format(model.output_shape))
        #print("Actual input: {}".format(inputs_final.shape))
        #print("Actual output: {}".format(targets_final.shape))
        print("Actual input: {}".format(data_final.shape))
        print("Actual output: {}".format(data_final.shape))

        # not using loss='sparse_categorical_crossentropy' since that's for non-hot encoded
        #model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        until = data_final.shape[1]
        for i in range(until-1):
            inp = data_final[:,i,:].reshape((self.batch_size, 1, self.num_features))
            tar = data_final[:,i+1,:].reshape((self.batch_size, 1, self.num_features))

            if VERBOSE > 0:
                printt(f'i == {i}/{until}')
            if VERBOSE > 1:
                print('char == ', inp, ' -> ', tar)

            output = model.fit(
                x=inp,
                y=tar,
                verbose=VERBOSE,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                #callbacks=[CustomCallback()],
                shuffle=False
            )

            #lstm = backend.function([model.layers[0].input], [model.layers[0].output])

            #printt('Output history: ')
            #print(output.history)

    def __init__(self):
        printt('eNcoder initiated.')


if __name__ == '__main__':
    enc = LstmEncoder()
    enc.load_data()
    enc.print_hyperparams_and_data_info()
    enc.build_model()
