import os
# not using GPU because of GPU non-determinism (randomness)
# https://discuss.pytorch.org/t/how-to-make-a-cuda-available-using-cuda-visible-devices/45186
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from datetime import datetime

# for GPU error "OP_REQUIRES failed at cudnn_rnn_ops.cc:1510 : Unknown: Fail to find the dnn implementation."
from tensorflow.python.keras import backend
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

from arithmeticcompress import ArithmeticCompress
from arithmeticdecompress import ArithmeticDecompress

SEED = 7                        # universal seed for all
VERBOSE = 2                     # verbosity for fitting function and more (+)

# setting random seed in numpy and tensorflow so we always get the same output from the same input
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.seed.html
np.random.seed(SEED)
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(SEED)

# not using GPU because of GPU non-determinism (randomness)
# https://riptutorial.com/tensorflow/example/31875/run-tensorflow-on-cpu-only---using-the--cuda-visible-devices--environment-variable-
tf.config.set_visible_devices([], 'GPU')

pid = os.getpid()

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
    data = []                   # will be filled in load_data
    batch_size = 1              # also seems to be set in stone?
    num_epochs = 1              # set in stone!
    trunc_bp_len = 64
    total_series_len = -1       # will be set in load_data => len(data)
    time_steps = -1             # will be set in load_data => total_series_len//batch_size//num_features
    dictionary = []             # will be set in load_data => set(data)
    dictionary_size = 256       # set in stone!
    num_features = dictionary_size          # set in stone! (was: will be set after one hot encoding to dictionary_size)
    dictionary_frequency = []
    dictionary_frequency__max_from_rnn = [0 for _ in range(dictionary_size)]

    base_training_on = 'loss'   # 'loss' or 'accuracy' - unused actually
    from_file = ''              # will be set in __init__
    to_file = ''                # will be set in __init__

    wr = []                     # list of file writers; for debugging

    def load_data(self):
        #text = np.loadtxt(fname="data/space_separated/input501_4.txt", dtype=int, delimiter=" ")
        with open(self.from_file, 'rb') as file:
            byte = file.read(1)
            while byte:
                self.data.append(byte[0])
                byte = file.read(1)

        self.total_series_len = len(self.data)
        self.dictionary = set(self.data)
        self.dictionary_frequency = [(x, self.data.count(x)) for x in self.dictionary]      # informational only

        printt(f'Data loaded.')

    def print_hyperparams_and_data_info(self):
        printt('Hyper-parameters and data info:')
        print(f'\tBatch size: {str(self.batch_size)}')
        print(f'\tNumber of epochs: {str(self.num_epochs)}')
        print(f'\tTotal data series length: {str(self.total_series_len)}')
        print(f'\tNumber of batches (time steps): {str(self.time_steps)}')
        print(f'\tDictionary entries: ', self.dictionary)
        print(f'\tDictionary size (later number of features): {str(self.dictionary_size)}')

    def open_file_for_writing(self, file_path):
        i = len(self.wr)
        self.wr.append(open(file_path, "wt"))
        return i

    def close_file_for_writing(self, i=None):
        if i is None:
            for x in range(len(self.wr)):
                self.wr[x].close()
        else:
            self.wr[i].close()

    # NOT USED !
    # simple implementation for binary one hot encoding
    def one_hot_encode_test(self, array_of_data):
        new_array = [[1, 0] if x == 0 else [0, 1] for x in array_of_data]
        return new_array

    def one_hot_encode(self, array_of_data, num_of_classes):
        new_array = tf.one_hot(array_of_data, num_of_classes, dtype=tf.dtypes.int32)
        return new_array

    def prepare_data(self):
        data_array = list(self.data)
        data_array = self.one_hot_encode(data_array, self.dictionary_size)

        real_time_steps = self.total_series_len // self.batch_size
        data_final = np.array(data_array).reshape((self.batch_size, real_time_steps, self.num_features))

        return data_final

    def shape_input_data(self, input_data):
        return np.array(input_data).reshape((self.batch_size, 1, self.trunc_bp_len))    # should be self.trunc_bp_len

    def get_starting_input_array(self):
        return self.shape_input_data([0 for _ in range(self.trunc_bp_len)])             # should be self.trunc_bp_len

    def shift_and_replace_last(self, input_data, new_element):
        shape = input_data.shape
        data_array = input_data[0][0][1:].tolist()
        data_array.append(new_element)
        elems = np.array(data_array)

        return elems.reshape(shape)

    def build_model(self, data_final=None, is_only_training=False):
        # debugging purposes
        # print(data_array, '\r\n', data_final)

        batch_input_shape = ((self.batch_size, None, self.trunc_bp_len), (self.time_steps, self.batch_size, self.trunc_bp_len))[is_only_training]

        model = keras.Sequential(name='eNcoder-network')
        model.add(layers.GRU(self.trunc_bp_len,
                              name='gru-1',
                              stateful=True,
                              return_sequences=True,
                              batch_input_shape=batch_input_shape,
                              kernel_initializer=keras.initializers.RandomNormal()
                              )
                  )
        #model.add(layers.GRU(self.trunc_bp_len,
        #                      name='gru-2',
        #                      stateful=True,
        #                      return_sequences=True,
        #                      batch_input_shape=batch_input_shape,
        #                      kernel_initializer=keras.initializers.RandomNormal()
        #                      )
        #          )
        model.add(layers.Dense(self.num_features, activation='softmax', name='dense-softmax-1'))
        # model.add(layers.Activation(activation='softmax', name='softmax-act-1'))

        model.summary()
        print("Inputs: {}".format(model.input_shape))
        print("Outputs: {}".format(model.output_shape))
        if data_final is not None and data_final.any():
            print("Actual input: {}".format(data_final.shape))
            print("Actual output: {}".format(data_final.shape))

        # not using loss='sparse_categorical_crossentropy' since that's for non-hot encoded
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7),  # defaults actually :/
            loss='categorical_crossentropy',
            metrics=[tf.metrics.CategoricalAccuracy()]
        )

        # HERE we can return the model -- return model
        return model

    def compress(self, model, data_final):
        # written to txt file to check consecutive runs
        doc1 = self.open_file_for_writing('data/compressing.txt')
        doc2 = self.open_file_for_writing('data/compressing2.txt')

        now = datetime.now()
        predicting_time = now - now
        training_time = now - now
        compressing_time = now - now

        enc = ArithmeticCompress(self.to_file)
        enc.start(dictionary_size=self.dictionary_size)

        enc.compress_next(None, self.data[0])

        inp = self.get_starting_input_array()

        until = data_final.shape[1]
        tenPercent = until // 10
        onePercent = until // 100
        tens = 0
        ones = 0
        for i in range(until-1):
            #inp = data_final[:,i,:].reshape((self.batch_size, self.time_steps, self.num_features))
            inp = self.shift_and_replace_last(inp, self.data[i])
            tar = data_final[:,i+1,:].reshape((self.batch_size, self.time_steps, self.num_features))

            # PREDICTING ##############################################
            ###########################################################
            start = datetime.now()

            predict_output = model.predict_on_batch(inp)

            lasted = datetime.now() - start
            predicting_time += lasted
            ###########################################################

            # COMPRESSING #############################################
            ###########################################################
            start = datetime.now()

            compress_input = predict_output[0][0].numpy().tolist()
            enc.compress_next(compress_input, self.data[i+1])

            lasted = datetime.now() - start
            compressing_time += lasted
            ###########################################################

            # TRAINING ################################################
            ###########################################################
            start = datetime.now()

            training_output = model.train_on_batch(x=inp, y=tar, reset_metrics=False)

            lasted = datetime.now() - start
            training_time += lasted
            ###########################################################

            #if i % tenPercent == 0:
            #    printt(f'{str(tens)} % done...')
            #    tens = tens + 10

            if i % onePercent == 0:
                printt(f'{str(ones)} % done...\t Loss: ' + '{:.7f}'.format(training_output[0]) + f'\t Accuracy: {str(training_output[1])}')
                ones = ones + 1


            # written to txt file to check consecutive runs
            # self.wr.write(str(output.history['loss'][0]) + '\r\n')
            max_value = max(compress_input)
            max_index = compress_input.index(max_value)
            #self.wr[doc1].write(str(compress_input) + '\n' + str(max_index) + '    ' + '{:.12f}'.format(max_value) + '\n')
            self.dictionary_frequency__max_from_rnn[max_index] += 1

            #printt('Output history: ')
            #print(output.history)

        enc.stop()
        printt('Done!')

        print('Predicting lasted: ', predicting_time)
        print('Compression lasted: ', compressing_time)
        print('Training lasted: ', training_time)

        self.wr[doc2].write('Statistical frequency:')
        for x in self.dictionary_frequency:
            self.wr[doc2].write('\n' + str(x[0]) + '\t - \t' + str(x[1]))

        self.wr[doc2].write('\n\nFrequency from RNN (maxs):')
        for i in range(self.dictionary_size):
            if self.dictionary_frequency__max_from_rnn[i] > 0:
                self.wr[doc2].write('\n' + str(i) + '\t - \t' + str(self.dictionary_frequency__max_from_rnn[i]))

        # written to txt file to check consecutive runs
        self.close_file_for_writing()

    # not working with the new model_build
    def decompress(self, model):
        # written to txt file to check consecutive runs
        doc1 = self.open_file_for_writing('data/decompressing.txt')

        now = datetime.now()
        predicting_time = now - now
        training_time = now - now
        decompressing_time = now - now

        dec = ArithmeticDecompress(self.from_file, self.to_file)
        dec.start(dictionary_size=self.dictionary_size)

        symbol1 = dec.decompress_next(None)
        #symbol2 = dec.decompress_next(DEFAULT_PROB_DISTR_EST_2)

        count = 0
        while symbol1 != 256:
            input_data = self.one_hot_encode([symbol1], self.dictionary_size)
            inp = np.array(input_data).reshape((self.batch_size, self.time_steps, self.num_features))

            # PREDICTING ##############################################
            ###########################################################
            start = datetime.now()

            predict_output = model.predict_on_batch(inp)

            lasted = datetime.now() - start
            predicting_time += lasted
            ###########################################################

            # DECOMPRESSING ###########################################
            ###########################################################
            start = datetime.now()

            decompress_input = predict_output[0][0].numpy().tolist()
            symbol2 = dec.decompress_next(decompress_input)

            lasted = datetime.now() - start
            decompressing_time += lasted
            ###########################################################

            # TRAINING ################################################
            ###########################################################
            start = datetime.now()

            target_data = self.one_hot_encode([symbol2], self.dictionary_size)
            tar = np.array(target_data).reshape((self.batch_size, self.time_steps, self.num_features))
            output = model.train_on_batch(x=inp, y=tar, reset_metrics=False)

            lasted = datetime.now() - start
            training_time += lasted
            ###########################################################

            # written to txt file to check consecutive runs
            self.wr[doc1].write(str(output) + '\r\n')

            symbol1 = symbol2

            count += 1
            if count % 1000 == 0:
                printt(f'Another 1000 bytes done...')

        dec.stop()
        printt('Done!')

        print('Predicting lasted: ', training_time)
        print('Decompression lasted: ', decompressing_time)
        print('Training lasted: ', training_time)

    # not working with the new model_build
    def load_data_for_training(self):
        with open(self.from_file, 'rb') as file:
            byte = file.read(1)
            while byte:
                self.data.append(byte[0])
                byte = file.read(1)

        self.total_series_len = int((len(self.data) // self.batch_size) * self.batch_size) + 1
        self.data = self.data[:self.total_series_len] # +1 here is for when training taking last/first sample for x/y
        self.dictionary = set(self.data)
        printt(f'Data loaded.')

    # not working with the new model_build
    def prepare_data_for_training(self):

        printt('Preparing data for training...')

        data_array = list(self.data)

        new_array = [self.one_hot_encode(data_array[i:i+self.batch_size], self.dictionary_size) for i in range(self.total_series_len - self.batch_size)]

        self.time_steps = self.total_series_len - self.batch_size
        data_final = np.array(new_array).reshape((self.time_steps, self.batch_size, self.num_features))

        printt('Data for training prepared.')

        return data_final

    # not working with the new model_build
    def train_for_testing(self, model, data_final):

        output = model.fit(
           x=data_final[:-1, :,:],
           y=data_final[1:, :, :],
           verbose=VERBOSE,
           epochs=self.num_epochs,
           batch_size=self.time_steps,
           #callbacks=[CustomCallback()],
           shuffle=False
        )

        printt('Done!')

    def __init__(self, from_file, to_file):
        self.from_file = from_file
        self.to_file = to_file
        printt('eNcoder initiated.')


if __name__ == '__main__':

    op = 'encode'       # 'encode' or 'decode' or 'testing'*

    if op == 'encode':
        lstmEnc = LstmEncoder(
            from_file=f'data/arbitrary/seq_32_input_100k.bin',
            to_file=f'data/compressed/compressed_seq_32_input_100k_2.bin'
        )
        lstmEnc.load_data()
        lstmEnc.print_hyperparams_and_data_info()
        data = lstmEnc.prepare_data()
        model = lstmEnc.build_model(data)
        lstmEnc.compress(model, data)
    elif op == 'decode':
        lstmEnc = LstmEncoder(
            from_file=f'data/compressed/compressed_input60k_all.bin',
            to_file=f'data/decompressed/input60k_all.txt',
        )
        lstmEnc.print_hyperparams_and_data_info()
        model = lstmEnc.build_model()
        lstmEnc.decompress(model)
    else:
        lstmEnc = LstmEncoder(
            from_file=f'data/arbitrary/input5k_all.txt',
            to_file=f'data/compressed/compressed_input5k_all.bin'
        )
        lstmEnc.batch_size = 5
        lstmEnc.load_data_for_training()
        lstmEnc.print_hyperparams_and_data_info()
        data = lstmEnc.prepare_data_for_training()
        model = lstmEnc.build_model(data_final=data, is_only_training=True)
        lstmEnc.train_for_testing(model, data)
