import os

# not using GPU because of GPU non-determinism (randomness)
# https://discuss.pytorch.org/t/how-to-make-a-cuda-available-using-cuda-visible-devices/45186
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from datetime import datetime
from decimal import *
from byte_huffman import HuffmanEncoder, HuffmanDecoder

# for GPU error "OP_REQUIRES failed at cudnn_rnn_ops.cc:1510 : Unknown: Fail to find the dnn implementation."
from tensorflow.python.keras import backend

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

from arithmeticcompress import ArithmeticCompress
from arithmeticdecompress import ArithmeticDecompress

SEED = 7  # universal seed for all
VERBOSE = 2  # verbosity for fitting function and more (+)

ARITHMETIC_CODER_TYPE = 'arithmetic'
HUFFMAN_CODER_TYPE = 'huffman'

# not using GPU because of GPU non-determinism (randomness)
# https://riptutorial.com/tensorflow/example/31875/run-tensorflow-on-cpu-only---using-the--cuda-visible-devices--environment-variable-
tf.config.set_visible_devices([], 'GPU')

pid = os.getpid()

# initializing log file to log everything printed with printt(...) function
log_file = open(f'./data/logs/log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', 'wt')

# precision for Decimal numbers - useless now
getcontext().prec = 128

# setting random seed in numpy and tensorflow so we always get the same output from the same input
def seed_randoms():
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.seed.html
    np.random.seed(SEED)
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(SEED)


# def open_log_file():
#    log_file = open(f'./data/logs/log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', 'wt')

def close_log_file():
    log_file.close()


def printt(text_to_print):
    now = datetime.now()
    msg = f'({str(pid)}) {str(now)}: {text_to_print}'
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')


class RnnCoder:
    data = []                           # will be filled in load_data
    batch_size = 1                      # also seems to be set in stone?
    num_epochs = 1                      # set in stone!
    trunc_bp_len = 128                  # adjustable at will (number of neurons in RNN layer)
    history_len = 64                    # adjustable at will (number of previous symbols to use in every batch)
    total_series_len = -1               # will be set in load_data => len(data)
    time_steps = -1                     # will be set in load_data => total_series_len//batch_size//num_features
    dictionary = []                     # will be set in load_data => set(data)
    dictionary_size = 256               # set in stone!
    num_features = dictionary_size      # set in stone! (was: will be set after one hot encoding to dictionary_size)
    dictionary_frequency = []
    dictionary_frequency__max_from_rnn = [0 for _ in range(dictionary_size)]

    base_training_on = 'loss'           # 'loss' or 'accuracy' - unused actually
    from_file = ''                      # will be set in __init__
    to_file = ''                        # will be set in __init__

    wr = []                             # list of file writers; for debugging

    def load_data(self):
        with open(self.from_file, 'rb') as file:
            byte = file.read(1)
            while byte:
                self.data.append(byte[0])
                byte = file.read(1)

        self.total_series_len = len(self.data)
        self.dictionary = set(self.data)
        self.dictionary_frequency = [(x, self.data.count(x)) for x in self.dictionary]  # informational only

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

    def one_hot_encode(self, array_of_data, num_of_classes):
        new_array = tf.one_hot(array_of_data, num_of_classes, dtype=tf.dtypes.int32)
        return new_array

    def prepare_expected_output_data(self):
        data_array = list(self.data)
        data_array = self.one_hot_encode(data_array, self.dictionary_size)

        real_time_steps = self.total_series_len // self.batch_size
        data_final = np.array(data_array).reshape((self.batch_size, real_time_steps, self.num_features))

        return data_final

    def shape_input_data(self, input_data):
        return np.array(input_data).reshape((self.batch_size, 1, self.history_len))

    def get_starting_input_array(self):
        return self.shape_input_data([0 for _ in range(self.history_len)])

    def shift_and_replace_last(self, input_data, new_element):
        shape = input_data.shape
        data_array = input_data[0][0][1:].tolist()
        data_array.append(new_element)
        elems = np.array(data_array)

        return elems.reshape(shape)

    def build_model(self, data_final=None):
        batch_input_shape = (self.batch_size, None, self.history_len)

        model = keras.Sequential(name=f'{self.name}-network')
        model.add(layers.GRU(self.trunc_bp_len,
                             name='gru-1',
                             stateful=True,
                             return_sequences=True,
                             batch_input_shape=batch_input_shape,
                             kernel_initializer=keras.initializers.RandomNormal()
                             )
                  )
        model.add(layers.Dense(self.num_features, activation='softmax', name=f'dense-softmax-1'))

        model.summary()
        printt("Inputs: {}".format(model.input_shape))
        printt("Outputs: {}".format(model.output_shape))
        if data_final is not None and data_final.any():
            printt("Actual input: {}".format(data_final.shape))
            printt("Actual output: {}".format(data_final.shape))

        # not using loss='sparse_categorical_crossentropy' since that's for non-hot encoded
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7),  # defaults actually :/
            loss='categorical_crossentropy',
            metrics=[tf.metrics.CategoricalAccuracy()]
        )

        return model

    def compress(self, model, data_final):
        # initialization of debugging log files
        doc1, doc2 = -1, -1
        if self.log_debugging_info:
            doc1 = self.open_file_for_writing('data/compressing.txt')
            doc2 = self.open_file_for_writing('data/compressing2.txt')

        now = datetime.now()
        predicting_time = now - now
        training_time = now - now
        compressing_time = now - now

        enc = None
        if self.encoder_type == HUFFMAN_CODER_TYPE:
            enc = HuffmanEncoder(self.to_file)
        elif self.encoder_type == ARITHMETIC_CODER_TYPE:
            enc = ArithmeticCompress(self.to_file)
        else:
            raise AttributeError(f'Unknown encoder type "{self.encoder_type}"')

        enc.start(dictionary_size=self.dictionary_size)

        enc.compress_next(None, self.data[0])

        if self.log_debugging_info:
            self.wr[doc1].write('x -> ' + str(self.data[0]) + '\n\n')
            self.wr[doc1].flush()

        inp = self.get_starting_input_array()

        until = data_final.shape[1]
        tenPercent = until // 10
        onePercent = until // 100
        tens = 0
        ones = 0
        for i in range(until - 1):
            # inp = data_final[:,i,:].reshape((self.batch_size, self.time_steps, self.num_features))
            inp = self.shift_and_replace_last(inp, self.data[i])
            tar = data_final[:, i + 1, :].reshape((self.batch_size, self.time_steps, self.num_features))

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
            if self.encoder_type == ARITHMETIC_CODER_TYPE:
                compress_input = [int(i * 1000000 + 1) for i in compress_input]
            enc.compress_next(compress_input, self.data[i + 1])

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

            # for really small files - useless now
            # if i % tenPercent == 0:
            #    printt(f'{str(tens)} % done...')
            #    tens = tens + 10

            if i % onePercent == 0:
                printt(f'{str(ones)} % done...\t Loss: ' + '{:.7f}'.format(
                    training_output[0]) + f'\t Accuracy: {str(training_output[1])}')
                ones = ones + 1

            # written to data/compressing.txt file to check consecutive runs - debugging purposes
            if self.log_debugging_info:
                max_value = max(compress_input)
                max_index = compress_input.index(max_value)
                self.wr[doc1].write('Probabilities: ' + str(compress_input) + '\n')
                self.wr[doc1].write('Max value index: ' + str(max_index) + ', Max value: ' + str(max_value) + '\n')
                self.wr[doc1].write('Encrypting: ' + str(self.data[i]) + ' -> ' + str(self.data[i + 1]) + '\n')
                self.wr[doc1].flush()
            #

            # printt('Output history: ')
            # print(output.history)

        enc.stop()

        self.print_file_stats()

        printt('')
        printt(f'Predicting lasted: {str(predicting_time)}')
        printt(f'Compression lasted: {str(compressing_time)}')
        printt(f'Training lasted: {str(training_time)}\n')
        printt(f'TOTAL TIME: {str(training_time + compressing_time + predicting_time)}')

        if self.log_debugging_info:
            self.wr[doc2].write('Statistical frequency:')
            for x in self.dictionary_frequency:
                self.wr[doc2].write('\n' + str(x[0]) + '\t - \t' + str(x[1]))

            self.wr[doc2].write('\n\nFrequency from RNN (maxs):')
            for i in range(self.dictionary_size):
                if self.dictionary_frequency__max_from_rnn[i] > 0:
                    self.wr[doc2].write('\n' + str(i) + '\t - \t' + str(self.dictionary_frequency__max_from_rnn[i]))

            self.close_file_for_writing()

    def decompress(self, model):
        # initialization of debugging log file
        doc1 = -1
        if self.log_debugging_info:
            doc1 = self.open_file_for_writing('data/decompressing.txt')

        now = datetime.now()
        predicting_time = now - now
        training_time = now - now
        decompressing_time = now - now

        dec = None
        if self.encoder_type == HUFFMAN_CODER_TYPE:
            dec = HuffmanDecoder(self.from_file, self.to_file)
        elif self.encoder_type == ARITHMETIC_CODER_TYPE:
            dec = ArithmeticDecompress(self.from_file, self.to_file)
        else:
            raise AttributeError(f'Unknown encoder type "{self.encoder_type}"')

        dec.start(dictionary_size=self.dictionary_size)

        symbol1 = dec.decompress_next(None)

        if self.log_debugging_info:
            self.wr[doc1].write('x -> ' + str(symbol1) + '\n\n')
            self.wr[doc1].flush()

        inp = self.get_starting_input_array()

        count = 0
        while symbol1 < self.dictionary_size:  # using dictionary_size because it's greater by 1 from max element

            inp = self.shift_and_replace_last(inp, symbol1)

            # PREDICTING #############################################################################
            ##########################################################################################
            start = datetime.now()

            predict_output = model.predict_on_batch(inp)

            lasted = datetime.now() - start
            predicting_time += lasted
            ##########################################################################################

            # DECOMPRESSING ##########################################################################
            ##########################################################################################
            start = datetime.now()

            decompress_input = predict_output[0][0].numpy().tolist()
            if self.encoder_type == ARITHMETIC_CODER_TYPE:
                decompress_input = [int(i * 1000000 + 1) for i in decompress_input]
            symbol2 = dec.decompress_next(decompress_input)

            lasted = datetime.now() - start
            decompressing_time += lasted
            ##########################################################################################

            # TRAINING ###############################################################################
            ##########################################################################################
            start = datetime.now()

            target_data = self.one_hot_encode([symbol2], self.dictionary_size)
            tar = np.array(target_data).reshape((self.batch_size, self.time_steps, self.num_features))
            output = model.train_on_batch(x=inp, y=tar, reset_metrics=False)

            lasted = datetime.now() - start
            training_time += lasted
            ##########################################################################################

            # written to data/decompressing.txt file to check consecutive runs - debugging purposes
            if self.log_debugging_info:
                max_value = max(decompress_input)
                max_index = decompress_input.index(max_value)
                self.wr[doc1].write('Probabilities: ' + str(decompress_input) + '\n')
                self.wr[doc1].write('Max value index: ' + str(max_index) + ', Max value: ' + str(max_value) + '\n')
                self.wr[doc1].write('Encrypting: ' + str(symbol1) + ' -> ' + str(symbol2) + '\n')
                self.wr[doc1].flush()
            #

            symbol1 = symbol2

            count += 1
            if count % 1000 == 0:
                printt(f'Another 1000 bytes done...')

        dec.stop()

        printt('')
        printt(f'Predicting lasted: {str(predicting_time)}')
        printt(f'Decompression lasted: {str(decompressing_time)}')
        printt(f'Training lasted: {str(training_time)}\n')
        printt(f'TOTAL TIME: {str(training_time + decompressing_time + predicting_time)}')

        # close debugging log files
        if self.log_debugging_info:
            self.close_file_for_writing()

    def print_file_stats(self):
        input_file_stats = os.stat(self.from_file)
        output_file_stats = os.stat(self.to_file)

        printt('')
        printt(f'Input file size: {str(input_file_stats.st_size)} bytes')
        printt(f'Output file size: {str(output_file_stats.st_size)} bytes')
        printt(f'Compression ratio: {"{:.3f}".format(input_file_stats.st_size / output_file_stats.st_size)}')

    def __init__(self, name, from_file, to_file, encoder_type, log_debugging_info=True):
        self.name = name
        self.from_file = from_file
        self.to_file = to_file
        self.encoder_type = encoder_type
        self.log_debugging_info = log_debugging_info
        printt(f'{self.name} initiated.')


if __name__ == '__main__':

    # open_log_file()

    file = [
        # 'seq_8_input_1k.txt'
        # 'input5k_all.txt'
        # 'input20k_all.txt'
        # 'seq_32_input_100k.bin'
        # 'seq_128_input_100k.bin'
        # 'seq_256_input_100k.bin'
        # 'seq_input_100k.bin'
        # 'input_HMM_0.3_HMM_10_markovity.txt'
        # 'input_HMM_0.3_HMM_20_markovity.txt'
        # 'input_HMM_0.3_HMM_40_markovity.txt'
        'input_HMM_0.3_HMM_90_markovity.txt'
    ][0]

    op = [
        'encode'
        # 'decode'
    ][0]

    coder_type = [
        HUFFMAN_CODER_TYPE
        # ARITHMETIC_CODER_TYPE
    ][0]

    printt(f' -- Started: {coder_type} - {op} -> {file}')

    # always seed the randoms before a new call !!!
    seed_randoms()

    if op == 'encode':
        rnnCoder = RnnCoder(
            name='eNcoder',
            from_file=f'data/arbitrary/{file}',
            to_file=f'data/compressed/compressed_{file}_{coder_type}.bin',
            encoder_type=coder_type,
            log_debugging_info=False
        )
        rnnCoder.load_data()
        rnnCoder.print_hyperparams_and_data_info()
        data = rnnCoder.prepare_expected_output_data()
        model = rnnCoder.build_model(data)
        rnnCoder.compress(model, data)

    if op == 'decode':
        rnnCoder = RnnCoder(
            name='dEcoder',
            from_file=f'data/compressed/compressed_{file}_{coder_type}.bin',
            to_file=f'data/decompressed/decompressed_{file}',
            encoder_type=coder_type,
            log_debugging_info=False
        )
        rnnCoder.print_hyperparams_and_data_info()
        model = rnnCoder.build_model()
        rnnCoder.decompress(model)

    printt(f' -- Done: {coder_type} - {op} -> {file}')

    close_log_file()
