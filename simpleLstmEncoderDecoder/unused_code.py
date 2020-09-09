
class CustomCallback(keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        #print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        #print("...Training: end of batch {}; got log keys: {}".format(batch, keys))



    # NOT USED !
    # simple implementation for binary one hot encoding
    def one_hot_encode_test(self, array_of_data):
        new_array = [[1, 0] if x == 0 else [0, 1] for x in array_of_data]
        return new_array
