#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    24-Jul-2023 18:42:51

import tensorflow as tf
import numpy as np
import sys     # Remove this line after completing the layer definition.

class packingGLayer(tf.keras.layers.Layer):
    # Add any additional layer hyperparameters to the constructor's
    # argument list below.
    def __init__(self, name=None):
        super(packingGLayer, self).__init__(name=name)
        self.Description = "Packing Gaussian pyr layer"
        self.NumInputs = 4

    def call(self, input1, input2, input3, input4):
        # Add code to implement the layer's forward pass here.
        # The input tensor format(s) are: BSSC, BSSC, BSSC, BSSC
        # The output tensor format(s) are: BSSC
        # where B=batch, C=channels, T=time, S=spatial(in order of height, width, depth,...)

        sz = tf.shape(input1)
        if len(input1.shape) == 4:
            L = sz[3]//4
            # output1 = tf.zeros((sz[0], sz[1], sz[2], sz[3] * 4), dtype=input1.dtype)
            # output1[:, :, :, 0:3] = input1
            # output1[:, 0:input2.shape[1], 0:input2.shape[2], 3:6] = input2
            # output1[:, 0:input3.shape[1], 0:input3.shape[2], 6:9] = input3 * 2
            # output1[:, 0:input4.shape[1], 0:input4.shape[2], 9:12] = input4 * 2 * 2
            x1 = input1
            x2 = tf.pad(input2, [[0, 0], [0, sz[1]-tf.shape(input2)[1]], [0, sz[2]-tf.shape(input2)[2]], [0, 0]], "CONSTANT")
            x3 = tf.pad(input3, [[0, 0], [0, sz[1]-tf.shape(input3)[1]], [0, sz[2]-tf.shape(input3)[2]], [0, 0]], "CONSTANT")
            x4 = tf.pad(input4, [[0, 0], [0, sz[1]-tf.shape(input4)[1]], [0, sz[2]-tf.shape(input4)[2]], [0, 0]], "CONSTANT")
            output1 = tf.concat([x1, x2, x3, x4], axis=3)
            # output1 = tf.stack([x1, x2, x3, x4], axis=3)

        else:
            L = sz[2]//4
            # output1 = np.zeros((sz[0], sz[1], sz[2] * 4), dtype=input1.numpy().dtype)
            # output1[:, :, 0:3] = input1
            # output1[0:input2.shape[0], 0:input2.shape[1], 3:6] = input2
            # output1[0:input3.shape[0], 0:input3.shape[1], 6:9] = input3 * 2
            # output1[0:input4.shape[0], 0:input4.shape[1], 9:12] = input4 * 2 * 2
            x1 = input1
            x2 = tf.pad(input2, [[0, sz[0]-tf.shape(input2)[0]], [0, sz[1]-tf.shape(input2)[1]], [0, 0]], "CONSTANT")
            x3 = tf.pad(input3, [[0, sz[0]-tf.shape(input3)[0]], [0, sz[1]-tf.shape(input3)[1]], [0, 0]], "CONSTANT")
            x4 = tf.pad(input4, [[0, sz[0]-tf.shape(input4)[0]], [0, sz[1]-tf.shape(input4)[1]], [0, 0]], "CONSTANT")
            output1 = tf.concat([x1, x2, x3, x4], axis=2)
            # output1 = tf.stack([x1, x2, x3, x4], axis=2)


        # # Remove the following 3 lines after completing the custom layer definition:
        # print("Warning: load_model(): Before you can load the model, you must complete the definition of custom layer packingGLayer in the customLayers folder.")
        # print("Exiting...")
        # sys.exit("See the warning message above.")

        return output1
