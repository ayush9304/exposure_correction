#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    24-Jul-2023 18:42:51

import tensorflow as tf
import sys     # Remove this line after completing the layer definition.

class extractPyrLayer(tf.keras.layers.Layer):
    # Add any additional layer hyperparameters to the constructor's
    # argument list below.
    def __init__(self, name=None, Plevel=None):
        super(extractPyrLayer, self).__init__(name=name)
        self.Plevel = Plevel
        self.Description = "extract lap pyr layer"

    def call(self, input1):
        # Add code to implement the layer's forward pass here.
        # The input tensor format(s) are: BSSC
        # The output tensor format(s) are: BSSC
        # where B=batch, C=channels, T=time, S=spatial(in order of height, width, depth,...)

        sz = tf.shape(input1)
        if len(input1.shape) == 4:
            L = sz[3]//4
            output1 = input1[:, 0:int(sz[1]/2**(self.Plevel-1)), 0:int(sz[2]/2**(self.Plevel-1)), \
                    L*(self.Plevel-1):L*(self.Plevel-1)+L]
        else:
            L = sz[2]//4
            output1 = input1[0:int(sz[0]/2**(self.Plevel-1)), 0:int(sz[1]/2**(self.Plevel-1)), \
                    L*(self.Plevel-1):L*(self.Plevel-1)+L]

        # # Remove the following 3 lines after completing the custom layer definition:
        # print("Warning: load_model(): Before you can load the model, you must complete the definition of custom layer extractPyrLayer in the customLayers folder.")
        # print("Exiting...")
        # sys.exit("See the warning message above.")

        return output1
