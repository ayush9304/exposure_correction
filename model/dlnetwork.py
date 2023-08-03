#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    24-Jul-2023 18:42:51

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.customLayers.extractPyrLayer import extractPyrLayer
from model.customLayers.packingGLayer import packingGLayer

def create_model():
    InputLayer = keras.Input(shape=(512,512,12))
    # extractPyrLayer_level_1 =  extractPyrLayer('level_1_extract_pyr',1);
    # extractPyrLayer_level_2 =  extractPyrLayer('level_2_extract_pyr',2);
    # extractPyrLayer_level_3 =  extractPyrLayer('level_3_extract_pyr',3);
    # extractPyrLayer_level_4 =  extractPyrLayer('level_4_extract_pyr',4);
    level_1_extract_pyr = extractPyrLayer('level_1_extract_pyr',1)(InputLayer)
    level_2_extract_pyr = extractPyrLayer('level_2_extract_pyr',2)(InputLayer)
    level_3_extract_pyr = extractPyrLayer('level_3_extract_pyr',3)(InputLayer)
    level_4_extract_pyr = extractPyrLayer('level_4_extract_pyr',4)(InputLayer)
    level_4_Encoder_Stage_1_Conv_1 = layers.Conv2D(24, (3,3), padding="same", name="level_4_Encoder_Stage_1_Conv_1_")(level_4_extract_pyr)
    level_4_Encoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_1_Conv_1)
    level_4_Encoder_Stage_1_Conv_2 = layers.Conv2D(24, (3,3), padding="same", name="level_4_Encoder_Stage_1_Conv_2_")(level_4_Encoder_Stage_1_L_ReLU_1)
    level_4_Encoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_1_Conv_2)
    level_4_Encoder_Stage_1_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_4_Encoder_Stage_1_L_ReLU_2)
    level_4_Encoder_Stage_2_Conv_1 = layers.Conv2D(48, (3,3), padding="same", name="level_4_Encoder_Stage_2_Conv_1_")(level_4_Encoder_Stage_1_MaxPool)
    level_4_Encoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_2_Conv_1)
    level_4_Encoder_Stage_2_Conv_2 = layers.Conv2D(48, (3,3), padding="same", name="level_4_Encoder_Stage_2_Conv_2_")(level_4_Encoder_Stage_2_L_ReLU_1)
    level_4_Encoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_2_Conv_2)
    level_4_Encoder_Stage_2_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_4_Encoder_Stage_2_L_ReLU_2)
    level_4_Encoder_Stage_3_Conv_1 = layers.Conv2D(96, (3,3), padding="same", name="level_4_Encoder_Stage_3_Conv_1_")(level_4_Encoder_Stage_2_MaxPool)
    level_4_Encoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_3_Conv_1)
    level_4_Encoder_Stage_3_Conv_2 = layers.Conv2D(96, (3,3), padding="same", name="level_4_Encoder_Stage_3_Conv_2_")(level_4_Encoder_Stage_3_L_ReLU_1)
    level_4_Encoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_3_Conv_2)
    level_4_Encoder_Stage_3_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_4_Encoder_Stage_3_L_ReLU_2)
    level_4_Encoder_Stage_4_Conv_1 = layers.Conv2D(192, (3,3), padding="same", name="level_4_Encoder_Stage_4_Conv_1_")(level_4_Encoder_Stage_3_MaxPool)
    level_4_Encoder_Stage_4_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_4_Conv_1)
    level_4_Encoder_Stage_4_Conv_2 = layers.Conv2D(192, (3,3), padding="same", name="level_4_Encoder_Stage_4_Conv_2_")(level_4_Encoder_Stage_4_L_ReLU_1)
    level_4_Encoder_Stage_4_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Encoder_Stage_4_Conv_2)
    level_4_Encoder_Stage_4_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_4_Encoder_Stage_4_L_ReLU_2)
    level_4_Bridge_Conv_1 = layers.Conv2D(384, (3,3), padding="same", name="level_4_Bridge_Conv_1_")(level_4_Encoder_Stage_4_MaxPool)
    level_4_Bridge_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Bridge_Conv_1)
    level_4_Bridge_Conv_2 = layers.Conv2D(384, (3,3), padding="same", name="level_4_Bridge_Conv_2_")(level_4_Bridge_L_ReLU_1)
    level_4_Bridge_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Bridge_Conv_2)
    level_4_Decoder_Stage_1_UpConv = layers.Conv2DTranspose(192, (2,2), strides=(2,2), name="level_4_Decoder_Stage_1_UpConv_")(level_4_Bridge_L_ReLU_2)
    level_4_Decoder_Stage_1_UpReLU = layers.ReLU()(level_4_Decoder_Stage_1_UpConv)
    level_4_Decoder_Stage_1_DepthConcatenation = layers.Concatenate(axis=-1)([level_4_Decoder_Stage_1_UpReLU, level_4_Encoder_Stage_4_L_ReLU_2])
    level_4_Decoder_Stage_1_Conv_1 = layers.Conv2D(192, (3,3), padding="same", name="level_4_Decoder_Stage_1_Conv_1_")(level_4_Decoder_Stage_1_DepthConcatenation)
    level_4_Decoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_1_Conv_1)
    level_4_Decoder_Stage_1_Conv_2 = layers.Conv2D(192, (3,3), padding="same", name="level_4_Decoder_Stage_1_Conv_2_")(level_4_Decoder_Stage_1_L_ReLU_1)
    level_4_Decoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_1_Conv_2)
    level_4_Decoder_Stage_2_UpConv = layers.Conv2DTranspose(96, (2,2), strides=(2,2), name="level_4_Decoder_Stage_2_UpConv_")(level_4_Decoder_Stage_1_L_ReLU_2)
    level_4_Decoder_Stage_2_UpReLU = layers.ReLU()(level_4_Decoder_Stage_2_UpConv)
    level_4_Decoder_Stage_2_DepthConcatenation = layers.Concatenate(axis=-1)([level_4_Decoder_Stage_2_UpReLU, level_4_Encoder_Stage_3_L_ReLU_2])
    level_4_Decoder_Stage_2_Conv_1 = layers.Conv2D(96, (3,3), padding="same", name="level_4_Decoder_Stage_2_Conv_1_")(level_4_Decoder_Stage_2_DepthConcatenation)
    level_4_Decoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_2_Conv_1)
    level_4_Decoder_Stage_2_Conv_2 = layers.Conv2D(96, (3,3), padding="same", name="level_4_Decoder_Stage_2_Conv_2_")(level_4_Decoder_Stage_2_L_ReLU_1)
    level_4_Decoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_2_Conv_2)
    level_4_Decoder_Stage_3_UpConv = layers.Conv2DTranspose(48, (2,2), strides=(2,2), name="level_4_Decoder_Stage_3_UpConv_")(level_4_Decoder_Stage_2_L_ReLU_2)
    level_4_Decoder_Stage_3_UpReLU = layers.ReLU()(level_4_Decoder_Stage_3_UpConv)
    level_4_Decoder_Stage_3_DepthConcatenation = layers.Concatenate(axis=-1)([level_4_Decoder_Stage_3_UpReLU, level_4_Encoder_Stage_2_L_ReLU_2])
    level_4_Decoder_Stage_3_Conv_1 = layers.Conv2D(48, (3,3), padding="same", name="level_4_Decoder_Stage_3_Conv_1_")(level_4_Decoder_Stage_3_DepthConcatenation)
    level_4_Decoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_3_Conv_1)
    level_4_Decoder_Stage_3_Conv_2 = layers.Conv2D(48, (3,3), padding="same", name="level_4_Decoder_Stage_3_Conv_2_")(level_4_Decoder_Stage_3_L_ReLU_1)
    level_4_Decoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_3_Conv_2)
    level_4_Decoder_Stage_4_UpConv = layers.Conv2DTranspose(24, (2,2), strides=(2,2), name="level_4_Decoder_Stage_4_UpConv_")(level_4_Decoder_Stage_3_L_ReLU_2)
    level_4_Decoder_Stage_4_UpReLU = layers.ReLU()(level_4_Decoder_Stage_4_UpConv)
    level_4_Decoder_Stage_4_DepthConcatenation = layers.Concatenate(axis=-1)([level_4_Decoder_Stage_4_UpReLU, level_4_Encoder_Stage_1_L_ReLU_2])
    level_4_Decoder_Stage_4_Conv_1 = layers.Conv2D(24, (3,3), padding="same", name="level_4_Decoder_Stage_4_Conv_1_")(level_4_Decoder_Stage_4_DepthConcatenation)
    level_4_Decoder_Stage_4_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_4_Conv_1)
    level_4_Decoder_Stage_4_Conv_2 = layers.Conv2D(24, (3,3), padding="same", name="level_4_Decoder_Stage_4_Conv_2_")(level_4_Decoder_Stage_4_L_ReLU_1)
    level_4_Decoder_Stage_4_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_4_Decoder_Stage_4_Conv_2)
    level_4_Final_ConvolutionLayer = layers.Conv2D(3, (1,1), padding="same", name="level_4_Final_ConvolutionLayer_")(level_4_Decoder_Stage_4_L_ReLU_2)
    level_4_upsampling = layers.Conv2DTranspose(3, (2,2), strides=(2,2), name="level_4_upsampling_")(level_4_Final_ConvolutionLayer)
    out_L_4_in_L_3 = layers.Add()([level_4_upsampling, level_3_extract_pyr])
    level_3_Encoder_Stage_1_Conv_1 = layers.Conv2D(24, (3,3), padding="same", name="level_3_Encoder_Stage_1_Conv_1_")(out_L_4_in_L_3)
    level_3_Encoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_3_Encoder_Stage_1_Conv_1)
    level_3_Encoder_Stage_1_Conv_2 = layers.Conv2D(24, (3,3), padding="same", name="level_3_Encoder_Stage_1_Conv_2_")(level_3_Encoder_Stage_1_L_ReLU_1)
    level_3_Encoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_3_Encoder_Stage_1_Conv_2)
    level_3_Encoder_Stage_1_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_3_Encoder_Stage_1_L_ReLU_2)
    level_3_Encoder_Stage_2_Conv_1 = layers.Conv2D(48, (3,3), padding="same", name="level_3_Encoder_Stage_2_Conv_1_")(level_3_Encoder_Stage_1_MaxPool)
    level_3_Encoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_3_Encoder_Stage_2_Conv_1)
    level_3_Encoder_Stage_2_Conv_2 = layers.Conv2D(48, (3,3), padding="same", name="level_3_Encoder_Stage_2_Conv_2_")(level_3_Encoder_Stage_2_L_ReLU_1)
    level_3_Encoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_3_Encoder_Stage_2_Conv_2)
    level_3_Encoder_Stage_2_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_3_Encoder_Stage_2_L_ReLU_2)
    level_3_Encoder_Stage_3_Conv_1 = layers.Conv2D(96, (3,3), padding="same", name="level_3_Encoder_Stage_3_Conv_1_")(level_3_Encoder_Stage_2_MaxPool)
    level_3_Encoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_3_Encoder_Stage_3_Conv_1)
    level_3_Encoder_Stage_3_Conv_2 = layers.Conv2D(96, (3,3), padding="same", name="level_3_Encoder_Stage_3_Conv_2_")(level_3_Encoder_Stage_3_L_ReLU_1)
    level_3_Encoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_3_Encoder_Stage_3_Conv_2)
    level_3_Encoder_Stage_3_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_3_Encoder_Stage_3_L_ReLU_2)
    level_3_Bridge_Conv_1 = layers.Conv2D(192, (3,3), padding="same", name="level_3_Bridge_Conv_1_")(level_3_Encoder_Stage_3_MaxPool)
    level_3_Bridge_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_3_Bridge_Conv_1)
    level_3_Bridge_Conv_2 = layers.Conv2D(192, (3,3), padding="same", name="level_3_Bridge_Conv_2_")(level_3_Bridge_L_ReLU_1)
    level_3_Bridge_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_3_Bridge_Conv_2)
    level_3_Decoder_Stage_1_UpConv = layers.Conv2DTranspose(96, (2,2), strides=(2,2), name="level_3_Decoder_Stage_1_UpConv_")(level_3_Bridge_L_ReLU_2)
    level_3_Decoder_Stage_1_UpReLU = layers.ReLU()(level_3_Decoder_Stage_1_UpConv)
    level_3_Decoder_Stage_1_DepthConcatenation = layers.Concatenate(axis=-1)([level_3_Decoder_Stage_1_UpReLU, level_3_Encoder_Stage_3_L_ReLU_2])
    level_3_Decoder_Stage_1_Conv_1 = layers.Conv2D(96, (3,3), padding="same", name="level_3_Decoder_Stage_1_Conv_1_")(level_3_Decoder_Stage_1_DepthConcatenation)
    level_3_Decoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_3_Decoder_Stage_1_Conv_1)
    level_3_Decoder_Stage_1_Conv_2 = layers.Conv2D(96, (3,3), padding="same", name="level_3_Decoder_Stage_1_Conv_2_")(level_3_Decoder_Stage_1_L_ReLU_1)
    level_3_Decoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_3_Decoder_Stage_1_Conv_2)
    level_3_Decoder_Stage_2_UpConv = layers.Conv2DTranspose(48, (2,2), strides=(2,2), name="level_3_Decoder_Stage_2_UpConv_")(level_3_Decoder_Stage_1_L_ReLU_2)
    level_3_Decoder_Stage_2_UpReLU = layers.ReLU()(level_3_Decoder_Stage_2_UpConv)
    level_3_Decoder_Stage_2_DepthConcatenation = layers.Concatenate(axis=-1)([level_3_Decoder_Stage_2_UpReLU, level_3_Encoder_Stage_2_L_ReLU_2])
    level_3_Decoder_Stage_2_Conv_1 = layers.Conv2D(48, (3,3), padding="same", name="level_3_Decoder_Stage_2_Conv_1_")(level_3_Decoder_Stage_2_DepthConcatenation)
    level_3_Decoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_3_Decoder_Stage_2_Conv_1)
    level_3_Decoder_Stage_2_Conv_2 = layers.Conv2D(48, (3,3), padding="same", name="level_3_Decoder_Stage_2_Conv_2_")(level_3_Decoder_Stage_2_L_ReLU_1)
    level_3_Decoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_3_Decoder_Stage_2_Conv_2)
    level_3_Decoder_Stage_3_UpConv = layers.Conv2DTranspose(24, (2,2), strides=(2,2), name="level_3_Decoder_Stage_3_UpConv_")(level_3_Decoder_Stage_2_L_ReLU_2)
    level_3_Decoder_Stage_3_UpReLU = layers.ReLU()(level_3_Decoder_Stage_3_UpConv)
    level_3_Decoder_Stage_3_DepthConcatenation = layers.Concatenate(axis=-1)([level_3_Decoder_Stage_3_UpReLU, level_3_Encoder_Stage_1_L_ReLU_2])
    level_3_Decoder_Stage_3_Conv_1 = layers.Conv2D(24, (3,3), padding="same", name="level_3_Decoder_Stage_3_Conv_1_")(level_3_Decoder_Stage_3_DepthConcatenation)
    level_3_Decoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_3_Decoder_Stage_3_Conv_1)
    level_3_Decoder_Stage_3_Conv_2 = layers.Conv2D(24, (3,3), padding="same", name="level_3_Decoder_Stage_3_Conv_2_")(level_3_Decoder_Stage_3_L_ReLU_1)
    level_3_Decoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_3_Decoder_Stage_3_Conv_2)
    level_3_Final_ConvolutionLayer = layers.Conv2D(3, (1,1), padding="same", name="level_3_Final_ConvolutionLayer_")(level_3_Decoder_Stage_3_L_ReLU_2)
    level_3_reconstructLayer = layers.Add()([level_3_Final_ConvolutionLayer, level_4_upsampling])
    level_3_upsampling = layers.Conv2DTranspose(3, (2,2), strides=(2,2), name="level_3_upsampling_")(level_3_reconstructLayer)
    out_L_3_in_L_2 = layers.Add()([level_3_upsampling, level_2_extract_pyr])
    level_2_Encoder_Stage_1_Conv_1 = layers.Conv2D(24, (3,3), padding="same", name="level_2_Encoder_Stage_1_Conv_1_")(out_L_3_in_L_2)
    level_2_Encoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_2_Encoder_Stage_1_Conv_1)
    level_2_Encoder_Stage_1_Conv_2 = layers.Conv2D(24, (3,3), padding="same", name="level_2_Encoder_Stage_1_Conv_2_")(level_2_Encoder_Stage_1_L_ReLU_1)
    level_2_Encoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_2_Encoder_Stage_1_Conv_2)
    level_2_Encoder_Stage_1_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_2_Encoder_Stage_1_L_ReLU_2)
    level_2_Encoder_Stage_2_Conv_1 = layers.Conv2D(48, (3,3), padding="same", name="level_2_Encoder_Stage_2_Conv_1_")(level_2_Encoder_Stage_1_MaxPool)
    level_2_Encoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_2_Encoder_Stage_2_Conv_1)
    level_2_Encoder_Stage_2_Conv_2 = layers.Conv2D(48, (3,3), padding="same", name="level_2_Encoder_Stage_2_Conv_2_")(level_2_Encoder_Stage_2_L_ReLU_1)
    level_2_Encoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_2_Encoder_Stage_2_Conv_2)
    level_2_Encoder_Stage_2_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_2_Encoder_Stage_2_L_ReLU_2)
    level_2_Encoder_Stage_3_Conv_1 = layers.Conv2D(96, (3,3), padding="same", name="level_2_Encoder_Stage_3_Conv_1_")(level_2_Encoder_Stage_2_MaxPool)
    level_2_Encoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_2_Encoder_Stage_3_Conv_1)
    level_2_Encoder_Stage_3_Conv_2 = layers.Conv2D(96, (3,3), padding="same", name="level_2_Encoder_Stage_3_Conv_2_")(level_2_Encoder_Stage_3_L_ReLU_1)
    level_2_Encoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_2_Encoder_Stage_3_Conv_2)
    level_2_Encoder_Stage_3_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_2_Encoder_Stage_3_L_ReLU_2)
    level_2_Bridge_Conv_1 = layers.Conv2D(192, (3,3), padding="same", name="level_2_Bridge_Conv_1_")(level_2_Encoder_Stage_3_MaxPool)
    level_2_Bridge_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_2_Bridge_Conv_1)
    level_2_Bridge_Conv_2 = layers.Conv2D(192, (3,3), padding="same", name="level_2_Bridge_Conv_2_")(level_2_Bridge_L_ReLU_1)
    level_2_Bridge_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_2_Bridge_Conv_2)
    level_2_Decoder_Stage_1_UpConv = layers.Conv2DTranspose(96, (2,2), strides=(2,2), name="level_2_Decoder_Stage_1_UpConv_")(level_2_Bridge_L_ReLU_2)
    level_2_Decoder_Stage_1_UpReLU = layers.ReLU()(level_2_Decoder_Stage_1_UpConv)
    level_2_Decoder_Stage_1_DepthConcatenation = layers.Concatenate(axis=-1)([level_2_Decoder_Stage_1_UpReLU, level_2_Encoder_Stage_3_L_ReLU_2])
    level_2_Decoder_Stage_1_Conv_1 = layers.Conv2D(96, (3,3), padding="same", name="level_2_Decoder_Stage_1_Conv_1_")(level_2_Decoder_Stage_1_DepthConcatenation)
    level_2_Decoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_2_Decoder_Stage_1_Conv_1)
    level_2_Decoder_Stage_1_Conv_2 = layers.Conv2D(96, (3,3), padding="same", name="level_2_Decoder_Stage_1_Conv_2_")(level_2_Decoder_Stage_1_L_ReLU_1)
    level_2_Decoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_2_Decoder_Stage_1_Conv_2)
    level_2_Decoder_Stage_2_UpConv = layers.Conv2DTranspose(48, (2,2), strides=(2,2), name="level_2_Decoder_Stage_2_UpConv_")(level_2_Decoder_Stage_1_L_ReLU_2)
    level_2_Decoder_Stage_2_UpReLU = layers.ReLU()(level_2_Decoder_Stage_2_UpConv)
    level_2_Decoder_Stage_2_DepthConcatenation = layers.Concatenate(axis=-1)([level_2_Decoder_Stage_2_UpReLU, level_2_Encoder_Stage_2_L_ReLU_2])
    level_2_Decoder_Stage_2_Conv_1 = layers.Conv2D(48, (3,3), padding="same", name="level_2_Decoder_Stage_2_Conv_1_")(level_2_Decoder_Stage_2_DepthConcatenation)
    level_2_Decoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_2_Decoder_Stage_2_Conv_1)
    level_2_Decoder_Stage_2_Conv_2 = layers.Conv2D(48, (3,3), padding="same", name="level_2_Decoder_Stage_2_Conv_2_")(level_2_Decoder_Stage_2_L_ReLU_1)
    level_2_Decoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_2_Decoder_Stage_2_Conv_2)
    level_2_Decoder_Stage_3_UpConv = layers.Conv2DTranspose(24, (2,2), strides=(2,2), name="level_2_Decoder_Stage_3_UpConv_")(level_2_Decoder_Stage_2_L_ReLU_2)
    level_2_Decoder_Stage_3_UpReLU = layers.ReLU()(level_2_Decoder_Stage_3_UpConv)
    level_2_Decoder_Stage_3_DepthConcatenation = layers.Concatenate(axis=-1)([level_2_Decoder_Stage_3_UpReLU, level_2_Encoder_Stage_1_L_ReLU_2])
    level_2_Decoder_Stage_3_Conv_1 = layers.Conv2D(24, (3,3), padding="same", name="level_2_Decoder_Stage_3_Conv_1_")(level_2_Decoder_Stage_3_DepthConcatenation)
    level_2_Decoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_2_Decoder_Stage_3_Conv_1)
    level_2_Decoder_Stage_3_Conv_2 = layers.Conv2D(24, (3,3), padding="same", name="level_2_Decoder_Stage_3_Conv_2_")(level_2_Decoder_Stage_3_L_ReLU_1)
    level_2_Decoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_2_Decoder_Stage_3_Conv_2)
    level_2_Final_ConvolutionLayer = layers.Conv2D(3, (1,1), padding="same", name="level_2_Final_ConvolutionLayer_")(level_2_Decoder_Stage_3_L_ReLU_2)
    level_2_reconstructLayer = layers.Add()([level_2_Final_ConvolutionLayer, level_3_upsampling])
    level_2_upsampling = layers.Conv2DTranspose(3, (2,2), strides=(2,2), name="level_2_upsampling_")(level_2_reconstructLayer)
    out_L_2_in_L_1 = layers.Add()([level_2_upsampling, level_1_extract_pyr])
    level_1_Encoder_Stage_1_Conv_1 = layers.Conv2D(16, (3,3), padding="same", name="level_1_Encoder_Stage_1_Conv_1_")(out_L_2_in_L_1)
    level_1_Encoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_1_Encoder_Stage_1_Conv_1)
    level_1_Encoder_Stage_1_Conv_2 = layers.Conv2D(16, (3,3), padding="same", name="level_1_Encoder_Stage_1_Conv_2_")(level_1_Encoder_Stage_1_L_ReLU_1)
    level_1_Encoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_1_Encoder_Stage_1_Conv_2)
    level_1_Encoder_Stage_1_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_1_Encoder_Stage_1_L_ReLU_2)
    level_1_Encoder_Stage_2_Conv_1 = layers.Conv2D(32, (3,3), padding="same", name="level_1_Encoder_Stage_2_Conv_1_")(level_1_Encoder_Stage_1_MaxPool)
    level_1_Encoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_1_Encoder_Stage_2_Conv_1)
    level_1_Encoder_Stage_2_Conv_2 = layers.Conv2D(32, (3,3), padding="same", name="level_1_Encoder_Stage_2_Conv_2_")(level_1_Encoder_Stage_2_L_ReLU_1)
    level_1_Encoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_1_Encoder_Stage_2_Conv_2)
    level_1_Encoder_Stage_2_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_1_Encoder_Stage_2_L_ReLU_2)
    level_1_Encoder_Stage_3_Conv_1 = layers.Conv2D(64, (3,3), padding="same", name="level_1_Encoder_Stage_3_Conv_1_")(level_1_Encoder_Stage_2_MaxPool)
    level_1_Encoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_1_Encoder_Stage_3_Conv_1)
    level_1_Encoder_Stage_3_Conv_2 = layers.Conv2D(64, (3,3), padding="same", name="level_1_Encoder_Stage_3_Conv_2_")(level_1_Encoder_Stage_3_L_ReLU_1)
    level_1_Encoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_1_Encoder_Stage_3_Conv_2)
    level_1_Encoder_Stage_3_MaxPool = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(level_1_Encoder_Stage_3_L_ReLU_2)
    level_1_Bridge_Conv_1 = layers.Conv2D(128, (3,3), padding="same", name="level_1_Bridge_Conv_1_")(level_1_Encoder_Stage_3_MaxPool)
    level_1_Bridge_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_1_Bridge_Conv_1)
    level_1_Bridge_Conv_2 = layers.Conv2D(128, (3,3), padding="same", name="level_1_Bridge_Conv_2_")(level_1_Bridge_L_ReLU_1)
    level_1_Bridge_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_1_Bridge_Conv_2)
    level_1_Decoder_Stage_1_UpConv = layers.Conv2DTranspose(64, (2,2), strides=(2,2), name="level_1_Decoder_Stage_1_UpConv_")(level_1_Bridge_L_ReLU_2)
    level_1_Decoder_Stage_1_UpReLU = layers.ReLU()(level_1_Decoder_Stage_1_UpConv)
    level_1_Decoder_Stage_1_DepthConcatenation = layers.Concatenate(axis=-1)([level_1_Decoder_Stage_1_UpReLU, level_1_Encoder_Stage_3_L_ReLU_2])
    level_1_Decoder_Stage_1_Conv_1 = layers.Conv2D(64, (3,3), padding="same", name="level_1_Decoder_Stage_1_Conv_1_")(level_1_Decoder_Stage_1_DepthConcatenation)
    level_1_Decoder_Stage_1_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_1_Decoder_Stage_1_Conv_1)
    level_1_Decoder_Stage_1_Conv_2 = layers.Conv2D(64, (3,3), padding="same", name="level_1_Decoder_Stage_1_Conv_2_")(level_1_Decoder_Stage_1_L_ReLU_1)
    level_1_Decoder_Stage_1_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_1_Decoder_Stage_1_Conv_2)
    level_1_Decoder_Stage_2_UpConv = layers.Conv2DTranspose(32, (2,2), strides=(2,2), name="level_1_Decoder_Stage_2_UpConv_")(level_1_Decoder_Stage_1_L_ReLU_2)
    level_1_Decoder_Stage_2_UpReLU = layers.ReLU()(level_1_Decoder_Stage_2_UpConv)
    level_1_Decoder_Stage_2_DepthConcatenation = layers.Concatenate(axis=-1)([level_1_Decoder_Stage_2_UpReLU, level_1_Encoder_Stage_2_L_ReLU_2])
    level_1_Decoder_Stage_2_Conv_1 = layers.Conv2D(32, (3,3), padding="same", name="level_1_Decoder_Stage_2_Conv_1_")(level_1_Decoder_Stage_2_DepthConcatenation)
    level_1_Decoder_Stage_2_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_1_Decoder_Stage_2_Conv_1)
    level_1_Decoder_Stage_2_Conv_2 = layers.Conv2D(32, (3,3), padding="same", name="level_1_Decoder_Stage_2_Conv_2_")(level_1_Decoder_Stage_2_L_ReLU_1)
    level_1_Decoder_Stage_2_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_1_Decoder_Stage_2_Conv_2)
    level_1_Decoder_Stage_3_UpConv = layers.Conv2DTranspose(16, (2,2), strides=(2,2), name="level_1_Decoder_Stage_3_UpConv_")(level_1_Decoder_Stage_2_L_ReLU_2)
    level_1_Decoder_Stage_3_UpReLU = layers.ReLU()(level_1_Decoder_Stage_3_UpConv)
    level_1_Decoder_Stage_3_DepthConcatenation = layers.Concatenate(axis=-1)([level_1_Decoder_Stage_3_UpReLU, level_1_Encoder_Stage_1_L_ReLU_2])
    level_1_Decoder_Stage_3_Conv_1 = layers.Conv2D(16, (3,3), padding="same", name="level_1_Decoder_Stage_3_Conv_1_")(level_1_Decoder_Stage_3_DepthConcatenation)
    level_1_Decoder_Stage_3_L_ReLU_1 = layers.LeakyReLU(alpha=0.200000)(level_1_Decoder_Stage_3_Conv_1)
    level_1_Decoder_Stage_3_Conv_2 = layers.Conv2D(16, (3,3), padding="same", name="level_1_Decoder_Stage_3_Conv_2_")(level_1_Decoder_Stage_3_L_ReLU_1)
    level_1_Decoder_Stage_3_L_ReLU_2 = layers.LeakyReLU(alpha=0.200000)(level_1_Decoder_Stage_3_Conv_2)
    level_1_Final_ConvolutionLayer = layers.Conv2D(3, (1,1), padding="same", name="level_1_Final_ConvolutionLayer_")(level_1_Decoder_Stage_3_L_ReLU_2)
    level_1_reconstructLayer = layers.Add()([level_1_Final_ConvolutionLayer, level_2_upsampling])
    packingLayer = packingGLayer()(level_1_reconstructLayer, level_2_upsampling, level_3_upsampling, level_4_upsampling)

    model = keras.Model(inputs=[InputLayer], outputs=[packingLayer])
    return model
