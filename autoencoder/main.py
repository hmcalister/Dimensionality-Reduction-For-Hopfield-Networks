import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import argparse
import json
import pickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist

import hmcalisterHopfieldUtils
from autoEncoder import AutoEncoder, AutoEncoderSigmoidCustomLoss, AutoEncoderTanhCustomLoss

argParser = argparse.ArgumentParser()
argParser.add_argument("--modelSavePath", type=str, default="model.keras", help="Path to save/load model weights")
argParser.add_argument("--loadModel", action=argparse.BooleanOptionalAction, help="Bool to load model from modelSavePath")
argParser.add_argument("--settingsFile", type=str, default="settings.json", help="Path to settings file (JSON)")
args = argParser.parse_args()

with open(args.settingsFile) as f:
    settingsDict = json.load(f)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
dataShape = x_test.shape[1:]

targetTrainMask = np.isin(y_train, settingsDict["targetClasses"])
x_train = x_train[targetTrainMask,:,:]
y_train = y_train[targetTrainMask]
targetTestMask = np.isin(y_test, settingsDict["targetClasses"])
x_test = x_test[targetTestMask,:,:]
y_test = y_test[targetTestMask]

autoencoder = AutoEncoder(inputShape=dataShape, latentShape=settingsDict["latentDimension"], hiddenLayerShapes=settingsDict["hiddenLayerShapes"], encoderOutputActivationFunction=settingsDict["encoderActivation"])
autoencoder.summary()
if not args.loadModel:
    if settingsDict["encoderActivation"] == "sigmoid":
        encoderLoss = AutoEncoderSigmoidCustomLoss()
    elif settingsDict["encoderActivation"] == "tanh":
        encoderLoss = AutoEncoderTanhCustomLoss()
    else:
        print("ERR: encoder activation must be sigmoid or tanh")
        exit(1)

    autoencoder.compile(optimizer='adam', loss={
        "decoderOutput": "mse",
        "encoderOutput": encoderLoss
    }, loss_weights=[1, settingsDict["encoderLossWeighting"]])
    modelHistory = autoencoder.fit(x_train, x_train,
                    epochs=settingsDict["epochs"],
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=1)
    with open("modelHistory", "wb") as f:
        pickle.dump(modelHistory.history, f)
    autoencoder.save_weights(args.modelSavePath)
else:
    autoencoder.load_weights(args.modelSavePath)
