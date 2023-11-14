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


# encodedVectors = autoencoder.encoderNetwork(x_test).numpy()
# decodedImages = autoencoder.decoderNetwork(encodedVectors).numpy()

# binaryHeaviside = hmcalisterHopfieldUtils.hopfield.binaryHeaviside
# bipolarHeaviside = hmcalisterHopfieldUtils.hopfield.bipolarHeaviside
# heavisideVectors = binaryHeaviside(encodedVectors-0.5)
# heavisideDecodedImages = autoencoder.decoderNetwork(heavisideVectors).numpy()

# # ---------- ORIGINAL AND RECONSTRUCTED IMAGES ----------
# n = 10
# imsize = 2
# fig = plt.figure(figsize=(n*imsize, 3*imsize))
# subfigs = fig.subfigures(nrows=3, ncols=1)

# for targetSubFig, imgs, title in zip(
#     subfigs,
#     [x_test, decodedImages, heavisideDecodedImages],
#     ["Original Images", "Directly Decoded Images", "Heaviside Decoded Images"]
# ):
#     targetSubFig.suptitle(title, y=0.98)
#     axes = targetSubFig.subplots(nrows=1, ncols=n)
#     for img, ax in zip(imgs, np.ravel(axes)):
#         ax.imshow(img, cmap="gray")
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

# # plt.tight_layout()
# plt.show()


# cmap = mpl.colormaps["tab10"]
# # ---------- AVERAGE ENCODED VEC BY CLASS ----------
# classOffset = 0.1
# for classLabel in settingsDict["targetClasses"]:
#     classIndicesMask = y_test==classLabel
#     classEncodedVecs = encodedVectors[classIndicesMask]
#     averageEncodedVec = np.average(classEncodedVecs, axis=0)
#     stdEncodedVec = np.std(classEncodedVecs, axis=0)
#     # plt.plot(averageEncodedVec, label=f"Class: {classLabel}")
#     plt.errorbar(np.arange(settingsDict["latentDimension"])+classLabel*classOffset, averageEncodedVec, stdEncodedVec, fmt="o", color=cmap(classLabel), ecolor=cmap(classLabel), label=f"Class: {classLabel}")
# plt.title(f"Average + Standard Deviation of Encoded Vector by Class")
# plt.legend()
# plt.tight_layout()
# plt.show()


# # ---------- ERRORBAR OF EACH CLASS ----------
# fig, axes = plt.subplots(len(settingsDict["targetClasses"]), 1)
# for classLabel, ax in zip(settingsDict["targetClasses"], np.ravel(axes)):
#     classIndicesMask = y_test==classLabel
#     classEncodedVecs = encodedVectors[classIndicesMask]
#     ax.errorbar(np.arange(settingsDict["latentDimension"]), np.average(classEncodedVecs, axis=0), np.std(classEncodedVecs), color=cmap(classLabel), ecolor="k")
#     ax.set_title(f"Average Encoding of Class {classLabel}")
# plt.tight_layout()
# plt.show()    


# # ---------- PCA OF ENCODED VEC ----------
# pca= PCA(n_components=16)
# pcaResults = pca.fit_transform(encodedVectors)
# pcaResults = pd.DataFrame({
#     "classLabels": y_test,
#     "pcaComponentOne": pcaResults[:,0],
#     "pcaComponentTwo": pcaResults[:,1],
# })
# print(f"PCA COMPONENT VARIANCES: {pca.explained_variance_ratio_}")
# print(f"PCA COMPONENT TOTAL VARIANCE: {np.sum(pca.explained_variance_ratio_)}")
# sns.scatterplot(pcaResults, x="pcaComponentOne", y="pcaComponentTwo", hue="classLabels", legend="full")
# plt.show()


# # ---------- TSNE OF ENCODED VEC ----------
# tsne= TSNE(n_components=2)
# tsneResults = tsne.fit_transform(encodedVectors)
# tsneResults = pd.DataFrame({
#     "classLabels": y_test,
#     "tsneComponentOne": tsneResults[:,0],
#     "tsneComponentTwo": tsneResults[:,1],
# })
# sns.scatterplot(tsneResults, x="tsneComponentOne", y="tsneComponentTwo", hue="classLabels", legend="full")
# plt.show()
