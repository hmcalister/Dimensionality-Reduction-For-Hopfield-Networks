import pandas as pd
import numpy as np
import pickle
import json
import os


aggregateDataFrameFile = "modelHistoryAggregate.pq"
if not os.path.exists(aggregateDataFrameFile):
    newDF = pd.DataFrame(columns=["EncoderActivation", "EncoderLossWeighting", "TargetClasses", "LatentDimension", "Epochs",
                                  "TotalLoss", "DecoderLoss", "EncoderLoss",
                                  "TotalValidationLoss", "DecoderValidationLoss", "EncoderValidationLoss"])
    newDF.to_parquet(aggregateDataFrameFile)
aggregateDataFrame = pd.read_parquet(aggregateDataFrameFile)

with open("settings.json") as f:
    settingsDict = json.load(f)

with open("modelHistory", "rb") as f:
    modelHistory = pickle.load(f)

newRow = {
    "EncoderActivation": settingsDict["encoderActivation"],
    "EncoderLossWeighting": settingsDict["encoderLossWeighting"],
    "TargetClasses": str(settingsDict["targetClasses"]),
    "LatentDimension": settingsDict["latentDimension"],
    "Epochs": settingsDict["epochs"],
    "TotalLoss": modelHistory["loss"][-1],
    "DecoderLoss": modelHistory["decoderOutput_loss"][-1],
    "EncoderLoss": modelHistory["encoderOutput_loss"][-1],
    "TotalValidationLoss": modelHistory["val_loss"][-1],
    "DecoderValidationLoss": modelHistory["val_decoderOutput_loss"][-1],  
    "EncoderValidationLoss": modelHistory["val_encoderOutput_loss"][-1],
}

aggregateDataFrame = pd.concat([aggregateDataFrame, pd.DataFrame(newRow, index=[0])], ignore_index=True)
print(aggregateDataFrame)
aggregateDataFrame.to_parquet(aggregateDataFrameFile)
