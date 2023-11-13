import json
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--epochs", type=int, help="Number of epochs to train for")
argParser.add_argument("--hiddenLayerShapes", type=int, nargs="+", help="Hidden Layer Shapes")
argParser.add_argument("--latentDimension", type=int, help="The latent dimensions")
argParser.add_argument("--encoderActivation", type=str, help="Encoder Activation Function. `sigmoid` or `tanh`")
argParser.add_argument("--targetClasses", type=int, nargs="+", help="The labels of the target classes. All others are filtered.")
argParser.add_argument("--encoderLossWeighting", type=float, help="The weighting of the encoder loss.")
args = argParser.parse_args()

with open("settings.json", "w") as f:
    settingsDict = json.dump(vars(args), f)