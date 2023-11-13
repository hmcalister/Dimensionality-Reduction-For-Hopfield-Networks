python constructSettings.py --epochs 10 \
            --hiddenLayerShapes 128 64 64 \
            --latentDimension 16 \
            --encoderActivation sigmoid \
            --targetClasses 0 1 \
            --encoderLossWeighting 0.01

python main.py

python aggregateHistory.py