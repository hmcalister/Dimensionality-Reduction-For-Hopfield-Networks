EPOCHS=10
HIDDEN_LAYER_SHAPES="128 64 64"
ENCODER_ACTIVATION="sigmoid"
TARGET_CLASSES="0 1"

for trialIndex in {0..100}; do
    for LATENT_DIMENSION in {16,64,128}; do
        for ENCODER_LOSS_WEIGHTING in $(seq 0 0.002 0.05); do
            printf "\33[2K\r"
            printf "TRIAL: $trialIndex\tLATENT DIMENSION: $LATENT_DIMENSION\tENCODER LOSS WEIGHTING: $ENCODER_LOSS_WEIGHTING\r"

            python constructSettings.py --epochs $EPOCHS \
                        --hiddenLayerShapes $HIDDEN_LAYER_SHAPES \
                        --latentDimension $LATENT_DIMENSION \
                        --encoderActivation $ENCODER_ACTIVATION \
                        --targetClasses $TARGET_CLASSES \
                        --encoderLossWeighting $ENCODER_LOSS_WEIGHTING

            python main.py &> main.log
            python aggregateHistory.py &> aggregate.log

        done
    done
done


