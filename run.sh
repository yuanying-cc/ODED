#!bin/bash

python generate_data_for_SDN_train.py

python train_SDN.py

python load_SDN_model_pred_data.py

python imitation_learning_on_pred_data.py
