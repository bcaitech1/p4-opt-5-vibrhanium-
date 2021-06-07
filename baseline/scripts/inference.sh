python inference.py --dst /opt/ml/output \
                    --weight /opt/ml/input/config/optuna_model/0606_1718/116/best.pt \
                    --model_config /opt/ml/input/config/optuna_model/0606_1718/116/0606_1718_116_model.yaml \
                    --data_config configs/data/taco.yaml \
                    --hyperparam /opt/ml/input/config/optuna_model/0606_1718/116/0606_1718_116_hyperparameter.yaml \
                    --img_root /opt/ml/input/data/test/ \
                    --decompose False