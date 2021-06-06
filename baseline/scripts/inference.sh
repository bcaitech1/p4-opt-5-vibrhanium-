python inference.py --dst /opt/ml/output \
                    --weight /opt/ml/input/optuna_exp/0604_2353/70/best.pt \
                    --model_config /opt/ml/input/config/optuna_model/0604_2353/70/0604_2353_70_model.yaml \
                    --data_config configs/data/taco.yaml \
                    --hyperparam /opt/ml/input/config/optuna_model/0604_2353/70/0604_2353_70_hyperparameter.yaml \
                    --img_root /opt/ml/input/data/test/ \
                    --decompose False