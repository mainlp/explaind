export PYTHONPATH="./"

# train model
python experiments/train_models/mnist10_model.py

# compute ExPLAIND scores (aggregated to data perspective) + validate
python experiments/validate_epk/mnist10_epk_prediction.py
python experiments/mnist_data_attribution/lds_attribution_baselines.py

# compute lds and loo ground truths
python experiments/mnist_data_attribution/create_lds_ground_truth.py

python experiments/mnist_data_attribution/eval_lds.py

python experiments/mnist_data_attribution/plot.py