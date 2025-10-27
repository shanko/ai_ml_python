# https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
import argparse
from datetime import datetime

start_time = datetime.now()
print(f"Script started at: {start_time}")

from autogluon.tabular import TabularDataset, TabularPredictor

# Parse command line arguments
parser = argparse.ArgumentParser(description="AutoGluon training and prediction script")
parser.add_argument(
    "--train_data_csv",
    type=str,
    default="dt_train_3000.csv",
    help="Path to training CSV file (default: dt_train_3000.csv)",
)
parser.add_argument(
    "--test_data_csv",
    type=str,
    default="dt_test_380.csv",
    help="Path to test CSV file (default: dt_test_380.csv)",
)
args = parser.parse_args()

# data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
data_url = "/home/shanko/code/lang/py/"

train_data = TabularDataset(f"{data_url}{args.train_data_csv}")

print(
    f"\n======== Train Data using {args.train_data_csv} at {datetime.now()} ============\n"
)
print(train_data.head())

label = "state"
train_data[label].describe()
predictor = TabularPredictor(
    label=label, eval_metric="accuracy", problem_type="multiclass"
).fit(
    train_data,
    presets="best_quality",
    time_limit=7200,
    num_bag_folds=10,
    num_stack_levels=3,
    auto_stack=True,
    excluded_model_types=["FASTTEXT"],
)
"""
    hyperparameter_tune_kwargs={
        "num_trials": 100,
        "scheduler": "local",
        "searcher": "auto",
    },
    hyperparameters={
        "GBM": {
            "num_boost_round": 2000,
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 6, 9],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
        "CAT": {
            "iterations": 2000,
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8, 10],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
        },
        "XGB": {
            "n_estimators": 2000,
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 6, 9],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0.1, 1, 2],
        },
        "RF": {
            "n_estimators": 500,
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "XT": {
            "n_estimators": 500,
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "LR": {},
        "KNN": {"weights": ["uniform", "distance"], "n_neighbors": [3, 5, 7, 9]},
        "NN_TORCH": {
            "num_epochs": 200,
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "weight_decay": [1e-6, 1e-4, 1e-2],
            "dropout_prob": [0.0, 0.1, 0.2, 0.3],
        },
    },
)
"""


test_data = TabularDataset(f"{data_url}{args.test_data_csv}")

y_pred = predictor.predict(test_data.drop(columns=[label]))
print(
    f"\n======== Mismatched Predictions using {args.test_data_csv} at {datetime.now()} ============\n"
)
mismatch_count = 0
count = 0
print("i,id,state,prediction")
for i, (index, row) in enumerate(test_data.iterrows()):
    if row["state"] != y_pred.iloc[i]:
        print(f"{i},{row['id']},{row['state']},{y_pred.iloc[i]}")
        mismatch_count += 1
    count += 1

print(
    f"\nTotal mismatches: {mismatch_count} out of {count}, which is {(((mismatch_count * 1.0) / count) * 100.0):.2f} %"
)

print(predictor.evaluate(test_data, silent=True))
print(f"\n======== LeaderBoard {datetime.now()} ============\n")

leaderboard = predictor.leaderboard(test_data)
# Print column headers
print(
    f"{'Model':<20} {'Score':<10} {'Pred_time':<12} {'Fit_time':<12} {'Stack_level':<12}"
)
print("-" * 76)
# Print each row in tabular format
for index, row in leaderboard.iterrows():
    model = str(row.name)[:19]  # Truncate long model names
    score = (
        f"{row.get('score_val', 'N/A'):<10.4f}"
        if isinstance(row.get("score_val"), (int, float))
        else f"{'N/A':<10}"
    )
    pred_time = (
        f"{row.get('pred_time_val', 'N/A'):<12.4f}"
        if isinstance(row.get("pred_time_val"), (int, float))
        else f"{'N/A':<12}"
    )
    fit_time = (
        f"{row.get('fit_time', 'N/A'):<12.2f}"
        if isinstance(row.get("fit_time"), (int, float))
        else f"{'N/A':<12}"
    )
    stack_level = f"{row.get('stack_level', 'N/A'):<12}"
    print(f"{model:<20} {score} {pred_time} {fit_time} {stack_level}")

end_time = datetime.now()
print(f"Script completed at: {end_time}")
elapsed_seconds = (end_time - start_time).total_seconds()
print(f"Total elapsed time: {elapsed_seconds:.2f} seconds")
