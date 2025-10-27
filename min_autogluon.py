# https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html

from autogluon.tabular import TabularDataset, TabularPredictor

# data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
data_url = '/Users/shashankdate/code/lang/py/ai_ml_python/'
train_data = TabularDataset(f"{data_url}train_1000.csv")
print(train_data.head())

label = 'signature'
train_data[label].describe()
predictor = TabularPredictor(label=label).fit(train_data)

test_data = TabularDataset(f"{data_url}test_500.csv")

y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()

print(predictor.evaluate(test_data, silent=True))
leaderboard = predictor.leaderboard(test_data)

