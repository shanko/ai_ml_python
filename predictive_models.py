#  pip install scikit-learn pandas matplotlib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Class balance — Malignant: {(y==0).sum()}, Benign: {(y==1).sum()}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

for name, model in [("Random Forest", rf), ("Gradient Boosting", gb)]:
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.2%}")
    print(f"Test accuracy:  {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

import matplotlib.pyplot as plt
import numpy as np

# Top 10 features for Random Forest
indices = np.argsort(rf.feature_importances_)[-10:]
plt.barh(range(10), rf.feature_importances_[indices])
plt.yticks(range(10), [data.feature_names[i] for i in indices])
plt.title("Random Forest — Top 10 Feature Importances")
plt.tight_layout()
plt.show()
