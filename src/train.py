import os
from credit_risk_detection import CreditRiskDetection
from preprocessing import Preprocessing
from model_evaluation import ModelEvaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import joblib

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

data_path = os.path.join(WORKDIR, "data", "credit_risk_dataset.csv")
model_path = os.path.join(WORKDIR, "models", "credit_risk.pkl")

df = CreditRiskDetection(data_path).clean()

X_train, X_test, y_train, y_test = Preprocessing(df, target="loan_status").split()

models = {
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "LR":  LogisticRegression(max_iter=1000),
    "XGB": XGBClassifier(eval_metric="logloss")
}

results = ModelEvaluation(models).evaluate(X_train, X_test, y_train, y_test)

best_model = models["XGB"]
best_model.fit(X_train, y_train)
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")

for model, res in results.items():
    print(f"\n{model}  AUC: {res['AUC']:.3f}")
    print(res["Report"])
