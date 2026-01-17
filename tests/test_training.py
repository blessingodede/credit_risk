import os
from src.credit_risk_detection import CreditRiskDetection
from src.preprocessing import Preprocessing
from sklearn.linear_model import LogisticRegression

def test_model():
    CDIR = os.path.dirname(os.path.realpath(__file__))
    WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))
    
    data_path = os.path.join(WORKDIR, "data", "credit_risk_dataset.csv")
    
    df = CreditRiskDetection(data_path).clean()
    X_train, X_test, y_train, y_test = Preprocessing(df, target="loan_status").split()
    
    model = LogisticRegression(max_iter=1000)
    model  = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    assert model is not None, "Model was not trained"
    assert len(y_pred) == len(y_test), "Prediction does not match the test set"
    assert set(y_pred).issubset({0, 1}), "Predictions contain invalid class"
    
    print("Test Passed: The model passed all the test cases")
    

if __name__=="__main__":
    test_model()
    
    
    