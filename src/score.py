import json
import pickle
import numpy as np

def init():
    global model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

def run(raw_data):
    data = json.loads(raw_data)
    X = np.array(data["data"])
    preds = model.predict(X)
    return preds.tolist()
