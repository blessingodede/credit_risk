import pandas as pd
class CreditRiskDetection:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
    
    def clean(self):
        self.df = self.df.drop_duplicates().dropna()
        return self.df