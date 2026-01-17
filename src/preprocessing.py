from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, df, target="loan_status"):
        self.df = df
        self.target = target
        
    def split(self):
        for col in self.df.select_dtypes(include ="object"):
            self.df[col] = LabelEncoder().fit_transform(self.df[col])
        
        X = self.df.drop(columns = [self.target])
        y = self.df[self.target]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    