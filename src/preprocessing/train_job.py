
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class TrainingModel():

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X = self.df.drop('payment_status', axis=1)
        self.y = self.df['payment_status']
        self.model = XGBClassifier()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        return score
    
        