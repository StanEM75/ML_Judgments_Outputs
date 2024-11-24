import time
import pandas as pd 

from my_model_package.preprocessing.train_job import TrainingModel

if __name__ == '__main__':

    df = pd.read_csv('data/processed_data.csv')
    
    # Add a timer to see how long it takes to run the code
    start = time.time()
    train = TrainingModel(df)
    X_train, X_test, y_train, y_test  = train.split_data()
    model = train.train_model(X_train, y_train)

    # Evaluate the model
    score = train.evaluate_model(X_test, y_test)
    print(f'Model score: {score}')
    end = time.time()
