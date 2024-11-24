import time
import pandas as pd 

from my_model_package.preprocessing.utils import PreProcessing

if __name__ == '__main__':

    df = pd.read_csv('data/blight_violations.csv')
    
    # Add a timer to see how long it takes to run the code
    start = time.time()
    preprocessing = PreProcessing(df)
    processed_data = preprocessing.pre_processing()
    end = time.time()
    print(f'Time taken to process data: {round(end - start, 2)} seconds')


    processed_data.to_csv('data/processed_data.csv', index=False)