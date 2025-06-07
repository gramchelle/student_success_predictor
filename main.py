from preprocessing import preprocess_data
from stacking_model import run_stacking_model
from voting_model import run_voting_model
import pandas as pd

def main():
    data_source = "C:\Users\Özlem Nur\Desktop\data.csv"
    df, X_train, X_test, y_train, y_test = preprocess_data(data_source)

