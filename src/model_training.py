import argparse
import polars as pl
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

def load_data(file_path):
    df = pl.read_csv(file_path)
    return df

def split_data(df):
    X_train, X_val, y_train, y_val = train_test_split(df.drop("Insect").to_numpy(), df["Insect"].to_numpy())
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=100,
            learning_rate=0.1,
            eval_metric='mlogloss',
            random_state=27
        )
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)