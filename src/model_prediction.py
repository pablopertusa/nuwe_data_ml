import argparse
import polars as pl
import joblib
from data_processing import clean_data, preprocess_data
import json

def load_data(file_path):
    df_test = pl.read_csv(file_path)
    df_test = preprocess_data(clean_data(df_test))
    return df_test

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def make_predictions(df, model):
    predictions = model.predict(df.to_numpy())
    return predictions

def save_predictions(predictions, predictions_file):
    aux = pl.read_csv("data/index.csv")
    ids = aux["Unnamed: 0"].to_list()
    resul = {"target": {}}
    for i in range(predictions.shape[0]):
        resul["target"][str(ids[i])] = int(predictions[i])
    with open(predictions_file, "w") as writer:
        json.dump(resul, writer, indent = 4)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
