import argparse
import polars as pl
from sklearn.preprocessing import MinMaxScaler
from utils import normalize_columns, is_day

def load_data(file_path):
    df = pl.read_csv(file_path)
    return df


def clean_data(df):
    cols_to_clean = ["Hour", "Minutes"
    'Sensor_alpha',
    'Sensor_beta',
    'Sensor_gamma',
    'Sensor_alpha_plus',
    'Sensor_beta_plus',
    'Sensor_gamma_plus']
    df_clean = normalize_columns(df, cols_to_clean)
    return df_clean

def preprocess_data(df):
    df_processed = df.with_columns(
        pl.col("Hour").map_elements(is_day).alias("is_day")
    )
    return df_processed

def save_data(df, output_file):
    df.write_csv(output_file)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)