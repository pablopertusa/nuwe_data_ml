import polars as pl

def normalize_columns(df: pl.DataFrame, cols: list[str]):
    df_normalized = df.with_columns(
        [
            ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(f"{col}_n")
            for col in cols
        ]
    )
    df_normalized.drop_in_place(cols)
    return df_normalized

def is_day(hour: int):
    if hour >= 7 or hour <= 20:
        return 1
    return 0