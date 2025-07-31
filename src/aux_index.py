import polars as pl

file_path = "data/test.csv"
df = pl.read_csv(file_path)
aux_df = pl.DataFrame({
    "Unnamed: 0": df["Unnamed: 0"].to_list(),
    "Index": list(range(len(df)))
})
output_file = "data/index.csv"
aux_df.write_csv(output_file)
print("Index csv created")