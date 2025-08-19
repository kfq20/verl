# scripts/convert_detection_to_parquet.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
input_path = "data/autogen_gsm8k/detection_results.jsonl"
output_path = "data/autogen_gsm8k/detection_results.parquet"
df = pd.read_json(input_path, lines=True)
table = pa.Table.from_pandas(df)
pq.write_table(table, output_path)
print(f"Converted {input_path} to {output_path}")