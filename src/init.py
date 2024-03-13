from src.load_data import df
import pandas as pd

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

test_size = 0.2

split_idx = int(len(df_shuffled) * (1 - test_size))

train_data = df_shuffled[:split_idx]
test_data = df_shuffled[split_idx:]

X_train = train_data.drop(['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient'], axis=1)
y_train = train_data[['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']]

X_test = test_data.drop(['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient'], axis=1)

y_test = test_data[['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']]
