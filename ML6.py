"""
Experiment-6:
Write a program to implement Categorical Encoding, One-hot
Encoding
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {'Color': ['Red', 'Green', 'Blue', 'Red', 'Green']}
df = pd.DataFrame(data)

print("Original Data:")
print(df)

encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(df[['Color']])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Color']))

print("\nOne-Hot Encoded Data:")
print(encoded_df)

final_df = pd.concat([df, encoded_df], axis=1)
print("\nFinal DataFrame with One-Hot Encoding:")
print(final_df)
