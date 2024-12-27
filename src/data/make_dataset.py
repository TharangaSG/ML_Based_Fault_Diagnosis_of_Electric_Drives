import pandas as pd

df = pd.read_csv('../../data/external/All Data.csv')
df = df.drop(columns=['Unnamed: 0','time (sec)'], axis=1)
df.head()

df.to_pickle('../../data/processed/01_data_processed.pkl')