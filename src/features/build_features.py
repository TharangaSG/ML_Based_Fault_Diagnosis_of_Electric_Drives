import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from sklearn.cluster import KMeans

df = pd.read_pickle('../../data/processed/01_data_processed.pkl')
df.head()

df.info()
df.describe()
df.isnull().sum() #discribe the number of null values in each column

# Drop rows with missing values
df = df.dropna()

df.columns
df.Category.unique() #discribe the unique categories in the dataset
df.Category.value_counts() #count the number of faults in the dataset

predictor_columns = list(df.columns[:-1])

# --------------------------------------------------------------
# Principal component analysis PCA

df_pca = df.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explain  variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 4)

top_features = PCA.get_top_contributing_features(df_pca, predictor_columns, number_comp=4, top_n=4)

#save dataframe with all features
df_pca.to_pickle("../../data/interim/02_data_features.pkl")

#top features
# Speed (rad/s)    1.492252
# Vab (V)          1.375541
# Ic (Amp)         1.249964
# Ia (Amp)         0.919304
# dtype: float64

