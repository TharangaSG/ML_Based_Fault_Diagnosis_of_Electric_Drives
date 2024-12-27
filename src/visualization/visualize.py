import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython.display as display
import seaborn as sns

df = pd.read_pickle('../../data/processed/01_data_processed.pkl')

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Drop rows with missing values
df = df.dropna()

columns = df.columns
# Data Visualization: Histograms for each feature

#display features
feature_df = df.drop(columns=['Category'])
label_df = df['Category']

label_df.unique()

features = columns[:-1]


import seaborn as sns
import matplotlib.pyplot as plt


# Create a figure and axes for subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
axes = axes.flatten()  # Flatten to easily iterate through all axes

# Iterate through each column and plot
for i, column in enumerate(feature_df.columns):
    sns.histplot(feature_df[column], kde=True, ax=axes[i])  
    axes[i].set_xlabel(column) 
    axes[i].set_ylabel('Count')  

plt.show()


