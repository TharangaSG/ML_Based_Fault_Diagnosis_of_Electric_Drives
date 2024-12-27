import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from your classification reports
models_data = {
    "KNN": {
        "precision": [0.84, 0.98, 0.98, 0.83, 0.85, 0.98],
        "recall": [0.86, 0.97, 0.98, 0.86, 0.79, 0.97],
        "f1-score": [0.85, 0.98, 0.98, 0.85, 0.82, 0.97],
    },
    "SVM": {
        "precision": [0.81, 0.81, 1.00, 1.00, 1.00, 0.98],
        "recall": [0.96, 0.93, 0.96, 0.79, 0.76, 0.96],
        "f1-score": [0.88, 0.86, 0.98, 0.88, 0.86, 0.97],
    },
    "RF": {
        "precision": [0.82, 0.99, 1.00, 0.95, 0.97, 1.00],
        "recall": [0.97, 0.98, 0.96, 0.83, 0.80, 0.97],
        "f1-score": [0.89, 0.98, 0.98, 0.88, 0.87, 0.98],
    },
    "NN": {
        "precision": [0.84, 1.00, 0.99, 0.90, 0.97, 1.00],
        "recall": [0.95, 0.97, 0.97, 0.87, 0.79, 0.97],
        "f1-score": [0.89, 0.98, 0.98, 0.88, 0.87, 0.98],
    },
}

# Convert data to a DataFrame for easier plotting
classes = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
metrics = ["precision", "recall", "f1-score"]

data = []
for model, metrics_data in models_data.items():
    for metric in metrics:
        for class_idx, value in enumerate(metrics_data[metric]):
            data.append({"Model": model, "Class": classes[class_idx], "Metric": metric, "Value": value})

df = pd.DataFrame(data)

# Plot grouped bar charts
plt.figure(figsize=(14, 8))
sns.barplot(data=df, x="Class", y="Value", hue="Model", ci=None)
plt.title("Classification Metrics by Class and Model", fontsize=16)
plt.ylabel("Metric Value", fontsize=12)
plt.xlabel("Class", fontsize=12)
plt.legend(title="Model", fontsize=10)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data for heatmaps
models_data = {
    "KNN": {
        "precision": [0.84, 0.98, 0.98, 0.83, 0.85, 0.98],
        "recall": [0.86, 0.97, 0.98, 0.86, 0.79, 0.97],
        "f1-score": [0.85, 0.98, 0.98, 0.85, 0.82, 0.97],
    },
    "SVM": {
        "precision": [0.81, 0.81, 1.00, 1.00, 1.00, 0.98],
        "recall": [0.96, 0.93, 0.96, 0.79, 0.76, 0.96],
        "f1-score": [0.88, 0.86, 0.98, 0.88, 0.86, 0.97],
    },
    "RF": {
        "precision": [0.82, 0.99, 1.00, 0.95, 0.97, 1.00],
        "recall": [0.97, 0.98, 0.96, 0.83, 0.80, 0.97],
        "f1-score": [0.89, 0.98, 0.98, 0.88, 0.87, 0.98],
    },
    "NN": {
        "precision": [0.84, 1.00, 0.99, 0.90, 0.97, 1.00],
        "recall": [0.95, 0.97, 0.97, 0.87, 0.79, 0.97],
        "f1-score": [0.89, 0.98, 0.98, 0.88, 0.87, 0.98],
    },
}

# Create heatmaps for each metric
metrics = ["precision", "recall", "f1-score"]
classes = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

for metric in metrics:
    heatmap_data = {model: data[metric] for model, data in models_data.items()}
    df = pd.DataFrame(heatmap_data, index=classes)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={"label": metric.capitalize()})
    plt.title(f"{metric.capitalize()} Heatmap Across Models", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Class", fontsize=12)
    plt.tight_layout()
    plt.show()
