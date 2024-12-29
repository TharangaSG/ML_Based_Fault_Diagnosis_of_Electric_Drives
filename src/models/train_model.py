import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

df = pd.read_pickle("../../data/interim/02_data_features.pkl")

df.columns

#top features
# Speed (rad/s)    1.492252
# Vab (V)          1.375541
# Ic (Amp)         1.249964
# Ia (Amp)         0.919304

#Select top contribution features and Ib for traing feature set
feature_set1 = ['Ia (Amp)','Ib (Amp)', 'Ic (Amp)', 'Vab (V)', 'Speed (rad/s)']

df['Category'].unique()

label_category = LabelEncoder()
df['Category'] = label_category.fit_transform(df['Category'])
df.head()

df['Category'].unique()

df.Category.value_counts()

#array(['NOM', 'PTPF', 'PTGF', 'UVF', 'OVF', 'OLF']
#array([0, 4, 3, 5, 2, 1])

x = df[feature_set1]
x.info()

y = df['Category']
y.info()


# #make a pipeline for preprocessing the data
# numeric_transformer = make_pipeline(
#     SimpleImputer(strategy = "mean"),
#     MinMaxScaler()
# )
# numeric_transformer.fit(x)


#create a column transformer to apply the numeric transformer to all numeric columns
scaler = MinMaxScaler()

# Apply scaling to the feature set 'x'
X = scaler.fit_transform(x)

#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2) 



#-------------------------------------------------------------------------------
#KNN without grid search

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Fit the model
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

#save knn model
joblib.dump(knn_model, '../../models/knn_model.pkl')


#KNN with grid search to find the best hyperparameters
cv = KFold(n_splits=5)
params = [{'n_neighbors': np.arange(1, 50, 2),
           'weights': ['uniform', 'distance'],
           'p': [1,2],
           'n_jobs': [-1]
          }]
knn_clf = KNeighborsClassifier()
clf = GridSearchCV(knn_clf,
                      param_grid=params,
                      scoring='accuracy',
                      cv=cv)
clf.fit(X_train,y_train)

clf.best_params_
#{'n_jobs': -1, 'n_neighbors': 19, 'p': 1, 'weights': 'uniform'}

knn_clf = KNeighborsClassifier(n_neighbors=19, weights='uniform', p=1, n_jobs=-1 )
cv = KFold(n_splits=10)
scores_knn = pd.DataFrame(cross_validate(knn_clf, X_train, y_train, scoring= 'accuracy', cv=cv, n_jobs=-1, 
error_score='raise', return_train_score=True))

#save the model
joblib.dump(knn_clf, '../../models/knn_model.pkl')



#-------------------------------------------------------------------------------
#Random Forest without grid search

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Fit the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

joblib.dump(rf, '../../models/rf_model.pkl')



#Random Forest with grid search to find the best hyperparameters

cv = KFold(n_splits=5)
params = [{'n_estimators': np.arange(start = 5, stop = 105, step = 5),
         'criterion': ['gini'],
         'max_depth': np.arange(start = 1, stop = 20, step = 1),
         'max_features':['sqrt', 'log2'],
         'min_samples_leaf': np.arange(start = 1, stop = 10, step = 1),
           'n_jobs': [-1]
          }]
rf_clf = RandomForestClassifier()
clf = GridSearchCV(rf_clf,
                      param_grid=params,
                      scoring='accuracy',
                      cv=cv)
clf.fit(X_train,y_train)



#-------------------------------------------------------------------------------
#SVM without grid search

from sklearn.svm import SVC

# Initialize the SVM model
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # Using RBF kernel

# Fit the model
svm.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm.predict(X_test)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

#save the model
joblib.dump(svm, '../../models/svm_model.pkl')


#-------------------------------------------------------------------------------
# Neural Network

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Convert to categorical format for the neural network
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)
len(y_train_categorical[0])

# Define the neural network model
model_v2 = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),  
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')  # Output layer
])

# Compile the model
model_v2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model_v2.fit(X_train, y_train_categorical, 
          epochs=500, 
          batch_size=32, 
          validation_data=(X_test, y_test_categorical))


# Evaluate the model on the test set
y_pred_categorical = model_v2.predict(X_test)
y_pred_classes = np.argmax(y_pred_categorical, axis=1)  # Convert probabilities to class labels

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_classes)
print(f"Neural Network Accuracy: {accuracy}")

from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Ensure label_encoder.classes_ is properly formatted
class_names = [str(cls) for cls in label_encoder.classes_]

# Check the length of y_test_encoded and y_pred_classes
print(f"Length of y_test_encoded: {len(y_test_encoded)}")
print(f"Length of y_pred_classes: {len(y_pred_classes)}")

# Generate classification report
class_report = classification_report(
    y_test_encoded, 
    y_pred_classes, 
    target_names=class_names
)

print("Neural Network Classification Report:")
print(class_report)


# Save the model in HDF5 format
model_v2.save("../../models/model_v2.h5")  # Creates a single file named "model_v2.h5"

# save the model in keras format
model_v2.save("../../models/model_v2.keras")

import joblib
joblib.dump(scaler, "../../models/scaler.pkl")  # Save the scaler

# Save the fitted LabelEncoder
joblib.dump(label_encoder, '../../models/label_encoder.pkl')

#1001/1001 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9161 - loss: 0.1916 - val_accuracy: 0.9121 - val_loss: 0.1934

