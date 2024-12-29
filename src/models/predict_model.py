from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import joblib

# Load the model
model = load_model("../../models/model_v2.keras")  
scaler_loaded = joblib.load("../../models/scaler.pkl")  # Load the scaler
label_encoder = joblib.load('../../models/label_encoder.pkl') # Load the LabelEncoder 

# prediction using NN model

def predict_fault(input_features): # type: ignore
    # Scale the input features
    input_scaled = scaler_loaded.transform(np.array([input_features]))
    # Predict using the trained model
    prediction = model.predict(input_scaled)
    # Convert the prediction to a fault label
    predicted_class = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_class)

# Example usage
# new_input = [20.72434706,19.6774055,40.16305742,474.5844146,13.76353332] #PTGF
# new_input = [33.78554135,4.618768599,31.45620687,428.1455691,139.0761582] #PTPF 
# new_input = [13.302364,13.3023021,13.30235051,474.5657245,146.3062041] #NOM
new_input = [21.84772445,21.84767972,21.84771613,474.9807774,137.1003487] #OLF
# new_input = [26.5483319,26.98450226,23.72602731,346.3654977,138.1897918] #UVF
# new_input = [11.0971421,11.09704879,11.09709495,584.551318,150.7162151] #OVF

predicted_fault = predict_fault(new_input)
print("Predicted Fault:", predicted_fault[0])

#array(['NOM', 'PTPF', 'PTGF', 'UVF', 'OVF', 'OLF']
#array([0, 4, 3, 5, 2, 1])


#prediction using SVM, KNN, RF

import joblib
import numpy as np

# Load the scaler and models
scaler = joblib.load('../../models/scaler.pkl')          # Scaler
svm_model = joblib.load('../../models/svm_model.pkl')    # SVM model
knn_model = joblib.load('../../models/knn_model.pkl')    # KNN model
rf_model = joblib.load('../../models/random_forest_model.pkl')      # Random Forest model

# Example new input
new_input = np.array([[26.5483319,26.98450226,23.72602731,346.3654977,138.1897918]])  

# Scale the input
new_input_scaled = scaler.transform(new_input)  # Apply the same scaling as during training

# Predict with each model
svm_prediction = svm_model.predict(new_input_scaled)
knn_prediction = knn_model.predict(new_input_scaled)
rf_prediction = rf_model.predict(new_input_scaled)

# Print predictions
print("SVM Prediction:", svm_prediction)
print("KNN Prediction:", knn_prediction)
print("Random Forest Prediction:", rf_prediction)





import numpy as np
import joblib
import time

# Load the model
model = load_model("../../models/model_v2.keras")  
scaler_loaded = joblib.load("../../models/scaler.pkl")  # Load the scaler
label_encoder = joblib.load('../../models/label_encoder.pkl') # Load the LabelEncoder 

# Prediction using NN model
def predict_fault(input_features):
    # Start timing
    start_time = time.time()
    
    # Scale the input features
    input_scaled = scaler_loaded.transform(np.array([input_features]))
    
    # Predict using the trained model
    prediction = model.predict(input_scaled)
    
    # End timing
    end_time = time.time()
    
    # Convert the prediction to a fault label
    predicted_class = np.argmax(prediction, axis=1)
    
    # Print the time taken
    print(f"Neural Network Prediction Time: {end_time - start_time:.6f} seconds")
    
    return label_encoder.inverse_transform(predicted_class)

# Example usage
new_input = [21.84772445,21.84767972,21.84771613,474.9807774,137.1003487] # OLF
predicted_fault = predict_fault(new_input)
print("Predicted Fault (NN):", predicted_fault[0])

# Prediction using SVM, KNN, RF
# Load the scaler and models
svm_model = joblib.load('../../models/svm_model.pkl')    # SVM model
knn_model = joblib.load('../../models/knn_model.pkl')    # KNN model
rf_model = joblib.load('../../models/random_forest_model.pkl')  # Random Forest model

# Example new input
new_input = np.array([[21.84772445,21.84767972,21.84771613,474.9807774,137.1003487] ])  

# Scale the input
new_input_scaled = scaler_loaded.transform(new_input)  # Apply the same scaling as during training

# Predict with each model and measure time
# SVM
start_time = time.time()
svm_prediction = svm_model.predict(new_input_scaled)
end_time = time.time()
print(f"SVM Prediction Time: {end_time - start_time:.6f} seconds")
print("SVM Prediction:", svm_prediction)

# KNN
start_time = time.time()
knn_prediction = knn_model.predict(new_input_scaled)
end_time = time.time()
print(f"KNN Prediction Time: {end_time - start_time:.6f} seconds")
print("KNN Prediction:", knn_prediction)

# Random Forest
start_time = time.time()
rf_prediction = rf_model.predict(new_input_scaled)
end_time = time.time()
print(f"Random Forest Prediction Time: {end_time - start_time:.6f} seconds")
print("Random Forest Prediction:", rf_prediction)
