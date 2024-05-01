from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import assist

app = Flask(__name__)
model = pickle.load(open('disease_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load the input values from the form
        symptoms = request.form.getlist('symptoms[]')
        symptom_array = np.zeros(132) 
        for i in range(0,132):
            if assist.S_p[i] in symptoms:
                symptom_array[i] = 1
            else:
                symptom_array[i] = 0
                

        # Preprocess the symptoms to create the input array for prediction
        
        # Assuming you have 132 symptoms
        

        # Make the prediction using the loaded model
        input_data = symptom_array.reshape(1,-1)
        prediction = model.predict(input_data)
        disease = assist.Z_p[np.where(assist.Y_p==prediction[0])[0][0]]

        

        predicted_disease = f"Prediction: You may have {disease}."  # Replace 'prediction' with your actual prediction

        return render_template('index.html', prediction_text=predicted_disease)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
