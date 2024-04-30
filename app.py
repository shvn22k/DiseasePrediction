from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

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

        # Preprocess the symptoms to create the input array for prediction
        symptom_array = np.zeros(132)  # Assuming you have 132 symptoms
        symptom_mapping = {
            'itching': 'symptom1',
            'skin_rash': 'symptom2',
            'nodal_skin_eruptions': 'symptom3',
            'continuous_sneezing': 'symptom4',
            'shivering': 'symptom5',
            'chills': 'symptom6',
            'joint_pain': 'symptom7',
            'stomach_pain': 'symptom8',
            'acidity': 'symptom9',
            'ulcers_on_tongue': 'symptom10',
            'muscle_wasting': 'symptom11',
            'vomiting': 'symptom12',
            'burning_micturition': 'symptom13',
            'spotting_urination': 'symptom14',
            'fatigue': 'symptom15',
            'weight_gain': 'symptom16',
            'anxiety': 'symptom17',
            'cold_hands_and_feets': 'symptom18',
            'mood_swings': 'symptom19',
            'weight_loss': 'symptom20',
            'restlessness': 'symptom21',
            'lethargy': 'symptom22',
            'patches_in_throat': 'symptom23',
            'irregular_sugar_level': 'symptom24',
            'cough': 'symptom25',
            'high_fever': 'symptom26',
            'sunken_eyes': 'symptom27',
            'breathlessness': 'symptom28',
            'sweating': 'symptom29',
            'dehydration': 'symptom30',
            'indigestion': 'symptom31',
            'headache': 'symptom32',
            'yellowish_skin': 'symptom33',
            'dark_urine': 'symptom34',
            'nausea': 'symptom35',
            'loss_of_appetite': 'symptom36',
            'pain_behind_the_eyes': 'symptom37',
            'back_pain': 'symptom38',
            'constipation': 'symptom39',
            'abdominal_pain': 'symptom40',
            'diarrhoea': 'symptom41',
            'mild_fever': 'symptom42',
            'yellow_urine': 'symptom43',
            'yellowing_of_eyes': 'symptom44',
            'acute_liver_failure': 'symptom45',
            'fluid_overload': 'symptom46',
            'swelling_of_stomach': 'symptom47',
            'swelled_lymph_nodes': 'symptom48',
            'malaise': 'symptom49',
            'blurred_and_distorted_vision': 'symptom50',
            'phlegm': 'symptom51',
            'throat_irritation': 'symptom52',
            'redness_of_eyes': 'symptom53',
            'sinus_pressure': 'symptom54',
            'runny_nose': 'symptom55',
            'congestion': 'symptom56',
            'chest_pain': 'symptom57',
            'weakness_in_limbs': 'symptom58',
            'fast_heart_rate': 'symptom59',
            'pain_during_bowel_movements': 'symptom60',
            'pain_in_anal_region': 'symptom61',
            'bloody_stool': 'symptom62',
            'irritation_in_anus': 'symptom63',
            'neck_pain': 'symptom64',
            'dizziness': 'symptom65',
            'cramps': 'symptom66',
            'bruising': 'symptom67',
            'obesity': 'symptom68',
            'swollen_legs': 'symptom69',
            'swollen_blood_vessels': 'symptom70',
            'puffy_face_and_eyes': 'symptom71',
            'enlarged_thyroid': 'symptom72',
            'brittle_nails': 'symptom73',
            'swollen_extremeties': 'symptom74',
            'excessive_hunger': 'symptom75',
            'extra_marital_contacts': 'symptom76',
            'drying_and_tingling_lips': 'symptom77',
            'slurred_speech': 'symptom78',
            'knee_pain': 'symptom79',
            'hip_joint_pain': 'symptom80',
            'muscle_weakness': 'symptom81',
            'stiff_neck': 'symptom82',
            'swelling_joints': 'symptom83',
            'movement_stiffness': 'symptom84',
            'spinning_movements': 'symptom85',
            'loss_of_balance': 'symptom86',
            'unsteadiness': 'symptom87',
            'weakness_of_one_body_side': 'symptom88',
            'loss_of_smell': 'symptom89',
            'bladder_discomfort': 'symptom90',
            'foul_smell_of_urine': 'symptom91',
            'continuous_feel_of_urine': 'symptom92',
            'passage_of_gases': 'symptom93',
            'internal_itching': 'symptom94',
            'toxic_look_(typhos)': 'symptom95',
            'depression': 'symptom96',
            'irritability': 'symptom97',
            'muscle_pain': 'symptom98',
            'altered_sensorium': 'symptom99',
            'red_spots_over_body': 'symptom100',
            'belly_pain': 'symptom101',
            'abnormal_menstruation': 'symptom102',
            'dischromic_patches': 'symptom103',
            'watering_from_eyes': 'symptom104',
            'increased_appetite': 'symptom105',
            'polyuria': 'symptom106',
            'family_history': 'symptom107',
            'mucoid_sputum': 'symptom108',
            'rusty_sputum': 'symptom109',
            'lack_of_concentration': 'symptom110',
            'visual_disturbances': 'symptom111',
            'receiving_blood_transfusion': 'symptom112',
            'receiving_unsterile_injections': 'symptom113',
            'coma': 'symptom114',
            'stomach_bleeding': 'symptom115',
            'distention_of_abdomen': 'symptom116',
            'history_of_alcohol_consumption': 'symptom117',
            'fluid_overload.1': 'symptom118',
            'blood_in_sputum': 'symptom119',
            'prominent_veins_on_calf': 'symptom120',
            'palpitations': 'symptom121',
            'painful_walking': 'symptom122',
            'pus_filled_pimples': 'symptom123',
            'blackheads': 'symptom124',
            'scurring': 'symptom125',
            'skin_peeling': 'symptom126',
            'silver_like_dusting': 'symptom127',
            'small_dents_in_nails': 'symptom128',
            'inflammatory_nails': 'symptom129',
            'blister': 'symptom130',
            'red_sore_around_nose': 'symptom131',
            'yellow_crust_ooze': 'symptom132',
            }
        def get_symptom_index(symptom_name, symptom_mapping):
            keys_list = list(symptom_mapping.keys())  # Convert keys to a list
            if symptom_name in symptom_mapping:
                return keys_list.index(symptom_name)


        for symptom in symptoms:
            actual_symptom = symptom_mapping[symptom]
            index = get_symptom_index(actual_symptom, symptom_mapping)  # You need to define a function to get the index based on the actual symptom name
            symptom_array[index] = 1  

        #for symptom in symptoms:
            #index = int(symptom.split('symptom')[1]) - 1  # Extracting the index from the symptom string
            #symptom_array[index] = 1  # Set the corresponding index to 1 for selected symptoms

        # Make the prediction using the loaded model
        input_data = symptom_array.reshape(1,-1)
        prediction = model.predict(input_data)
        predicted_disease = f"Disease Predicted: {prediction}"  # Replace 'prediction' with your actual prediction

        return render_template('index.html', prediction_text=predicted_disease)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
