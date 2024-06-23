from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import assist

app = Flask(__name__,static_folder="static",template_folder="templates")
model = pickle.load(open('disease_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
async def home():
    fobj=open("symptoms.txt",'r')
    tobj=fobj.read()
    symp_list=tobj.split(",")
    temp_count=0
    symp_list_html=''
    for i in symp_list:
        symp_list_html+=f'<div class="option" data-value="{temp_count}"> {i.replace("_"," ")} </div>'
        temp_count+=1
    return  render_template('new_index.html', symptom_list_html=symp_list_html )

@app.route("/predict", methods=['POST'])
async def predict():
    if request.method == 'POST':
        
        symptom_nums = request.get_json()["data"]
        symptom_nums=symptom_nums.split(", ")
        symptom_nums=[eval(i) for i in symptom_nums]
        
        symp_array=np.zeros(132)
        for i in range(0,132):
            if i in symptom_nums:
                symp_array[i]=1
            else:
                symp_array[i]=0


        input_data = symp_array.reshape(1,-1)
        prediction = model.predict(input_data)
        disease = assist.Z_p[np.where(assist.Y_p==prediction[0])[0][0]]

        

        predicted_disease = f"Prediction: You may have {disease}."  
        print(predicted_disease)

        return jsonify({"message":predicted_disease})
    else:
        return jsonify({"message":"Not a valid option selected."})

if __name__ == "__main__":
    app.run(debug=True)
