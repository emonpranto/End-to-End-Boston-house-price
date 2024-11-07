from flask import Flask, render_template, request,app, jsonify, url_for
import pickle
import numpy as np 
import pandas as pd 


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(data.values())
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    print(new_data)
    output = model.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route("/predict",methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    output = model.predict(final_input)[0]
    return render_template('home.html',prediction_text = 'The prediction of the house price is {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)