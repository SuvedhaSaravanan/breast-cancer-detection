import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('knn1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    ss = StandardScaler()
    final_features =  ss.fit_transform(final_features)
    
    prediction = model.predict(final_features)
    prob = model.predict_proba(final_features)
    print(prob)



    proba = prob[0][0] 
    probb = prob[0][1]
    
    prediction =   "Benign"  if prediction[0] == 0 else "Malignant"

    print(prediction)

    return render_template('index.html', prediction = prediction, proba = proba, probb = probb)

if __name__ == "__main__":
    app.run(debug=True)