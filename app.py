from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Credit_Card.pickle.dat', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('webpage.html')


@app.route('/predict', methods=['POST'])
def home():

    features= np.array([[float(x) for x in request.form.values()]])
    pred = model.predict(features)
    
    if(pred[0]==0):
    	a="Not Defaulter"
    else:
    	a= "Defaulter"
    
    out=( "The Person is "+ a )

    return render_template('output.html',prediction_text = out)

if __name__ == "__main__":
    app.run(debug=True)
