import numpy as np
from flask import Flask,request,render_template
import math
import pickle

app=Flask(__name__)#app initiate
model2=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
#get input and generate output

@app.route('/predict',methods=['POST'])
def predict():
    int_feature=[int(i) for i in request.form.values()]#store values from form  in int  iterate

    #convert 1d to 2d in numpy
    # print(model1.predict([[40,1690000,11800,105]]))

    final_features=np.array(int_feature).reshape(1,-1)# convert [1,2,3,4]  into  [[1,2,3,4] ]

    #import prediction
    prediction=model2.predict(final_features)
    output=round(prediction[0],2)

    #how to render in UI
    return render_template('index.html',predict_text='Number of Weekly Rides:- {}'.format(math.floor(output)))
if __name__ == '__main__':
    app.run(debug=True)
