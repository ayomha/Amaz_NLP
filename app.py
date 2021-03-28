#importing load_model for loading h5 file
from keras.models import load_model
#importing numpy
import numpy as np
from flask import Flask, request, jsonify, render_template 
#importing pickle for load bag of word model
import pickle
#loading phone.h5 file using load_model
model=load_model('phone.h5')

app = Flask(__name__)

with open('count_vec.pkl','rb') as file:
    cv=pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=["GET","POST"])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    #Giving review
    if request.method =='POST':
        inp = request.form.get("Review")
        
    x=cv.transform([inp])
    y=model.predict(x)
    if(y>0.5):
        y='Positive review'
    else:
        y='Negative review'
 
    return render_template('index.html',prediction_text = 'Analysis of Amazon Cell Phone Reviews '+y)

    
if __name__ == "__main__":
    app.run(debug=True)

   
 
  
