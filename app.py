#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


app= Flask(__name__)
Diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = Diabetes_model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return (flask.render_template('index.html', prediction_text='The probability that the patient has diabetes is {}'.format(output)))


# In[5]:


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = Diabetes_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# In[ ]:


if __name__ == "__main__":
    app.run(port=5000, debug=True,  use_reloader=False)
    


# In[ ]:





# In[ ]:




