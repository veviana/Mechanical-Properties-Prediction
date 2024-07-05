#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraies
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib


# In[ ]:


# Create the Flask app 
app = Flask(__name__)
model = joblib.load('rfY2End.pkl') #load trained model

# Define the '/' root route to display the content from index.html
@app.route('/')
def home():
    return render_template('index.html')

# Define the '/predict' route to:
# - Get form data and convert them to float values
# - Convert form data to numpy array
# - Pass form data to model for prediction

@app.route('/predict',methods=['POST'])
def predict():
    columns = ["A7_3", "A2_1", "A2_2", "A6_1", "A2_3", "A1_1", "A1_3"]

    form_data = [float(x) for x in request.form.values()]
    features = [np.array(form_data)]
    prediction = model.predict(features)

# Format prediction text for display in "index.html"
    return render_template('index.html', iris_prediction='The mechnical property of the final product is {}'.format(prediction[0]))


# In[ ]:


# Run the Flask application (if this module is the entry point to the program)
if __name__ == "__main__":
    app.run(debug=False)

