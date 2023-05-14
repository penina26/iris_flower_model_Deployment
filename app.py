from flask import Flask, render_template,session,url_for, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class FlowerForm(FlaskForm):
    sep_len = StringField("sepal Length")
    sep_wid = StringField("Sepal Width")
    pet_len = StringField("Petal Length")
    pet_wid = StringField("Petal Width")
    submit = SubmitField("analyze")


def return_prediction(model, scaler, sample_json):
    flower = pd.DataFrame([sample_json], index=[0])  # Provide an index for the DataFrame
    flower = scaler.transform(flower)
    class_probabilities = model.predict(flower)
    class_ind = np.argmax(class_probabilities, axis=1)
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    predicted_class = classes[class_ind][0]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():

	form = FlowerForm()
	if form.validate_on_submit():
		session['sep_len'] = form.sep_len.data
		session['sep_wid'] = form.sep_wid.data
		session['pet_len'] = form.pet_len.data
		session['pet_wid'] = form.pet_wid.data

		return redirect(url_for("prediction"))	

	return render_template('home.html', form=form)

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

@app.route('/prediction')
def prediction():
	content = {}
	content['sepal_length'] = float(session['sep_len'])
	content['sepal_width'] = float(session['sep_wid'])
	content['petal_length'] = float(session['pet_len'])
	content['petal_width'] = float(session['pet_wid'])

	# content = request.json
	results = return_prediction(flower_model, flower_scaler, content)
	return render_template('prediction.html',results=results)

if __name__ =='__main__':
	app.run(debug=True)