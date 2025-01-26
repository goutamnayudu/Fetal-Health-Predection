import pickle
from flask import Flask, render_template, request


app = Flask(__name__)


# Load the model
with open('xgb_name_updated.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    content = file.read()
    print(content)



print(f"Loaded model type: {type(loaded_model)}")
@app.route('/')
def home():
    return render_template('8 form.html', **locals())  # Ensure template name matches exactly

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Assuming the data is received from form submission and not JSON
    accelerations = float(request.form['acc'])
    prolongued_decelerations = float(request.form['pd'])
    abnormal_short_term_variability = float(request.form['astv'])
    percentage_of_time_with_abnormal_long_term_variability = float(request.form['pot'])
    mean_value_of_long_term_variability = float(request.form['mv'])
    histogram_mode = float(request.form['hm'])
    histogram_median = float(request.form['hmed'])
    histogram_variance = float(request.form['hv'])
    
    # Features for the model
    features = [
        accelerations,
        prolongued_decelerations,
        abnormal_short_term_variability,
        percentage_of_time_with_abnormal_long_term_variability,
        mean_value_of_long_term_variability,
        histogram_mode,
        histogram_median,
        histogram_variance
    ]
    
    # Make a prediction using the loaded model
    prediction = loaded_model.predict([features])[0]
    
    return render_template('8 form.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)
