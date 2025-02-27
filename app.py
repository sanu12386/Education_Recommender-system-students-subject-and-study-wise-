# Install dependencies (if not already installed)
# pip install scikit-learn==1.3.2
# pip install numpy
# pip install flask

# Load packages
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler, model, and class names
try:
    scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
    model = pickle.load(open("Models/model.pkl", 'rb'))
except FileNotFoundError:
    print("Error: Model files not found. Ensure 'Models/scaler.pkl' and 'Models/model.pkl' exist.")

class_names = [
    'Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
    'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
    'Banker', 'Writer', 'Accountant', 'Designer',
    'Construction Engineer', 'Game Developer', 'Stock Investor',
    'Real Estate Developer'
]

# Recommendations function
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    try:
        # Encode categorical variables
        gender_encoded = 1 if gender.lower() == 'female' else 0
        part_time_job_encoded = 1 if part_time_job else 0
        extracurricular_activities_encoded = 1 if extracurricular_activities else 0

        # Create feature array
        feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                                   weekly_self_study_hours, math_score, history_score, physics_score,
                                   chemistry_score, biology_score, english_score, geography_score, total_score,
                                   average_score]])

        # Scale features
        scaled_features = scaler.transform(feature_array)

        # Predict using the model
        probabilities = model.predict_proba(scaled_features)

        # Get top three predicted classes along with their probabilities
        top_classes_idx = np.argsort(-probabilities[0])[:3]
        top_classes_names_probs = [(class_names[idx], round(probabilities[0][idx] * 100, 2)) for idx in top_classes_idx]

        return top_classes_names_probs

    except Exception as e:
        print(f"Error in Recommendations function: {e}")
        return [("Error", 0)]

# Flask Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        try:
            # Extract form values safely
            gender = request.form.get('gender', 'male')
            part_time_job = request.form.get('part_time_job') == 'true'
            absence_days = int(request.form.get('absence_days', 0))
            extracurricular_activities = request.form.get('extracurricular_activities') == 'true'
            weekly_self_study_hours = int(request.form.get('weekly_self_study_hours', 0))
            math_score = int(request.form.get('math_score', 0))
            history_score = int(request.form.get('history_score', 0))
            physics_score = int(request.form.get('physics_score', 0))
            chemistry_score = int(request.form.get('chemistry_score', 0))
            biology_score = int(request.form.get('biology_score', 0))
            english_score = int(request.form.get('english_score', 0))
            geography_score = int(request.form.get('geography_score', 0))
            total_score = float(request.form.get('total_score', 0))
            average_score = float(request.form.get('average_score', 0))

            # Get recommendations
            recommendations = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                              weekly_self_study_hours, math_score, history_score, physics_score,
                                              chemistry_score, biology_score, english_score, geography_score,
                                              total_score, average_score)

            return render_template('results.html', recommendations=recommendations)

        except Exception as e:
            print(f"Error in /pred route: {e}")
            return render_template('error.html', error_message="Invalid input or server error")

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
