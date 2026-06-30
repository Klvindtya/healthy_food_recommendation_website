from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

NGROK_URL = "https://agreeable-landfill-commodity.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "true"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        'age': int(request.form['age']),
        'gender': request.form['gender'],
        'weight': float(request.form['weight']),
        'height': float(request.form['height']),
        'activity_level': request.form['activity_level'],
        'health_condition': request.form['health_condition'],
        'sleep_hours': float(request.form['sleep_hours']),
        'weekly_exercise': float(request.form['weekly_exercise']),
    }

    try:
        resp = requests.post(f"{NGROK_URL}/predict", json=user_input,
                              headers=HEADERS, timeout=30)
        result = resp.json()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)