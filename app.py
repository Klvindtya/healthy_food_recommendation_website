from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import math
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ── Load semua model dan preprocessor ──
print("⏳ Loading model dan preprocessor...")

best_model_name      = joblib.load('saved_models/best_model_name.pkl')
model_map = {
    'Deep MLP':     load_model('saved_models/deep_mlp.keras'),
    'Residual MLP': load_model('saved_models/residual_mlp.keras'),
    'Wide & Deep':  load_model('saved_models/wide_deep.keras'),
}
best_model           = model_map[best_model_name]
scaler               = joblib.load('saved_models/scaler.pkl')
encoders             = joblib.load('saved_models/label_encoders.pkl')
label_encoder_target = joblib.load('saved_models/target_encoder.pkl')
feature_cols         = joblib.load('saved_models/feature_cols.pkl')
df_food_healthy      = pd.read_csv('saved_models/healthy_foods.csv')

RANDOM_STATE = 42
TOP_N        = 10

print(f"✅ Model '{best_model_name}' berhasil diload")
print(f"✅ Dataset makanan: {len(df_food_healthy)} makanan sehat")

# ── Fungsi rekomendasi ──
def recommend_healthy_foods(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])

    if 'bmi' not in input_df.columns or pd.isna(input_df['bmi'].iloc[0]):
        w = input_df['weight'].iloc[0]
        h = input_df['height'].iloc[0]
        input_df['bmi'] = w / ((h / 100) ** 2)

    for col, le in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                input_df[col] = le.transform([le.classes_[0]])[0]

    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    X_input = scaler.transform(input_df[feature_cols].values.astype(np.float32))
    y_proba = best_model.predict(X_input, verbose=0)[0]
    y_pred_class = np.argmax(y_proba)
    predicted_diet = label_encoder_target.inverse_transform([y_pred_class])[0]
    confidence = float(y_proba[y_pred_class]) * 100

    diet_nutrition_rules = {
        'Low_Carb':       {'carbs': ('<', 20), 'protein': ('>', 10)},
        'Low_Sodium':     {'sodium': ('<', 200)},
        'High_Calorie':   {'calories': ('>', 100), 'protein': ('>', 5)},
        'Weight_Loss':    {'calories': ('<', 200), 'fiber': ('>', 2)},
        'Muscle_Gain':    {'protein': ('>', 15)},
        'Balanced_Diet':  {},
        'Improve_Health': {'fiber': ('>', 3), 'sugar': ('<', 10)},
    }

    df_filtered = df_food_healthy.copy()
    rules = diet_nutrition_rules.get(predicted_diet, {})
    for nutrient, (op, threshold) in rules.items():
        if nutrient in df_filtered.columns:
            if op == '<':
                df_filtered = df_filtered[df_filtered[nutrient] < threshold]
            elif op == '>':
                df_filtered = df_filtered[df_filtered[nutrient] > threshold]

    if len(df_filtered) < TOP_N:
        df_filtered = df_food_healthy.copy()

    df_result = df_filtered.nlargest(TOP_N * 3, 'health_score').sample(
        min(TOP_N, len(df_filtered)), random_state=RANDOM_STATE
    ).nlargest(TOP_N, 'health_score').reset_index(drop=True)

    # Fix NaN sebelum return
    df_result = df_result.fillna(0)

    return predicted_diet, confidence, df_result

# ── Routes ──
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {
            'age':              int(request.form['age']),
            'gender':           request.form['gender'],
            'weight':           float(request.form['weight']),
            'height':           float(request.form['height']),
            'activity_level':   request.form['activity_level'],
            'health_condition': request.form['health_condition'],
            'sleep_hours':      float(request.form['sleep_hours']),
            'weekly_exercise':  float(request.form['weekly_exercise']),
        }

        predicted_diet, confidence, df_rec = recommend_healthy_foods(user_input)

        recommendations = df_rec.to_dict(orient='records')
        # Double check NaN jadi 0
        for row in recommendations:
            for key, val in row.items():
                if isinstance(val, float) and math.isnan(val):
                    row[key] = 0

        return jsonify({
            'success': True,
            'predicted_diet': predicted_diet,
            'confidence': confidence,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)