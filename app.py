from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# =========================
# LOAD TRAINED MODEL
# (career_model.pkl is a dictionary saved by p1.py,
#  so we must load all parts from it)
# =========================
with open("career_model1.pkl", "rb") as f:
    model_data = pickle.load(f)

model          = model_data["model"]
scaler         = model_data["scaler"]
career_encoder = model_data["career_encoder"]


# =========================
# ROUTES
# =========================

# Serve HTML
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Extract values
        personality = data["personality_trait"]
        workstyle   = data["preferred_workstyle"]
        academic    = data["strongest_academic"]
        performance = data["academic_performance"]
        lifestyle   = data["desired_lifestyle"]
        social      = int(data["social_skills"])
        leadership  = int(data["leadership_score"])
        creativity  = int(data["creativity_score"])

        # Encoding mapping (matches nexora_dataset_final.xlsx exactly)
        mapping = {
            "personality_trait": {
                "Adventurous":      0,
                "Analytical":       1,
                "Creative":         2,
                "Detail-oriented":  3,
                "Empathetic":       4,
                "Logical":          5,
                "Organized":        6
            },
            "preferred_workstyle": {
                "Freelance": 0,
                "Hybrid":    1,
                "Office":    2,
                "Remote":    3
            },
            "strongest_academic": {
                "Accounting":       0,
                "Biology":          1,
                "Business Studies": 2,
                "Chemistry":        3,
                "Computer Science": 4,
                "Economics":        5,
                "Geography":        6,
                "History":          7,
                "Maths":            8,
                "Physics":          9,
                "Political Science":10,
                "Psychology":       11
            },
            "academic_performance": {
                "Average":   0,
                "Excellent": 1,
                "Good":      2
            },
            "desired_lifestyle": {
                "Balance":    0,
                "Creativity": 1,
                "Growth":     2,
                "Stability":  3
            }
        }

        # Convert to numeric features
        features = [
            mapping["personality_trait"][personality],
            mapping["preferred_workstyle"][workstyle],
            mapping["strongest_academic"][academic],
            mapping["academic_performance"][performance],
            mapping["desired_lifestyle"][lifestyle],
            social,
            leadership,
            creativity
        ]

        # Scale features using the same scaler from training
        final = scaler.transform(np.array(features).reshape(1, -1))

        # Get numeric prediction and convert back to career name
        prediction_encoded = model.predict(final)[0]
        prediction = career_encoder.inverse_transform([prediction_encoded])[0]

        # Confidence
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(final)) * 100)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2) if confidence else 95
        })

    except KeyError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
