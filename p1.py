import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("   NEXORA CAREER MODEL TRAINING PIPELINE")
print("=" * 50)

# =========================
# LOAD DATASET
# =========================
df = pd.read_excel("nexora_dataset_final.xlsx")
print(f"\nDataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# =========================
# MANUAL ENCODING (same mapping as app.py)
# =========================
personality_map = {
    "Adventurous": 0, "Analytical": 1, "Creative": 2,
    "Detail-oriented": 3, "Empathetic": 4, "Logical": 5, "Organized": 6
}
workstyle_map = {
    "Freelance": 0, "Hybrid": 1, "Office": 2, "Remote": 3
}
academic_map = {
    "Accounting": 0, "Biology": 1, "Business Studies": 2,
    "Chemistry": 3, "Computer Science": 4, "Economics": 5,
    "Geography": 6, "History": 7, "Maths": 8, "Physics": 9,
    "Political Science": 10, "Psychology": 11
}
performance_map = {
    "Average": 0, "Excellent": 1, "Good": 2
}
lifestyle_map = {
    "Balance": 0, "Creativity": 1, "Growth": 2, "Stability": 3
}

df["personality_trait"]    = df["personality_trait"].map(personality_map)
df["preferred_workstyle"]  = df["preferred_workstyle"].map(workstyle_map)
df["strongest_academic"]   = df["strongest_academic"].map(academic_map)
df["academic_performance"] = df["academic_performance"].map(performance_map)
df["desired_lifestyle"]    = df["desired_lifestyle"].map(lifestyle_map)

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("Career", axis=1).astype(float)
y = df["Career"]

# Encode target
career_encoder = LabelEncoder()
y_encoded = career_encoder.fit_transform(y)

print(f"\nCareers to predict: {len(career_encoder.classes_)} unique careers")
for i, career in enumerate(career_encoder.classes_):
    print(f"  {i:2d}. {career}")

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODELS
# =========================
print("\nTraining models...")

models = {}

dt = DecisionTreeClassifier(random_state=21)
dt.fit(X_train, y_train)
models["Decision Tree"] = (dt, accuracy_score(y_test, dt.predict(X_test)))

rf = RandomForestClassifier(n_estimators=30, random_state=10)
rf.fit(X_train, y_train)
models["Random Forest"] = (rf, accuracy_score(y_test, rf.predict(X_test)))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
models["KNN"] = (knn, accuracy_score(y_test, knn.predict(X_test)))

# =========================
# MODEL COMPARISON
# =========================
print("\nModel Performance:")
print("-" * 35)
for name, (_, acc) in models.items():
    print(f"  {name:15} → Accuracy: {acc*100:.2f}%")

best_name  = max(models, key=lambda x: models[x][1])
best_model, best_acc = models[best_name]

print("-" * 35)
print(f"  BEST MODEL : {best_name}")
print(f"  ACCURACY   : {best_acc*100:.2f}%")

# =========================
# SAVE MODEL
# =========================
model_data = {
    "model":          best_model,
    "scaler":         scaler,
    "career_encoder": career_encoder,
    "features":       list(X.columns)
}

with open("career_model1.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nModel saved as: career_model1.pkl")
print("=" * 50)
