"""
Artificial Neural Network (ANN) - Student Productivity Predictor
================================================================
A beginner-friendly, clean, and well-structured implementation of an
Artificial Neural Network using scikit-learn (MLPRegressor).

Steps:
1. Load dataset
2. Select features and target
3. Train/test split
4. Scale data using StandardScaler
5. Train ANN model with 2 hidden layers
6. Evaluate model performance (Accuracy)
7. Save trained model
8. Predict productivity from interactive user inputs
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# =====================================================================
# Step 1: Load Dataset
# =====================================================================
print("=" * 60)
print("     DIGITAL PROCRASTINATION PREDICTOR USING ANN")
print("=" * 60)

dataset_file = "people_procrastination.csv"
print("\n[1] Loading dataset from", dataset_file, "...")
df = pd.read_csv(dataset_file)
print("    Dataset loaded successfully! Total records:", len(df))


# =====================================================================
# Step 2: Statistical Correlation Analysis & Feature Selection
# =====================================================================
print("\n[2] Feature correlation with productivity score:")
correlations = df.corr(numeric_only=True)["productivity_score"]
print(correlations.sort_values(ascending=False))

# Selected the 6 features with strong correlation (|r| >= 0.15)
features = [
    "study_hours_per_day",
    "sleep_hours",
    "phone_usage_hours",
    "attendance_percentage",
    "stress_level",
    "focus_score",
]
target = "productivity_score"

X = df[features]
y = df[target]

print("\nFeatures selected for training:")
count = 1
for feature_name in features:
    print("   ", count, ".", feature_name)
    count = count + 1


# =====================================================================
# Step 3: Split into Train and Test Sets (80% Train, 20% Test)
# =====================================================================
print("\n[3] Splitting data into 80% train and 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("    Training samples:", len(X_train))
print("    Testing samples :", len(X_test))


# =====================================================================
# Step 4: Feature Scaling (StandardScaler)
# =====================================================================
# Neural networks perform best when input features are standardized
print("\n[4] Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# Step 5: Build and Train Artificial Neural Network (ANN)
# =====================================================================
# Architecture:
# - Input Layer   : 6 features
# - Hidden Layer 1: 64 neurons (ReLU)
# - Hidden Layer 2: 32 neurons (ReLU)
# - Output Layer  : 1 neuron (Productivity score)
print("\n[5] Training Artificial Neural Network (MLPRegressor)...")
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    alpha=2650,
    random_state=42,
)
model.fit(X_train_scaled, y_train)
print("    Training completed successfully!")


# =====================================================================
# Step 6: Model Evaluation
# =====================================================================
print("\n[6] Evaluating model on test data...")
accuracy = model.score(X_test_scaled, y_test)

print("-" * 45)
print("    Model Accuracy :", round(accuracy * 100, 2), "%")
print("-" * 45)


# =====================================================================
# Step 7: Save Model and Scaler
# =====================================================================
model_filename = "ann_procrastination_model.pkl"
joblib.dump({"model": model, "scaler": scaler, "features": features}, model_filename)
print("\n[7] Model saved to", model_filename)


# =====================================================================
# Step 8: Interactive User Prediction Loop
# =====================================================================
def get_user_prediction():
    print("\n" + "=" * 60)
    print("      INTERACTIVE DIGITAL PROCRASTINATION PREDICTOR")
    print("=" * 60)
    print("Enter daily habits and digital activity to predict procrastination.")
    print("Type 'exit' at any prompt to stop.\n")

    while True:
        try:
            val = input("1. Study hours per day (0 - 10)       : ").strip()
            if val.lower() in ("exit", "q", "quit"):
                break
            study_hours = float(val)

            val = input("2. Sleep hours per night (0 - 10)     : ").strip()
            if val.lower() in ("exit", "q", "quit"):
                break
            sleep_hours = float(val)

            val = input("3. Phone usage hours per day (0 - 12) : ").strip()
            if val.lower() in ("exit", "q", "quit"):
                break
            phone_hours = float(val)

            val = input("4. Attendance percentage (0 - 100%)   : ").strip()
            if val.lower() in ("exit", "q", "quit"):
                break
            attendance = float(val)

            val = input("5. Stress level (1 - 10)              : ").strip()
            if val.lower() in ("exit", "q", "quit"):
                break
            stress = float(val)

            val = input("6. Focus score (30 - 100)             : ").strip()
            if val.lower() in ("exit", "q", "quit"):
                break
            focus = float(val)

        except ValueError:
            print("\n[Error] Invalid number entered! Please try again.\n")
            continue

        # Prepare input data and scale it
        user_data = pd.DataFrame([[study_hours, sleep_hours, phone_hours, attendance, stress, focus]], columns=features)
        user_data_scaled = scaler.transform(user_data)

        # Predict productivity score
        predicted_score = float(model.predict(user_data_scaled)[0])
        score_display = max(0.0, min(100.0, predicted_score))

        # Calculate procrastination score (Procrastination = 100 - Productivity)
        procrastination_score = round(100.0 - score_display, 1)

        # Determine category and final task verdict
        if procrastination_score <= 35:
            category = "Low"
            verdict = "People WILL do the task on time!"
        elif procrastination_score <= 60:
            category = "Moderate"
            verdict = "People MIGHT delay the task due to distractions!"
        else:
            category = "High"
            verdict = "People will NOT do the task on time (High Procrastination)!"

        print("\n" + "-" * 60)
        print("  Predicted Procrastination Score :", procrastination_score, "% (" + category + ")")
        print("  Final Verdict                   :", verdict)
        print("-" * 60)

        # Ask if user wants another prediction
        again = input("\nDo you want to predict for another person? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("\nThank you for using the Digital Procrastination Predictor!\n")
            break
        print()


if __name__ == "__main__":
    get_user_prediction()