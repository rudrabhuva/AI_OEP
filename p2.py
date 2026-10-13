
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# 1. Load Dataset
df = pd.read_csv("student_productivity_distraction_dataset_20000.csv")

target = "productivity_score"

df = df.dropna(subset=[target]).copy()
df = df.drop_duplicates()

# 2. Remove ID and unnecessary identifier columns
id_columns = [
    col for col in df.columns
    if col.lower() == "student_id"
    or col.lower().endswith("_id")
]

df = df.drop(columns=id_columns, errors="ignore")

# 3. Separate input and target
X = df.drop(columns=[target])
y = df[target].astype(float)

# Remove columns that are completely empty or constant
X = X.loc[:, X.nunique(dropna=False) > 1]
X = X.dropna(axis=1, how="all")

# Remove rows with missing input values for this simple pipeline
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 4. Train-test split BEFORE feature selection
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Correlation heatmap using training data only
numeric_train = X_train.select_dtypes(include=np.number).copy()
numeric_train[target] = y_train

correlations = numeric_train.corr()[target].drop(target)
correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()

plt.figure(figsize=(8, max(5, len(correlations) * 0.35)))

sns.heatmap(
    correlations.to_frame(),
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)

plt.title("Feature Correlation with Productivity Score")
plt.tight_layout()
plt.show()

print("\nCorrelation with Target:")
print(correlations.sort_values(key=abs, ascending=False))

# 6. Feature selection using absolute correlation threshold
threshold = 0.10

selected_numeric = correlations[
    correlations.abs() >= threshold
].index.tolist()

# Retain non-ID categorical columns because numeric correlation
# cannot directly measure their relationship with the target.
categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

selected_features = list(dict.fromkeys(
    selected_numeric + categorical_columns
))

if not selected_features:
    raise ValueError(
        "No features selected. Reduce the correlation threshold."
    )

print("\nSelected Features:")
print(selected_features)

removed_features = [
    col for col in X.columns
    if col not in selected_features
]

print("\nRemoved Features:")
print(removed_features)

X_train = X_train[selected_features]
X_test = X_test[selected_features]

# 7. Preprocessing and standardization
numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 8. Configure ANN
hidden_layers = (64, 32)
learning_rate = 0.001
epochs = 300

model = Pipeline([
    ("preprocessor", preprocessor),
    ("ann", MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        max_iter=epochs,
        activation="relu",
        solver="adam",
        early_stopping=True,
        random_state=42
    ))
])

# 9. Train model
model.fit(X_train, y_train)

# 10. Predict numerical score
y_pred = model.predict(X_test)

print("\n========== ANN PERFORMANCE ==========")
print("R2 Score:", round(r2_score(y_test, y_pred), 4))
print("R2 Percentage:", round(r2_score(y_test, y_pred) * 100, 2), "%")

# 11. Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="r2"
)

print("\n========== 5-FOLD CROSS VALIDATION ==========")
print("Fold R2 Scores:", np.round(cv_scores, 4))
print("Mean CV R2:", round(cv_scores.mean(), 4))

# 12. Z-score outlier analysis on target
target_z = (y - y.mean()) / y.std(ddof=0)

print("\n========== Z-SCORE ANALYSIS ==========")
print("Target outliers (|z| > 3):", int((target_z.abs() > 3).sum()))

# 13. Actual vs predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Productivity Score")
plt.ylabel("Predicted Productivity Score")
plt.title("ANN: Actual vs Predicted Score")
plt.tight_layout()
plt.show()

# 14. Save model and selected features
joblib.dump({
    "model": model,
    "features": selected_features
}, "ann_procrastination_model.pkl")

print("\nModel saved successfully!")

# 15. Interactive prediction
print("\n========== PRODUCTIVITY SCORE PREDICTOR ==========")
print("Enter the selected feature values. Type exit to stop.")

while True:
    values = {}

    try:
        for feature in selected_features:
            value = input(f"Enter {feature}: ")

            if value.strip().lower() == "exit":
                raise KeyboardInterrupt

            if feature in categorical_features:
                values[feature] = value
            else:
                values[feature] = float(value)

        student = pd.DataFrame([values], columns=selected_features)

        score = model.predict(student)[0]

        print("Predicted Productivity Score:", round(float(score), 2))
        print()

    except KeyboardInterrupt:
        print("\nPrediction stopped.")
        break

    except ValueError:
        print("Invalid input. Enter a valid value.\n")