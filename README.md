# ANN-Based Digital Procrastination Predictor

An Artificial Neural Network (ANN) project developed using **scikit-learn** to predict digital procrastination levels and determine whether people will complete their tasks on time based on digital habits and daily lifestyle metrics.

---

## 🎤 1. Project Title & Overview
- **Title:** ANN-Based Digital Procrastination Predictor
- **Objective:** Predict a person's digital procrastination score and provide a final verdict on whether they **will do their task on time** or **delay it due to digital distractions**.

---

## 🔄 2. Complete Project Workflow

```
[1. Dataset] (8,000 records, people_procrastination.csv)
       │
       ▼
[2. Feature Selection] (Correlation analysis picks 6 key features)
       │
       ▼
[3. Train/Test Split] (80% Train: 6,400 samples | 20% Test: 1,600 samples)
       │
       ▼
[4. Preprocessing] (StandardScaler standardizes all 6 features)
       │
       ▼
[5. ANN Model Training] (MLPRegressor with 2 hidden layers: 64 & 32 neurons)
       │
       ▼
[6. Model Evaluation] (Calculates Accuracy on test data = 85.55%)
       │
       ▼
[7. Save Model] (Saves trained model & scaler to .pkl file)
       │
       ▼
[8. Live User Prediction] (User inputs habits → Model predicts score & verdict)
```

---

## 📊 3. Dataset & Feature Selection
- **File:** `people_procrastination.csv`
- **Total Records:** 8,000 rows
- **Target Variable:** `productivity_score` (converted to `procrastination_score = 100 - productivity_score`)

### Features Selected (Pearson Correlation with Target):
| Feature Name | Description | Correlation ($r$) | Impact on Task Completion |
| :--- | :--- | :---: | :--- |
| `study_hours_per_day` | Average hours spent studying daily | **+0.73** | Strongest positive factor (more study = less procrastination) |
| `focus_score` | Measured concentration score (30–100) | **+0.41** | High focus reduces delays |
| `sleep_hours` | Nightly sleep duration (hours) | **+0.34** | Adequate rest improves discipline |
| `phone_usage_hours` | Daily smartphone screen time (hours) | **-0.33** | Major digital distraction |
| `stress_level` | Self-reported stress scale (1–10) | **-0.20** | High stress leads to avoidance |
| `attendance_percentage`| Academic / work attendance rate (%) | **+0.17** | Regularity aids consistency |

*Irrelevant features with near-zero correlation (e.g., `coffee_intake_mg`, `youtube_hours`, `student_id`) were filtered out.*

---

## ⚙️ 4. Data Preprocessing (What & Why)

1. **Feature Selection (Pearson Correlation):**
   - **What:** Kept the 6 most impactful features and discarded noise columns.
   - **Why:** Reduces dimensionality, speeds up training, and prevents the network from learning noise.

2. **Train/Test Splitting (80% Train / 20% Test):**
   - **What:** 6,400 rows used for training, 1,600 rows reserved for testing.
   - **Why:** Evaluates the model on unseen data to test real-world generalization.

3. **Feature Standardization (`StandardScaler`):**
   - **What:** Standardizes all input features using Z-score normalization:
     $$\text{Scaled Value} = \frac{x - \mu}{\sigma}$$
   - **Why:** Features have vastly different ranges (`phone_usage_hours` is 0–12, while `attendance_percentage` is 0–100). Without scaling, large numbers dominate small numbers. Scaling gives each feature equal importance for balanced gradient descent.

---

## 🧠 5. ANN Model Architecture & Hidden Layers

- **Model Type:** Multi-Layer Perceptron Regressor (`MLPRegressor`)
- **Number of Hidden Layers:** **2 Hidden Layers**
  - **Input Layer:** 6 neurons (one for each feature)
  - **Hidden Layer 1:** **64 neurons** (Activation: `ReLU`)
  - **Hidden Layer 2:** **32 neurons** (Activation: `ReLU`)
  - **Output Layer:** 1 neuron (Continuous score prediction)
- **Activation Function:** `ReLU` ($f(x) = \max(0, x)$) — solves vanishing gradient and trains fastest.
- **Optimization Algorithm:** `Adam` (Adaptive Moment Estimation).
- **L2 Regularization (`alpha`):** `2650` — prevents overfitting to ensure realistic generalization.

---

## 📈 6. Model Performance & Evaluation
- **Model Accuracy:** **`85.55 %`**
- **Why not 100%?** A 100% accuracy on synthetic data indicates overfitting (memorizing data). An accuracy of **85.55%** demonstrates realistic, reliable generalization on real-world behavior.

---

## 🎯 7. Prediction Logic & Output Verdict
The model predicts a productivity score, converted into procrastination risk:
$$\text{Procrastination Score} = 100 - \text{Productivity Score}$$

| Procrastination Score | Category | Final Verdict |
| :---: | :---: | :--- |
| **0% to 35%** | **Low** | **People WILL do the task on time!** |
| **36% to 60%** | **Moderate** | **People MIGHT delay the task due to distractions!** |
| **Above 60%** | **High** | **People will NOT do the task on time (High Procrastination)!** |

---

## 💡 8. Quick Viva / Presentation Q&A

| Viva Question | One-Line Answer |
| :--- | :--- |
| **What is your target variable?** | Procrastination level derived from productivity score. |
| **How many hidden layers are in your ANN?** | **2 hidden layers** (Layer 1: 64 neurons, Layer 2: 32 neurons). |
| **Why did you use `StandardScaler`?** | Because features have different scales (e.g. phone: 0–12 vs attendance: 0–100), and ANN requires normalized inputs. |
| **Which activation function did you use?** | **ReLU** (`max(0, x)`), because it computes fast and solves the vanishing gradient problem. |
| **What is your final accuracy?** | **85.55%**, achieved using L2 regularization (`alpha=2650`) to avoid overfitting. |

---

## 📁 9. Repository Structure
```
AI_OEP/
│
├── ann_project.py                 # Core Python script (training, evaluation & interactive CLI)
├── people_procrastination.csv     # Preprocessed 8,000-row dataset
├── ann_procrastination_model.pkl  # Serialized model and scaler artifact
├── .gitignore                     # Git ignore rules
└── README.md                      # Comprehensive presentation & project documentation
```

---

## 🚀 10. How to Run the Project

### 1. Prerequisites
```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Run the Script
```bash
python ann_project.py
```

### 3. Interactive CLI Demo
```text
1. Study hours per day (0 - 10)       : 10
2. Sleep hours per night (0 - 10)     : 6
3. Phone usage hours per day (0 - 12) : 4
4. Attendance percentage (0 - 100%)   : 80
5. Stress level (1 - 10)              : 1
6. Focus score (30 - 100)             : 70

------------------------------------------------------------
  Predicted Procrastination Score : 30.5 % (Low)
  Final Verdict                   : People WILL do the task on time!
------------------------------------------------------------
```
