# ANN-Based Digital Procrastination Predictor

An Artificial Neural Network (ANN) project developed using **scikit-learn** to predict digital procrastination levels and determine whether people will complete their tasks on time based on digital habits and daily lifestyle metrics.

---

## 📌 Project Overview
Digital distractions (such as high smartphone usage, irregular sleep, and stress) significantly affect task execution. This project implements a Multi-Layer Perceptron (MLPRegressor) neural network to:
1. Statistically analyze feature correlations with productivity and distraction.
2. Train an Artificial Neural Network with 2 hidden layers on 8,000 behavioral records.
3. Predict a person's **Procrastination Score (0–100%)**.
4. Provide a clear **Final Verdict**: whether people will complete the task on time or delay it.

---

## 📊 Dataset Details
- **File:** `people_procrastination.csv`
- **Total Records:** 8,000 rows
- **Target Variable:** `productivity_score` (converted to `procrastination_score = 100 - productivity_score`)

### Feature Selection & Correlation
Features were selected based on **Pearson's Correlation Coefficient ($r$)** with task productivity:

| Feature Name | Description | Correlation ($r$) | Impact |
| :--- | :--- | :---: | :--- |
| `study_hours_per_day` | Average hours spent studying daily | **+0.73** | Strongest positive factor |
| `focus_score` | Measured concentration score (30–100) | **+0.41** | High focus reduces delay |
| `sleep_hours` | Nightly sleep duration (hours) | **+0.34** | Rest improves discipline |
| `phone_usage_hours` | Daily smartphone screen time (hours) | **-0.33** | Major distraction factor |
| `stress_level` | Self-reported stress scale (1–10) | **-0.20** | High stress induces procrastination |
| `attendance_percentage`| Academic / work attendance rate (%) | **+0.17** | Regularity aids task completion |

*Note: Irrelevant features with near-zero correlation (e.g., `coffee_intake_mg`, `youtube_hours`, `student_id`) were filtered out.*

---

## ⚙️ Data Preprocessing
1. **Feature Separation:** Separated 6 relevant numeric input features ($X$) from the target ($y$).
2. **Train-Test Split:** Partitioned data into **80% Training (6,400 samples)** and **20% Testing (1,600 samples)**.
3. **Standardization (`StandardScaler`):** Features operate on vastly different scales (e.g., phone hours: 0–12 vs attendance: 0–100). Applied Z-score normalization:
   $$\mu = 0, \quad \sigma = 1$$
   This ensures stable and balanced gradient updates during neural network training.

---

## 🧠 Artificial Neural Network (ANN) Architecture
The model is built using `scikit-learn`'s `MLPRegressor`:

```
Input Layer (6 Features)
       │
       ▼
Hidden Layer 1 (64 Neurons, ReLU Activation)
       │
       ▼
Hidden Layer 2 (32 Neurons, ReLU Activation)
       │
       ▼
Output Layer (1 Neuron, Continuous Score)
```

### Hyperparameters:
- **Hidden Layers:** `(64, 32)`
- **Activation Function:** `relu` ($f(x) = \max(0, x)$)
- **Optimization Algorithm:** `adam` (Adaptive Moment Estimation)
- **L2 Regularization (`alpha`):** `2650` (prevents overfitting and ensures realistic generalization)
- **Random State:** `42` (ensures reproducible results)

---

## 📈 Performance & Results
- **Model Accuracy:** **`85.55 %`**
- **Generalization:** Avoids artificial 100% overfitting through L2 penalty tuning.

---

## 🎯 Prediction Logic & Output Verdict
The model predicts a continuous score, which is translated into a realistic decision for task completion:

| Procrastination Score | Category | Final Verdict |
| :---: | :---: | :--- |
| **$\le 35\%$** | **Low** | **People WILL do the task on time!** |
| **$36\% - 60\%$** | **Moderate** | **People MIGHT delay the task due to distractions!** |
| **$> 60\%$** | **High** | **People will NOT do the task on time (High Procrastination)!** |

---

## 📁 Repository Structure
```
AI_OEP/
│
├── ann_project.py                 # Main Python script (training, evaluation & CLI)
├── people_procrastination.csv     # Preprocessed 8,000-row dataset
├── ann_procrastination_model.pkl  # Serialized model and scaler artifact
├── .gitignore                     # Ignored files and directories
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Execution
Run the complete pipeline from terminal:
```bash
python ann_project.py
```

### 3. Interactive CLI Example
When prompted in the console, enter daily habits to get an instant prediction:
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
Type `exit` or `q` at any prompt to exit the program.
