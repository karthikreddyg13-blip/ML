# ============================================
# 1. VARIABLES & TYPES
# ============================================

x = 5
y = 3.14
name = "CGPA"
is_placed = True

print(x, type(x))
print(y, type(y))
print(name, type(name))
print(is_placed, type(is_placed))


# ============================================
# 2. ARITHMETIC OPERATORS
# ============================================

a = 7
b = 2

print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a ** 2 =", a ** 2)


# ============================================
# 3. COMPARISON OPERATORS
# ============================================

cgpa = 8.4

print("cgpa == 8.4 :", cgpa == 8.4)
print("cgpa > 8 :", cgpa > 8)
print("cgpa < 6 :", cgpa < 6)


# ============================================
# 4. LOGICAL OPERATORS
# ============================================

attendance = 85

placed = (cgpa > 8) and (attendance > 80)

print("Placed:", placed)


# ============================================
# 5. STRINGS & F-STRINGS
# ============================================

model_name = "Logistic Regression"
accuracy = 0.87345

print("Accuracy of " + model_name + " is " + str(accuracy))
print(f"Accuracy of {model_name} is {accuracy}")
print(f"Accuracy of {model_name} is {accuracy:.3f}")


# ============================================
# 6. LISTS
# ============================================

features = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

print(features)
print(features[0])
print(features[1])
print(features[-1])
print(len(features))

print(features[0:2])

features.append("Internships")

print(features)


# ============================================
# 7. DICTIONARIES
# ============================================

from sklearn.linear_model import Ridge, Lasso, ElasticNet

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01)
}

print(models)
print(models["Ridge"])

for name, model in models.items():
    print(f"{name} -> {model}")


# ============================================
# 8. IF / ELIF / ELSE
# ============================================

cgpa = 8.2

if cgpa >= 9:
    tier = "High"
elif cgpa >= 7:
    tier = "Mid"
else:
    tier = "Low"

print(f"CGPA {cgpa} belongs to {tier} tier")

missing_count = 0

if missing_count == 0:
    print("No missing values.")
else:
    print("Missing values found.")


# ============================================
# 9. FOR LOOPS
# ============================================

depths = [2, 4, 6, 8, 10]

for depth in depths:
    print(f"Training Decision Tree with max_depth={depth}")

for epoch in range(5):
    print(f"Epoch {epoch}")

for i, d in enumerate(depths):
    print(f"Iteration {i}: depth={d}")


# ============================================
# 10. FUNCTIONS
# ============================================

def classify_cgpa(cgpa):
    if cgpa >= 9:
        return "High"
    elif cgpa >= 7:
        return "Mid"
    else:
        return "Low"


print(classify_cgpa(9.5))
print(classify_cgpa(8.1))
print(classify_cgpa(6.2))


def describe_model(name, accuracy):
    return f"{name} achieved {accuracy:.1%} accuracy"


print(describe_model("Random Forest", 0.913))


# ============================================
# 11. IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Pandas Version:", pd.__version__)
print("NumPy Version :", np.__version__)


# ============================================
# 12. MACHINE LEARNING EXAMPLE
# ============================================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Change this path if your CSV is in another location
csv_path = "placement_predict_50k_Dataset.csv"

min_acceptable_accuracy = 0.70

features = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "AttendancePercent"
]

try:
    df = pd.read_csv(csv_path)

    print("\nDataset Loaded Successfully!\n")

    print("Columns in Dataset:")
    print(df.columns.tolist())

    # Check required columns
    required_columns = features + ["PlacementStatus"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if len(missing_columns) > 0:
        print("\nERROR!")
        print("Missing columns:", missing_columns)

    else:

        # Fill missing values
        df[features] = df[features].fillna(df[features].median())

        X = df[features]
        y = df["PlacementStatus"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        }

        def evaluate_model(model, X_train, y_train, X_test, y_test):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            return accuracy_score(y_test, predictions)

        print("\nMODEL RESULTS\n")

        for model_name, model in models.items():

            acc = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            if acc >= min_acceptable_accuracy:
                verdict = "GOOD"
            else:
                verdict = "NEEDS WORK"

            print(f"{model_name:<22} Accuracy = {acc:.3f} --> {verdict}")

except FileNotFoundError:
    print("\nERROR: CSV file not found.")
    print("Make sure 'placement_predict_50k_Dataset.csv' is in the same folder as this Python file or notebook.")
