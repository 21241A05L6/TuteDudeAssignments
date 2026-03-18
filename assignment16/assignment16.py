import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# CREATE DATASET (SELF-CONTAINED)
# -----------------------------
np.random.seed(0)

data = {
    "feature1": np.random.randint(1,100,200),
    "feature2": np.random.randint(1,100,200),
    "feature3": np.random.randint(1,100,200),
}

df = pd.DataFrame(data)

# classification target
df["target"] = (df["feature1"] + df["feature2"] > 100).astype(int)

df.to_csv("dataset.csv", index=False)

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# TASK 1 — SVM
# -----------------------------
print("\nTASK 1 — SVM")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm_linear = SVC(kernel="linear")
svm_linear.fit(X_train, y_train)
pred_linear = svm_linear.predict(X_test)

svm_rbf = SVC(kernel="rbf")
svm_rbf.fit(X_train, y_train)
pred_rbf = svm_rbf.predict(X_test)

print("Linear Accuracy:", accuracy_score(y_test, pred_linear))
print("RBF Accuracy:", accuracy_score(y_test, pred_rbf))

# -----------------------------
# TASK 2 — DECISION TREE
# -----------------------------
print("\nTASK 2 — DECISION TREE")

tree_small = DecisionTreeClassifier(max_depth=2)
tree_small.fit(X_train, y_train)

tree_big = DecisionTreeClassifier()
tree_big.fit(X_train, y_train)

print("Small Tree Train:", tree_small.score(X_train, y_train))
print("Small Tree Test:", tree_small.score(X_test, y_test))

print("Big Tree Train:", tree_big.score(X_train, y_train))
print("Big Tree Test:", tree_big.score(X_test, y_test))

plt.figure(figsize=(10,5))
plot_tree(tree_small, filled=True)
plt.title("Decision Tree")
plt.show()

# -----------------------------
# TASK 3 — TRAIN / VALIDATION / TEST
# -----------------------------
print("\nTASK 3 — TRAIN VALIDATION TEST")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

print("Validation Accuracy:", model.score(X_val, y_val))
print("Test Accuracy:", model.score(X_test, y_test))

# -----------------------------
# TASK 4 — CROSS VALIDATION
# -----------------------------
print("\nTASK 4 — CROSS VALIDATION")

cv_scores = cross_val_score(model, X, y, cv=5)
print("CV Scores:", cv_scores)
print("Average CV:", cv_scores.mean())

# -----------------------------
# TASK 5 — BAGGING vs BOOSTING
# -----------------------------
print("\nTASK 5 — BAGGING vs BOOSTING")

bag = BaggingClassifier()
bag.fit(X_train, y_train)

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)

print("Bagging Accuracy:", bag.score(X_test, y_test))
print("Boosting Accuracy:", ada.score(X_test, y_test))

# -----------------------------
# TASK 6 — RANDOM FOREST
# -----------------------------
print("\nTASK 6 — RANDOM FOREST")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print("Random Forest Accuracy:", rf.score(X_test, y_test))

print("Feature Importance:", rf.feature_importances_)

# Compare with single tree
print("Decision Tree Accuracy:", tree_big.score(X_test, y_test))
print("Bagging Accuracy:", bag.score(X_test, y_test))
