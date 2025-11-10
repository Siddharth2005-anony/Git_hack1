   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------- 1) Load & Clean ----------
# Update the path if your file is elsewhere
CSV_PATH = "pp_sum4.csv"

# The first 3 rows contain metadata; skip them
df = pd.read_csv(CSV_PATH, skiprows=3)

# Drop unnamed/meta columns if any
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop columns that are entirely NaN
df = df.dropna(axis=1, how="all")

# (Optional) Keep only the columns we need
# RD and NRD are mostly NaN in this dataset; drop them if present
for col in ["RD", "NRD"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Separate features and target
if "time" in df.columns:
    X = df.drop(columns=["time", "Y/N"])
else:
    X = df.drop(columns=["Y/N"])
y = df["Y/N"].astype(int)

# ---------- 2) Scale & PCA (2 components) ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)
print("Total explained variance by first 2 PCs:", np.sum(pca.explained_variance_ratio_))

# ---------- 3) Plot PCA scatter ----------
plt.figure(figsize=(8, 6))
# Simple matplotlib scatter (no seaborn, no explicit colors)
for label in np.unique(y):
    idx = (y.values == label)
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", alpha=0.8)
plt.title("PCA of Features (2 Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_features.png", dpi=150)
# If you run as a script, comment next line; in notebooks you can show
# plt.show()

# ---------- 4) Train/Test split on 2 PCA features ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5) SVC on top 2 PCA components ----------
svc = SVC(kernel="rbf", random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# ---------- 6) Metrics ----------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n=== SVC Performance on 2 PCA Features ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Optional: visualize confusion matrix with matplotlib only
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - SVC on PCA Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
# Annotate counts
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
# plt.show()
