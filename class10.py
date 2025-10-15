# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('pp_sum4.csv', skiprows=3)
X = dataset.iloc[:, 1:5].values
y = pd.to_numeric(dataset.iloc[:, 5], errors='coerce').values  # Convert to numpy array to avoid series issues

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))  # Print the accuracy for visibility

# Visualising the Training set results (using first two features, fixing dimension mismatch and grid steps)
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.5),  # Reasonable step for age-like feature
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1000))  # Reasonable step for salary-like feature
# Create full 4D grid input (set features 2 and 3 to scaled mean=0)
full_grid = np.c_[X1.ravel(), X2.ravel(),
                  np.zeros(len(X1.ravel())),
                  np.zeros(len(X1.ravel()))]
Z = classifier.predict(sc.transform(full_grid)).reshape(X1.shape)
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results (same fixes)
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.5),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1000))
full_grid = np.c_[X1.ravel(), X2.ravel(),
                  np.zeros(len(X1.ravel())),
                  np.zeros(len(X1.ravel()))]
Z = classifier.predict(sc.transform(full_grid)).reshape(X1.shape)
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


import seaborn as sns

# Making the Confusion Matrix (cm must be calculated before this)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize=(6, 4))
labels = np.unique(y_test).astype(int)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False)
plt.title('Confusion Matrix Heatmap (Naive Bayes)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show() # or plt.savefig('confusion_matrix_heatmap.png')