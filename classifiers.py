# Import necessary libraries and modules
import numpy as np  # Import NumPy library
import matplotlib.pyplot as plt  # Import pyplot module from Matplotlib
from sklearn.datasets import make_moons  # Import make_moons function from scikit-learn
from sklearn.model_selection import train_test_split  # Import train_test_split function from scikit-learn
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression class from scikit-learn
from sklearn.svm import SVC  # Import SVC (Support Vector Classifier) class from scikit-learn
from sklearn.metrics import accuracy_score  # Import accuracy_score function from scikit-learn

def plot_decision_boundary(clf, X, X_train, y_train, X_test, y_test, ax, title):
    # Create grid for decision boundary plot
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    )# np.linspace creates evenly spaced values over a specified range, x[:, 0] means all rows from first column, .min() minimum value from feature 1

    # Predict for each grid point
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot data points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')

    # Draw decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3)

    # Set plot settings
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)


# Step 1: Create dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)  # Generate dataset with two opposing moon shapes using make_moons

# Step 2: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split dataset into training and test sets

# Step 3: Train Logistic Regression classifier
log_reg = LogisticRegression(random_state=42)  # Initialize logistic regression classifier
log_reg.fit(X_train, y_train)  # Fit model to training data
log_reg_train_accuracy = accuracy_score(y_train, log_reg.predict(X_train))  # Calculate training accuracy
log_reg_test_accuracy = accuracy_score(y_test, log_reg.predict(X_test))  # Calculate test accuracy

# Step 4: Train default SVM classifier
svm_clf = SVC(random_state=42)  # Initialize SVM classifier
svm_clf.fit(X_train, y_train)  # Fit model to training data
svm_train_accuracy = accuracy_score(y_train, svm_clf.predict(X_train))  # Calculate training accuracy
svm_test_accuracy = accuracy_score(y_test, svm_clf.predict(X_test))  # Calculate test accuracy

# Step 5: Plots
# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 12))  # Create two vertically arranged subplots

# Plot decision boundary for Logistic Regression
plot_decision_boundary(
    clf=log_reg,
    X=X,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    ax=axes[0],
    title=f"Logistic Regression\nTrain accuracy: {log_reg_train_accuracy:.4f}, Test accuracy: {log_reg_test_accuracy:.4f}"
)

# Plot decision boundary for SVM
plot_decision_boundary(
    clf=svm_clf,
    X=X,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    ax=axes[1],
    title=f"SVM\nTrain accuracy: {svm_train_accuracy:.4f}, Test accuracy: {svm_test_accuracy:.4f}"
)

plt.tight_layout()  # Adjust subplot layout
plt.legend()  # Add legend
plt.savefig("02_log_reg_svm.png")  # Save plot to file
plt.show()  # Display plot

"""
Selected SVC parameters:
C (penalty parameter): Determines penalty weight for classification errors in training data.
Default value is 1. Higher values may lead to more complex models that fit training data better,
but could result in overfitting.

Kernel: Specifies type of kernel function used in SVM algorithm.
Available options: 'linear', 'poly', 'rbf', 'sigmoid', etc.
Kernel choice affects model's ability to fit data.

gamma: Parameter for 'rbf', 'poly' and 'sigmoid'. Controls influence radius of training points.
Low values mean larger reach (points influence wider area),
high values mean closer relationships between training points.

degree for 'poly': Polynomial degree for 'poly'. Default is 3.

class_weight: Specifies class weights used during training.
Can be useful for handling imbalanced classes.
"""