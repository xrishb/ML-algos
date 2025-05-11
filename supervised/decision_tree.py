# Decision Tree Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Load the iris dataset
print("Loading Iris dataset...")
data = load_iris()

# Create a DataFrame with the features
feature_names = data.feature_names
X = pd.DataFrame(data.data, columns=feature_names)
y = pd.Series(data.target, name='target')

# Map target names for better readability
target_names = data.target_names
y_names = pd.Series([target_names[i] for i in y], name='species')

# Show dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(target_names)}")
print(f"Class distribution: {pd.Series(y_names).value_counts().to_dict()}")

print("\nData preview:")
preview_df = X.copy()
preview_df['species'] = y_names
print(preview_df.head())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train the decision tree model
print("\nTraining Decision Tree model...")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

print("\nFeature Importance:")
for feature, importance in zip(feature_names, model.feature_importances_):
    print(f"- {feature}: {importance:.4f}")

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=feature_names, 
          class_names=target_names, rounded=True, fontsize=10)
plt.title("Decision Tree for Iris Classification")
plt.tight_layout()
plt.savefig('decision_tree_visualization.png')
print("\nDecision tree visualization saved as 'decision_tree_visualization.png'")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm_display = confusion_matrix(y_test, y_pred, display_labels=target_names)
plt.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.xlabel('Predicted')
plt.ylabel('True')

# Add text annotations to the confusion matrix
thresh = cm_display.max() / 2
for i in range(cm_display.shape[0]):
    for j in range(cm_display.shape[1]):
        plt.text(j, i, format(cm_display[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm_display[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('decision_tree_confusion_matrix.png')
print("Confusion matrix saved as 'decision_tree_confusion_matrix.png'")

if __name__ == "__main__":
    print("\nDecision tree analysis complete!") 