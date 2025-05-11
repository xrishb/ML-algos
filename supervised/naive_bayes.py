# Naive Bayes Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load 20 Newsgroups dataset (subset of categories for faster processing)
print("Loading text dataset (20 Newsgroups)...")
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space'
]

print(f"Selected categories: {categories}")
newsgroups_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True, 
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

newsgroups_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True, 
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

print(f"Training samples: {len(newsgroups_train.data)}")
print(f"Testing samples: {len(newsgroups_test.data)}")

# Convert text to numerical features
print("\nConverting text to numerical features...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

print(f"Features shape after vectorization: {X_train.shape}")

# Feature selection to improve model performance
print("\nPerforming feature selection...")
k_best = 1000  # Select top 1000 features
selector = SelectKBest(chi2, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
print(f"Features shape after selection: {X_train_selected.shape}")

# Try different Naive Bayes variants
print("\nTraining and comparing Naive Bayes variants...")
nb_variants = {
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    # GaussianNB doesn't work well with sparse matrices from text vectorization
}

# Dictionary to store results
results = {}

for name, model in nb_variants.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=newsgroups_train.target_names)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{name} accuracy: {accuracy:.4f}")
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'report': report,
        'conf_matrix': conf_matrix
    }

# Find the best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

# Print detailed report for the best model
print("\nDetailed classification report for the best model:")
print(results[best_model_name]['report'])

# Visualize confusion matrix for the best model
plt.figure(figsize=(10, 8))
cm = results[best_model_name]['conf_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=newsgroups_train.target_names,
            yticklabels=newsgroups_train.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('naive_bayes_confusion_matrix.png')
print("\nConfusion matrix saved as 'naive_bayes_confusion_matrix.png'")

# Show top keywords for each category
print("\nTop discriminative keywords for each category:")
feature_names = np.array(vectorizer.get_feature_names_out())[selector.get_support()]
best_model = results[best_model_name]['model']

# For each class, find the keywords with highest probability
if hasattr(best_model, 'feature_log_prob_'):
    for i, category in enumerate(newsgroups_train.target_names):
        # Get the log probabilities for this class
        log_probs = best_model.feature_log_prob_[i]
        # Get the indices of the top 10 features with highest probability
        top_indices = np.argsort(log_probs)[::-1][:10]
        top_keywords = feature_names[top_indices]
        
        print(f"\nTop keywords for '{category}':")
        for keyword in top_keywords:
            print(f"- {keyword}")

# Plot ROC curve for multi-class classification (one-vs-rest)
plt.figure(figsize=(10, 8))

# Compute ROC curve and ROC area for each class
target_names = newsgroups_train.target_names
y_test_bin = np.zeros((len(y_test), len(target_names)))
for i in range(len(target_names)):
    y_test_bin[:, i] = (y_test == i).astype(int)

y_prob = results[best_model_name]['probabilities']

# Plot ROC curves
for i, class_name in enumerate(target_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves - {best_model_name}')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('naive_bayes_roc_curves.png')
print("\nROC curves saved as 'naive_bayes_roc_curves.png'")

if __name__ == "__main__":
    print("\nNaive Bayes text classification complete!") 