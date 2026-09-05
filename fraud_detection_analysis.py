import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Create an output directory for plots
os.makedirs('plots', exist_ok=True)

print("="*50)
print("1. Fetching the Credit Card Fraud dataset...")
print("This might take a minute...")
# Use as_frame=True to get a pandas DataFrame
df_X, df_y = fetch_openml(data_id=1597, return_X_y=True, as_frame=True, parser='auto')
df = pd.concat([df_X, df_y], axis=1)
df['Class'] = df['Class'].astype(int)
print(f"Dataset loaded! Shape: {df.shape}")

print("\n"+"="*50)
print("2. Exploratory Data Analysis (EDA)")
class_counts = df['Class'].value_counts()
fraud_rate = class_counts[1] / len(df) * 100
print(f"Normal transactions: {class_counts[0]}")
print(f"Fraud transactions: {class_counts[1]}")
print(f"Fraud Percentage: {fraud_rate:.3f}%")

# Target Distribution Plot
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Class')
plt.title('Target Variable Distribution (0: Normal, 1: Fraud)')
plt.savefig('plots/target_distribution.png')
plt.close()
print("Saved plots/target_distribution.png")

# Distribution of Amount
plt.figure(figsize=(9, 5))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', alpha=0.5, label='Normal')
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', alpha=0.5, label='Fraud')
plt.title('Distribution of Transaction Amount')
plt.yscale('log')
plt.legend()
plt.savefig('plots/amount_distribution.png')
plt.close()
print("Saved plots/amount_distribution.png")

print("\n"+"="*50)
print("3. Data Cleaning & Preprocessing")
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

print("\n"+"="*50)
print("4. Handling Class Imbalance (SMOTE)")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE - Normal: {sum(y_train==0)}, Fraud: {sum(y_train==1)}")
print(f"After SMOTE  - Normal: {sum(y_train_sm==0)}, Fraud: {sum(y_train_sm==1)}")

print("\n"+"="*50)
print("5. Model Building & Training")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
}

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_sm, y_train_sm)
    trained_models[name] = model

print("\n"+"="*50)
print("6. Model Evaluation & Comparison")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('plots/confusion_matrices.png')
plt.close()
print("\nSaved plots/confusion_matrices.png")

print("\n"+"="*50)
print("7. Feature Importance (Random Forest)")
rf_model = trained_models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()
print("Saved plots/feature_importance.png")

print("\n"+"="*50)
print("8. Business Observations & Recommendations")
print("""
1. Precision vs. Recall Trade-off: The SMOTE technique boosts Recall significantly but causes a drop in Precision (resulting in some False Positives). Adjust the decision threshold based on business goals.
2. Feature Importance: Features like V14, V4, and V12 are highly predictive. Since these are PCA components, collaborate with data engineers to map them back to real-world context (location, device type, etc.).
3. Model Selection: Gradient Boosting and Random Forest generally outperform Logistic Regression in managing complex, non-linear patterns in the data.
4. Actionable Step: Deploy the model in "Shadow Mode" to score live transactions without blocking them. Investigate False Positives to manually tune rules before enabling auto-blocking.
""")
print("="*50)
print("Analysis complete! All plots have been saved to the 'plots' directory.")
