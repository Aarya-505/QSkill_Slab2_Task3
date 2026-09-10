# End-to-End Fraud Detection & Risk Analysis System

**QSkill Internship - Task 3**

This repository contains a complete, end-to-end machine learning pipeline for detecting fraudulent financial transactions. The project involves exploratory data analysis (EDA), data cleaning, handling extreme class imbalance, and training multiple classification models to identify potentially fraudulent behavior.

## Dataset
This project uses the **Credit Card Fraud Detection** dataset fetched directly via `scikit-learn` from OpenML (Data ID: 1597). 
* The dataset contains European cardholder transactions from September 2013.
* It features 28 anonymized PCA components (`V1`-`V28`), an `Amount` feature, and a `Class` label (0 = Normal, 1 = Fraud).
* **Imbalance Challenge**: The dataset is highly imbalanced, with frauds accounting for only ~0.17% of all transactions.

## Project Structure
* `fraud_detection_analysis.py`: The main Python script containing the entire ML pipeline.
* `plots/`: Directory containing automatically generated visualizations from the script:
  * `target_distribution.png`: Visualizes the class imbalance.
  * `amount_distribution.png`: Compares transaction amounts between normal and fraudulent classes.
  * `confusion_matrices.png`: Evaluates True Positives, False Positives, False Negatives, and True Negatives across all 3 models.
  * `feature_importance.png`: Displays the top predictive features used by the Random Forest model.

## Methodology
1. **Exploratory Data Analysis (EDA)**: Analyzed feature distributions and quantified the class imbalance.
2. **Data Preprocessing**: Scaled the `Amount` feature using `StandardScaler` and applied a stratified 80/20 Train-Test split.
3. **Handling Class Imbalance**: Applied **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively to the training set to prevent the models from blindly predicting the majority class.
4. **Model Building & Training**: 
   * Logistic Regression (Baseline)
   * Random Forest Classifier
   * HistGradientBoostingClassifier (Optimized Gradient Boosting for large datasets)
5. **Evaluation**: Evaluated on the strictly separated testing set using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.

## Key Business Observations
1. **Precision vs. Recall Trade-off**: The SMOTE technique boosts Recall significantly (catching more fraud) but causes a drop in Precision (resulting in some False Positives). The decision threshold should be adjusted based on business goals (e.g., maximizing fraud caught vs. minimizing customer friction).
2. **Feature Importance**: Features like `V14`, `V4`, and `V12` emerged as highly predictive. Collaboration with data engineers is required to map these PCA components back to their real-world context (location, velocity, device type, etc.).
3. **Model Selection**: The **Random Forest** model performed the best overall, catching 83% of the fraud cases (Recall) while maintaining a highly accurate 85% Precision (minimizing false alarms). It also achieved an outstanding ROC-AUC score of 0.9753.
4. **Actionable Step**: Deploy the Random Forest model in "Shadow Mode" to score live transactions without actively blocking them. Investigate False Positives to manually tune heuristic rules before enabling auto-blocking.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aarya-505/QSkill_Slab2_Task3.git
   cd QSkill_Slab2_Task3
   ```

2. **Install the required dependencies:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn imbalanced-learn
   ```

3. **Run the analysis script:**
   ```bash
   python fraud_detection_analysis.py
   ```
   The script will download the dataset, train the models, output the classification reports to the console, and save all graphs into the `plots/` folder.
