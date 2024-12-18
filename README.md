# Predicting Bankruptcy

### Overview
This project builds a predictive model to classify whether a company is likely to go bankrupt within 5 years based on its financial ratios. The dataset contains financial information for multiple companies, and the goal is to develop a robust classifier while addressing challenges like class imbalance and feature importance.

---

### Features
- Classification of companies into "Bankrupt" or "Non-Bankrupt".
- Analysis of feature importance and impact on predictions.
- Evaluation of model performance with imbalanced datasets.
- Comparison of multiple machine learning models (Logistic Regression, Random Forest, Gradient Boosting).
- Hyperparameter tuning to improve accuracy.

---

### Approach
1. **Data Preprocessing**:
   - Handling duplicates and missing values.
   - Scaling numerical features.
   - Dimensionality reduction using PCA for multicollinear features.

2. **Model Building**:
   - Logistic Regression as a baseline model.
   - Advanced models: Random Forest and Gradient Boosting.
   - Evaluation with metrics such as accuracy, precision, recall, and AUC.

3. **Key Challenges**:
   - Addressing class imbalance using stratified sampling.
   - Visualizing bankruptcy rates using attribute bucketing.

4. **Evaluation**:
   - Confusion Matrix.
   - ROC Curve.
   - Feature importance visualization.

---

### Results
- Logistic Regression achieved **95% accuracy** on the test set.
- Gradient Boosting outperformed other models with optimized hyperparameters.
- Identified key financial ratios driving bankruptcy predictions.

---

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Predicting_Bankruptcy.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage
1. Place the dataset in the `data/` folder.
2. Run the script:
   ```bash
   python src/bankruptcy_prediction.py
   ```
3. Review the outputs, including feature importance and model metrics.

---

### Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

### Visual Outputs
Include visual outputs like:
1. ROC Curve.
2. Feature importance chart.
3. Attribute bucketing and bankruptcy rate plot.

---

### License
This project is licensed under the MIT License.
