# 🔬 Breast Cancer Classification using Logistic Regression

This project applies logistic regression to classify tumors as **benign** or **malignant** based on the Breast Cancer Wisconsin dataset. It includes data preprocessing, visualization, model training, evaluation, ROC curve analysis, threshold tuning, and a mathematical explanation of the sigmoid function.

---

## 📥 Import Necessary Libraries

This project uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `plotly`

---

## 🧹 Data Preprocessing

- Dropped unnecessary columns: `id`, `Unnamed: 32`
- Verified no missing values
- Visualized class distribution

---

## 📊 Data Visualization

- Class balance (`diagnosis`: Benign vs Malignant)
- Feature distributions
- Correlation heatmap
- Scatter plots and histograms

---

## 🔁 Feature Selection

Removed less relevant features based on correlation matrix and domain understanding.

---

## 🧠 Model Training

Trained a **Logistic Regression** model on 60% of the dataset and tested on 40%.

```python
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
```

---

## ✅ Model Evaluation

### 🎯 Accuracy
Evaluated using:
- Accuracy
- Precision
- Recall
- Confusion Matrix

### 📉 Confusion Matrix

Visualized using `ConfusionMatrixDisplay` from `sklearn`.

### 🎯 Precision & Recall

Used `precision_score()` and `recall_score()` with `pos_label='M'` for malignant detection.

---

## 📈 ROC Curve & AUC Score

Plotted using `roc_curve()` and evaluated with `roc_auc_score()` for binary classification.

---

## 📉 Threshold Tuning

By default, logistic regression uses a threshold of **0.5** to classify outputs. You can tune this threshold to **balance precision and recall**. 

Threshold tuning helps in domains like **medical diagnosis**, where:
- High **recall** = catch more cancer cases (sensitive test)
- High **precision** = reduce false positives (specific test)

A loop across thresholds (`0.0` to `1.0`) calculates precision and recall to help select an optimal threshold.

---

## 📘 Sigmoid Function in Logistic Regression

A logistic regression model is designed to output values between **0 and 1**, which are interpreted as **probabilities**. This is made possible by a mathematical function called the **sigmoid function**, also known as the **logistic function**.

### 🔢 Sigmoid Function Formula

The sigmoid function is defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where:

- \( \sigma(z) \) is the output (a value between 0 and 1),
- \( z \) is the input to the function (typically the linear combination of features and weights),
- \( e \) is Euler's number (approximately 2.718).

This function maps any real-valued number into the (0, 1) interval, making it ideal for **probability prediction** in binary classification.

---

## 🏁 Conclusion

- Logistic Regression provides interpretable probability-based predictions.
- Visualizations and metric evaluations helped assess model performance.
- Threshold tuning and sigmoid understanding support better classification decisions in real-world applications.
