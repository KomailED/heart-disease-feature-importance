# ❤️ Heart Disease Prediction — Decision Tree Feature Importance Analysis

A `DecisionTreeClassifier` trained to predict the presence of heart disease from a patient's clinical attributes — with the primary goal not being raw accuracy, but **interpreting which clinical measurements the model actually relies on** via feature importance analysis.

> Built as an exercise for **Session 6** of the Machine Learning course at [**Mojtame Fanni Tehran (MFT)**](https://mftplus.com).

---

## 🧠 How It Works

1. **Load & inspect** the raw dataset: shape, dtypes, summary statistics, and a missing-value check to confirm the data is clean before analysis.
2. **Exploratory Data Analysis:**
   - Target balance (presence vs. absence of heart disease) — confirms the classes are roughly balanced, making accuracy a reasonable evaluation metric.
   - Distribution of categorical clinical attributes (chest pain type, fasting blood sugar, thalassemia, etc.) and numerical attributes (age, resting blood pressure, cholesterol, max heart rate, ST depression).
   - A correlation heatmap across all features, kept as a reference point to later compare against the tree's own feature importance ranking.
3. **Preprocessing:** cast columns to memory-efficient dtypes (`uint8`/`uint16`/`float32`), split into `X`/`y`, then an 80/20 **stratified** train/test split (`stratify=y`) to preserve class balance in both sets. No feature scaling is applied — Decision Trees split on per-feature thresholds independently, so they're insensitive to feature magnitude.
4. **Baseline model:** a lightly-constrained `DecisionTreeClassifier` (`max_depth=4`) is trained and visualized first, giving an early intuitive read on which features the tree splits on near the root — and a performance reference point for later tuning.
5. **Hyperparameter tuning:** `GridSearchCV` (5-fold CV, `scoring="accuracy"`) searches over split criterion, depth, leaf/split sample thresholds, feature sampling strategy, and minimum impurity decrease. `class_weight="balanced"` ensures both classes contribute equally to split decisions.
6. **Evaluate the tuned model** on train and test sets via `classification_report`, comparing against the baseline to confirm tuning improved generalization without introducing overfitting.
   - **Test accuracy:** ~83% (F1 ≈ 0.83–0.84 for both classes)
   - **Train accuracy:** ~91% — the gap suggests reasonable generalization without severe overfitting.
7. **Extract feature importance** (`feature_importances_`) — each feature's normalized contribution to impurity reduction across every split where it was used — and visualize it as a bar chart.
8. **Visualize the final tuned tree structure**, making the importance ranking concrete: the root and near-root nodes correspond to the highest-ranked features.

### Key findings

**`cp`** (chest pain type, ~0.26), **`age`** (~0.17), and **`thalach`** (max heart rate achieved, ~0.12) stand out as the strongest predictors, followed by **`thal`** (thalassemia result, ~0.10) and **`ca`** (number of major vessels, ~0.08) — consistent with clinical intuition. **`restecg`** contributes essentially nothing, and **`trestbps`** very little, suggesting their signal is largely redundant with other features once the tree has split on them. The notebook notes that Decision Tree importance can be unstable across correlated features/random seeds, and suggests cross-checking against Random Forest or permutation importance as a natural next step.

## 📊 Dataset

Not included in this repo. Expected as `heart.csv` in the project root — downloaded separately from Kaggle: [`johnsmith88/heart-disease-dataset`](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

## 🛠️ Tech Stack

- Python 3
- [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — data handling
- [scikit-learn](https://scikit-learn.org/) — `DecisionTreeClassifier`, `GridSearchCV`, `Pipeline`, `train_test_split`, metrics
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) — visualization
- Jupyter Notebook

## 📁 Project Structure

```
Heart Disease Feature Importance/
├── heart_disease_feature_importance.ipynb   # Main notebook: EDA, preprocessing, tuning, evaluation, feature importance
├── heart.csv                                # Dataset (download separately from Kaggle — not included)
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook / JupyterLab (or VS Code with the Jupyter extension)
- A [Kaggle account](https://www.kaggle.com/) to download the dataset

### Installation

```bash
git clone https://github.com/<your-username>/heart-disease-feature-importance.git
cd heart-disease-feature-importance
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Download [`heart.csv`](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) from Kaggle and place it in the project root before running the notebook.

### Usage

1. Open `heart_disease_feature_importance.ipynb` in Jupyter or VS Code.
2. Run all cells in order.
3. Review the feature importance bar chart (Section 8) and the final decision tree visualization (Section 9) for the core results.

## 📌 Notes & Possible Improvements

- Feature importance from a single Decision Tree can be unstable when features are correlated; comparing against a **Random Forest** or **permutation importance** would help confirm how robust this ranking really is.
- `GridSearchCV` here optimizes for accuracy only — adding `scoring="f1"` or `"roc_auc"` as alternatives (especially useful if the class balance shifts) would be a reasonable extension.
- Could add SHAP values for a more granular, per-prediction explanation on top of the global feature importance.

## 📄 License

This project is for educational purposes as part of the MFT Machine Learning course.

## 🙏 Acknowledgements

- Dataset: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (Kaggle, by John Smith)
- Course: Machine Learning — Session 6, [Mojtame Fanni Tehran (MFT)](https://mftplus.com)
