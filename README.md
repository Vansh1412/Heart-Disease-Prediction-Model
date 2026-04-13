# ❤️ Heart Disease Prediction Model

End-to-end data science and machine learning project for heart disease risk prediction using the notebook:

- `Heart_Disease_Pipeline.ipynb`

The notebook covers full analysis from data loading and EDA to model training, evaluation, feature importance, risk scoring, and cross-validation.

## Repository Contents

- `Heart_Disease_Pipeline.ipynb` — complete analysis and ML pipeline
- `heart_dataset.csv` — dataset used in the notebook
- `plots/` — exported visualizations from the analysis
- `LICENSE`

## Dataset Overview

- Records: **12,688**
- Columns: **28**
- Target column: `Target` (0 = No Disease, 1 = Heart Disease)

Main feature groups used in the notebook:
- Demographic: age, sex
- Clinical: chest pain type, blood pressure, cholesterol, max heart rate, ST depression, major vessels, thalassemia, etc.
- Lifestyle: smoking status, alcohol consumption, exercise level, BMI category
- Administrative/time: patient/hospital/doctor IDs and visit date (used for analysis, dropped for ML training)

## Notebook Workflow

The notebook is organized into 19 sections:

1. Environment setup and imports
2. Dataset loading and overview
3. Numeric feature distributions
4. Categorical features vs target
5. Correlation heatmap
6. Age/gender/risk-level analysis
7. Boxplots for clinical variables
8. Temporal analysis of visits
9. Hospital and doctor analysis
10. Lifestyle vs disease analysis
11. Preprocessing and feature engineering
12. Training six classifiers
13. Confusion matrices
14. ROC and Precision-Recall curves
15. Model comparison
16. Feature importance
17. Clinical distributions (post-model view)
18. Rule-based clinical risk scoring engine
19. 5-fold stratified cross-validation summary

## Models Trained

The notebook trains and compares these classifiers:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF)

## Preprocessing Used

- Drops leakage/admin columns (`Patient_ID`, `Hospital_ID`, `Doctor_ID`, date-derived fields, and derived labels)
- Fills missing values in `Alcohol_Consumption` using mode
- Label-encodes categorical object columns
- Train/test split with stratification
- Standard scaling for scale-sensitive models

## Visual Outputs

The repository includes generated figures in `plots/`, such as:

- Numeric/categorical distributions
- Correlation heatmap
- Demographic and temporal trends
- Confusion matrices
- ROC/PR curves
- Model comparison charts
- Feature importance
- Clinical risk scoring visuals
- Cross-validation comparison

## How to Run

### 1) Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2) Open and run the notebook

```bash
jupyter notebook Heart_Disease_Pipeline.ipynb
```

Run cells in order from top to bottom.

## Notes

- This project is notebook-first and designed for analysis + model benchmarking.
- No separate packaged training script or automated test suite is currently included.

## Author

- **Vansh Chopra (4M028)**
