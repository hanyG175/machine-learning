# Machine Learning Projects

A collection of machine learning projects spanning supervised learning, unsupervised learning, and deep learning. Each notebook is built with an emphasis on understanding the underlying mathematical formulations before reaching for library abstractions — the goal is always to know *why* a method works, not just *that* it works.

> Some projects are intentionally kept close to first principles. Others push further into applied territory. All of them are part of an ongoing effort to build rigorous, end-to-end competence across the ML landscape.

---

## Table of Contents

- [Projects](#projects)
  - [1. Linear Regression Basics](#1-linear-regression-basics)
  - [2. Heart Disease Prediction](#2-heart-disease-prediction)
  - [3. Titanic Survival Prediction](#3-titanic-survival-prediction)
  - [4. Life Expectancy Prediction](#4-life-expectancy-prediction)
  - [5. Netflix Content Clustering](#5-netflix-content-clustering)
  - [6. Plant Disease Detection using CNN](#6-plant-disease-detection-using-cnn)
- [Getting Started](#getting-started)
- [Contact](#contact)

---

## Projects

### 1. Linear Regression Basics
**Notebook:** `linear-regression-basics.ipynb`

A rigorous treatment of linear regression grounded in the Ordinary Least Squares (OLS) framework. Rather than just fitting a `sklearn` model, this notebook derives the closed-form solution **β = (XᵀX)⁻¹Xᵀy** from first principles — including the geometric interpretation of projection onto the column space of X and the conditions under which (XᵀX) is invertible.

Evaluation is handled through MSE, RMSE, and the coefficient of determination R² = 1 − SS_res/SS_tot, with particular attention to what R² actually measures and where it misleads. Residual plots are used to verify the Gauss-Markov assumptions (homoscedasticity, zero-mean errors, no autocorrelation).

---

### 2. Heart Disease Prediction
**Notebook:** `heart-disease-prediction.ipynb`

A binary classification pipeline built to predict cardiovascular disease risk from clinical indicators. The project compares Logistic Regression — where the decision boundary is defined by the log-odds **log(p/1−p) = Xβ** — against tree-based methods including Decision Trees and Random Forests.

Feature engineering is driven by correlation analysis and domain knowledge. Model selection uses cross-validated AUC-ROC rather than raw accuracy, which matters significantly given the class imbalance typical of medical datasets. The ensemble approach via Random Forest is contextualized through its bias-variance decomposition: averaging over decorrelated trees reduces variance without increasing bias.

**Key libraries:** `pandas`, `numpy`, `seaborn`, `scikit-learn`

---

### 3. Titanic Survival Prediction
**Notebook:** `titanic-survival-prediction.ipynb`

The Titanic dataset is a canonical classification benchmark, and this notebook treats it as such — going beyond surface-level preprocessing to think carefully about which features carry genuine predictive signal versus noise. Missing data is handled through informed imputation strategies rather than naive mean-filling, and categorical encoding choices are justified rather than defaulted.

Models are evaluated on their precision-recall tradeoff, with ensemble methods (bagging, boosting) applied and compared through the lens of the bias-variance tradeoff. The goal is not just a high-accuracy model, but an interpretable one — knowing *which* features drive survival probability and *why* matters as much as the final metric.

**Key libraries:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`

---

### 4. Life Expectancy Prediction
**Notebook:** `life-expectancy-prediction.ipynb`

A multivariate regression problem using WHO health and socioeconomic indicators to predict national life expectancy. The analytical challenge here is not model complexity but data quality — the dataset requires careful handling of multicollinearity, missing values across heterogeneous features, and the distinction between confounders and true predictors.

Exploratory analysis uses correlation matrices and partial plots to understand feature relationships before modelling. Regularization via Ridge (L2) and Lasso (L1) penalties is applied and compared, with the regularization parameter λ tuned through cross-validation. The Lasso solution's sparsity property is particularly relevant here for feature selection across a high-dimensional socioeconomic feature space.

**Key libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

### 5. Netflix Content Clustering
**Notebook:** `netflix-content-clustering.ipynb`

An unsupervised learning project that applies K-Means clustering to segment Netflix titles by content features including genre and release metadata. The notebook covers the full preprocessing pipeline — missing data imputation, categorical encoding, and feature scaling (critical for any distance-based algorithm) — before fitting the model.

Cluster count k is selected through the elbow method on inertia (within-cluster sum of squared distances) and validated with the silhouette coefficient **s(i) = (b(i) − a(i)) / max(a(i), b(i))**, which measures how well each point fits its assigned cluster relative to the nearest alternative. Cluster profiles are then interpreted to extract meaningful content groupings.

**Key libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

---

### 6. Plant Disease Detection using CNN
**Notebook:** `Plant-Disease-Detection-CNN.ipynb`

A deep learning project applying Convolutional Neural Networks to the problem of plant disease classification from leaf images. The architecture is built with an understanding of what CNNs actually compute: early convolutional layers learn low-frequency edge detectors (analogous to Gabor filters), while deeper layers compose these into class-discriminative feature maps.

Data augmentation (random flips, rotations, zoom) is applied not as a heuristic but as a principled strategy to improve generalization by expanding the effective training distribution. Training dynamics are monitored through learning curves on both training and validation loss to detect overfitting early and adjust regularization accordingly.

Developed and trained on Google Colab leveraging GPU acceleration.

**Key libraries:** `tensorflow`, `keras`, `numpy`, `PIL`

---

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`

### Installation

```bash
# Clone the repository
git clone https://github.com/hanyG175/machine-learning.git
cd machine-learning

# (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install libraries individually as needed per notebook.

### Running the Notebooks

```bash
jupyter notebook
```

Open the desired notebook from the browser interface and execute cells sequentially to reproduce experiments and results.

---

## Contact

Questions, collaborations, or technical discussion — feel free to reach out:

- **GitHub:** [hanyG175](https://github.com/hanyG175)
- **Email:** [hani.gaouaou.hg@gmail.com](mailto:hani.gaouaou.hg@gmail.com)
