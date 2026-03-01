# Machine Learning Projects

A collection of machine learning projects spanning supervised learning, unsupervised learning, and deep learning. Each notebook is built with an emphasis on understanding the underlying mathematical formulations before reaching for library abstractions — the goal is always to know *why* a method works, not just *that* it works.

> Some projects are intentionally kept close to first principles. Others push further into applied territory. All of them are part of an ongoing effort to build rigorous, end-to-end competence across the ML landscape.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Supervised Learning](#supervised-learning)
  - [Regression](#regression)
  - [Classification](#classification)
- [Unsupervised Learning](#unsupervised-learning)
  - [Clustering](#clustering)
- [Getting Started](#getting-started)
- [Contact](#contact)

---

## Repository Structure

```
machine-learning/
├── README.md (this file)
├── supervised/
│   ├── README.md
│   ├── regression/
│   │   ├── README.md
│   │   ├── linear-regression-basics.ipynb
│   │   └── life-expectancy-prediction.ipynb
│   └── classification/
│       ├── README.md
│       ├── heart-disease-prediction.ipynb
│       ├── titanic-survival-prediction.ipynb
│       └── Plant-Disease-Detection-CNN.ipynb
└── unsupervised/
    ├── README.md
    └── clustering/
        ├── README.md
        └── netflix-content-clustering.ipynb
```

---

## Supervised Learning

Supervised learning projects use labeled data to train predictive models. The models learn to map inputs to known outputs, enabling prediction on new, unseen data.

See [supervised/README.md](supervised/README.md) for overview and structure.

### Regression

Projects predicting continuous numerical values. Techniques include Linear Regression, Ridge and Lasso regularization.

See [supervised/regression/README.md](supervised/regression/README.md) for details on:
- **Linear Regression Basics** — Closed-form OLS solutions and mathematical foundations
- **Life Expectancy Prediction** — Multivariate regression with regularization on real-world data

### Classification

Projects predicting discrete class labels. Techniques range from Logistic Regression and ensemble methods to deep learning.

See [supervised/classification/README.md](supervised/classification/README.md) for details on:
- **Heart Disease Prediction** — Binary classification with classical statistical methods
- **Titanic Survival Prediction** — Classification with careful feature engineering and ensemble methods
- **Plant Disease Detection using CNN** — Image classification with Convolutional Neural Networks

---

## Unsupervised Learning

Unsupervised learning projects discover hidden patterns in unlabeled data without explicit target variables.

See [unsupervised/README.md](unsupervised/README.md) for overview and structure.

### Clustering

Projects discovering natural groupings in data through distance-based algorithms.

See [unsupervised/clustering/README.md](unsupervised/clustering/README.md) for details on:
- **Netflix Content Clustering** — K-Means clustering with silhouette validation

---

## Original Project Summaries

For detailed mathematical foundations and methodology of each project, refer to the README files in each subfolder.

### 1. Linear Regression Basics
**Location:** `supervised/regression/linear-regression-basics.ipynb`

A rigorous treatment of linear regression grounded in the Ordinary Least Squares (OLS) framework. Derives the closed-form solution **β = (XᵀX)⁻¹Xᵀy** from first principles, including geometric interpretation and invertibility conditions. Emphasizes understanding through residual analysis and verification of Gauss-Markov assumptions.

---

### 2. Heart Disease Prediction
**Location:** `supervised/classification/heart-disease-prediction.ipynb`

A binary classification pipeline predicting cardiovascular disease from clinical indicators. Compares Logistic Regression against Decision Trees and Random Forests. Uses AUC-ROC for model selection, addressing class imbalance typical in medical datasets.

---

### 3. Titanic Survival Prediction
**Location:** `supervised/classification/titanic-survival-prediction.ipynb`

Classification using the Titanic dataset with emphasis on thoughtful feature engineering and data quality. Handles missing data through informed imputation and uses ensemble methods to balance bias-variance tradeoffs. Prioritizes model interpretability.

---

### 4. Life Expectancy Prediction
**Location:** `supervised/regression/life-expectancy-prediction.ipynb`

A multivariate regression predicting national life expectancy from WHO indicators. Demonstrates handling of multicollinearity and missing values. Applies Ridge (L2) and Lasso (L1) regularization with cross-validated λ tuning, leveraging Lasso's sparsity for feature selection.

---

### 5. Netflix Content Clustering
**Location:** `unsupervised/clustering/netflix-content-clustering.ipynb`

Unsupervised learning applying K-Means to segment Netflix content. Covers complete preprocessing (imputation, encoding, scaling) and cluster validation using inertia elbow method and silhouette coefficient **s(i) = (b(i) − a(i)) / max(a(i), b(i))**.

---

### 6. Plant Disease Detection using CNN
**Location:** `supervised/classification/Plant-Disease-Detection-CNN.ipynb`

Deep learning project applying Convolutional Neural Networks to plant disease classification from leaf images. Demonstrates CNN architecture design with data augmentation and learning curve monitoring for overfitting detection. Trained on Google Colab with GPU acceleration.

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
