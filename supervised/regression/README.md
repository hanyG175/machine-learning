# Regression Projects

Regression tasks focus on predicting continuous numerical outputs from input features. These projects demonstrate fundamental and advanced regression techniques, from the classical Ordinary Least Squares (OLS) framework to modern regularization approaches.

## Projects

### 1. Linear Regression Basics
**File:** `linear-regression-basics.ipynb`

A rigorous treatment of linear regression grounded in the Ordinary Least Squares (OLS) framework. Rather than just fitting a model, this notebook derives the closed-form solution **β = (XᵀX)⁻¹Xᵀy** from first principles — including the geometric interpretation of projection onto the column space of X and the conditions under which (XᵀX) is invertible.

Evaluation uses MSE, RMSE, and the coefficient of determination R² = 1 − SS_res/SS_tot, with particular attention to what R² actually measures. Residual plots verify the Gauss-Markov assumptions (homoscedasticity, zero-mean errors, no autocorrelation).

**Key Libraries:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`

---

### 2. Life Expectancy Prediction
**File:** `life-expectancy-prediction.ipynb`

A multivariate regression problem using WHO health and socioeconomic indicators to predict national life expectancy. The focus is on handling data quality challenges: managing multicollinearity, imputing missing values, and distinguishing confounders from true predictors.

Exploratory analysis uses correlation matrices and partial plots. Regularization via Ridge (L2) and Lasso (L1) is applied and compared, with λ tuned through cross-validation. The Lasso solution's sparsity is particularly valuable for feature selection in high-dimensional socioeconomic feature spaces.

**Key Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Learning Path

1. Start with **Linear Regression Basics** to understand the mathematical foundations
2. Move to **Life Expectancy Prediction** to see regularization in action on real-world messy data

---

**Difficulty Level:** Foundational → Intermediate
