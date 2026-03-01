# Classification Projects

Classification tasks focus on predicting discrete class labels. These projects span classical statistical methods (Logistic Regression), tree-based ensemble approaches (Random Forests), and modern deep learning techniques (Convolutional Neural Networks).

## Projects

### 1. Heart Disease Prediction
**File:** `heart-disease-prediction.ipynb`

A binary classification pipeline to predict cardiovascular disease risk from clinical indicators. The project compares Logistic Regression — where the decision boundary is defined by **log(p/1−p) = Xβ** — against tree-based methods including Decision Trees and Random Forests.

Feature engineering is driven by correlation analysis and domain knowledge. Model selection uses cross-validated AUC-ROC rather than raw accuracy, which is critical given typical class imbalance in medical datasets. The ensemble approach is contextualized through its bias-variance decomposition: averaging decorrelated trees reduces variance without increasing bias.

**Key Libraries:** `pandas`, `numpy`, `seaborn`, `scikit-learn`

---

### 2. Titanic Survival Prediction
**File:** `titanic-survival-prediction.ipynb`

A canonical classification benchmark treating the Titanic dataset with careful attention to preprocessing. The notebook goes beyond surface-level feature engineering to identify which features carry genuine predictive signal versus noise.

Missing data is handled through informed imputation strategies rather than naive mean-filling. Categorical encoding choices are justified, not defaulted. Models are evaluated on precision-recall tradeoffs with ensemble methods (bagging, boosting) applied and compared through the bias-variance lens. The goal is an interpretable model — knowing which features drive survival predictions matters as much as the final metric.

**Key Libraries:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`

---

### 3. Plant Disease Detection using CNN
**File:** `Plant-Disease-Detection-CNN.ipynb`

A deep learning project applying Convolutional Neural Networks to plant disease classification from leaf images. The architecture design reflects what CNNs actually compute: early convolutional layers learn low-frequency edge detectors (like Gabor filters), while deeper layers compose these into class-discriminative feature maps.

Data augmentation (random flips, rotations, zoom) is applied as a principled strategy to expand the effective training distribution and improve generalization. Training dynamics are monitored through learning curves on both training and validation loss to detect overfitting early and adjust regularization accordingly.

Developed and trained on Google Colab leveraging GPU acceleration.

**Key Libraries:** `tensorflow`, `keras`, `numpy`, `PIL`

---