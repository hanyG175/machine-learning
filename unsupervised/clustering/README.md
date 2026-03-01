# Clustering Projects

Clustering projects focus on discovering natural groupings in data through unsupervised learning. This section demonstrates distance-based clustering algorithms and validation techniques for determining cluster quality when ground truth labels are unavailable.

## Projects

### 1. Netflix Content Clustering
**File:** `netflix-content-clustering.ipynb`

An unsupervised learning project applying K-Means clustering to segment Netflix titles by content features including genre and release metadata. The notebook covers the complete preprocessing pipeline — missing data imputation, categorical encoding, and feature scaling (critical for any distance-based algorithm) — before fitting the model.

Cluster count selection uses the elbow method on inertia (within-cluster sum of squared distances) and is validated with the silhouette coefficient:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

This metric measures how well each point fits its assigned cluster relative to the nearest alternative. Cluster profiles are then interpreted to extract meaningful content groupings from the high-dimensional feature space.

**Key Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

---

## Learning Path

Start with **Netflix Content Clustering** to understand K-Means, the elbow method, and silhouette validation in the context of content-based recommendation systems.

---

**Difficulty Level:** Foundational → Intermediate

---

## Key Metrics

- **Inertia:** Sum of squared distances from each point to its assigned cluster center (lower is better, but used with elbow method)
- **Silhouette Score:** Ranges from -1 to 1, measuring cluster cohesion and separation (higher is better)
- **Davies-Bouldin Index:** Ratio of within-cluster to between-cluster distances (lower is better)
