# Module 4: Unsupervised Learning

This module explores unsupervised learning techniques that find hidden patterns in data without labeled examples. You'll learn clustering, dimensionality reduction, and anomaly detection methods essential for data exploration and feature engineering.

## Learning Objectives

By the end of this module, you will:
- Understand the principles and applications of unsupervised learning
- Master clustering algorithms (K-means, hierarchical, DBSCAN)
- Apply dimensionality reduction techniques (PCA, t-SNE, UMAP)
- Detect anomalies and outliers in datasets
- Perform association rule mining and market basket analysis
- Use unsupervised learning for data preprocessing and feature engineering
- Evaluate unsupervised learning results using appropriate metrics

## Prerequisites

- Completion of Modules 0-3
- Understanding of linear algebra and statistics
- Knowledge of distance metrics and similarity measures
- Experience with supervised learning for comparison

## Module Contents

1. **Introduction to Unsupervised Learning** (`01_intro_unsupervised_learning.ipynb`)
   - What is unsupervised learning?
   - Types of unsupervised learning tasks
   - Applications and use cases
   - Evaluation challenges and strategies

2. **K-Means Clustering** (`02_kmeans_clustering.ipynb`)
   - Algorithm mechanics and Lloyd's algorithm
   - Choosing optimal number of clusters (elbow method, silhouette)
   - Initialization strategies and convergence
   - Limitations and when to use K-means

3. **Hierarchical Clustering** (`03_hierarchical_clustering.ipynb`)
   - Agglomerative and divisive clustering
   - Linkage criteria (single, complete, average, Ward)
   - Dendrograms and cluster interpretation
   - Comparing with K-means

4. **Density-Based Clustering** (`04_density_based_clustering.ipynb`)
   - DBSCAN algorithm and parameters
   - Handling noise and varying densities
   - OPTICS and other density-based methods
   - Applications to spatial and irregular data

5. **Principal Component Analysis (PCA)** (`05_principal_component_analysis.ipynb`)
   - Mathematical foundations and eigendecomposition
   - Variance explained and component interpretation
   - PCA for visualization and preprocessing
   - Kernel PCA for non-linear relationships

6. **Advanced Dimensionality Reduction** (`06_advanced_dimensionality_reduction.ipynb`)
   - t-SNE for non-linear visualization
   - UMAP for scalable dimension reduction
   - Factor Analysis and Independent Component Analysis
   - Choosing the right technique

7. **Anomaly Detection** (`07_anomaly_detection.ipynb`)
   - Statistical methods and z-scores
   - Isolation Forest and One-Class SVM
   - Local Outlier Factor (LOF)
   - Using existing `isolation_visual.py` for educational demonstrations

8. **Association Rule Mining** (`08_association_rule_mining.ipynb`)
   - Market basket analysis concepts
   - Support, confidence, and lift metrics
   - Apriori and FP-Growth algorithms
   - Applications beyond retail

9. **Unsupervised Learning Projects** (`09_unsupervised_projects.ipynb`)
   - Customer segmentation analysis
   - Image compression with PCA
   - Anomaly detection in time series
   - Document clustering and topic modeling

## Key Algorithms Overview

### Clustering Algorithms
- **K-Means**: Partition data into k spherical clusters
- **Hierarchical**: Build cluster hierarchy (tree structure)
- **DBSCAN**: Density-based clustering with noise handling
- **Gaussian Mixture Models**: Probabilistic clustering

### Dimensionality Reduction
- **PCA**: Linear transformation maximizing variance
- **t-SNE**: Non-linear method for visualization
- **UMAP**: Uniform manifold approximation
- **Factor Analysis**: Identify underlying factors

### Anomaly Detection
- **Statistical Methods**: Z-score, modified Z-score
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Support vector approach
- **LOF**: Local density-based detection

## Leveraging Existing Repository Tools

This module uses the `isolation_visual.py` for educational demonstrations:

### Isolation Forest Visualization
```python
# The existing isolation_visual.py provides educational animations
# showing how Isolation Forest works step-by-step

# Example of applying similar concepts in practice:
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def demonstrate_isolation_forest(X, contamination=0.1):
    """
    Demonstrate Isolation Forest anomaly detection
    """
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    y_pred = iso_forest.fit_predict(X)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    
    # Plot normal points
    normal_points = X[y_pred == 1]
    plt.scatter(normal_points[:, 0], normal_points[:, 1], 
               c='blue', label='Normal', alpha=0.7)
    
    # Plot anomalies
    anomalies = X[y_pred == -1]
    plt.scatter(anomalies[:, 0], anomalies[:, 1], 
               c='red', label='Anomaly', alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Isolation Forest Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return y_pred
```

## Practical Projects

### Project 1: Customer Segmentation
Segment customers based on purchasing behavior:
- RFM analysis (Recency, Frequency, Monetary)
- K-means clustering for customer groups
- Cluster interpretation and business insights
- Targeted marketing strategy development

### Project 2: Image Compression with PCA
Use PCA to compress images while preserving quality:
- Load and preprocess image data
- Apply PCA with different numbers of components
- Visualize reconstruction quality vs. compression
- Compare with other compression methods

### Project 3: Document Clustering
Cluster text documents by topic:
- Text preprocessing and TF-IDF vectorization
- Apply different clustering algorithms
- Evaluate clustering quality
- Topic interpretation and labeling

### Project 4: Anomaly Detection in Network Traffic
Detect unusual patterns in network data:
- Feature engineering from network logs
- Compare different anomaly detection methods
- Handle streaming data scenarios
- Alert system design

## Clustering Evaluation Metrics

### Internal Metrics (No ground truth needed)
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering_internal(X, labels):
    """Evaluate clustering using internal metrics"""
    
    if len(set(labels)) > 1:  # Need at least 2 clusters
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        return silhouette, calinski_harabasz, davies_bouldin
    else:
        print("Need at least 2 clusters for evaluation")
        return None
```

### External Metrics (When ground truth available)
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_clustering_external(y_true, y_pred):
    """Evaluate clustering using external metrics"""
    
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    
    return ari, nmi
```

## Advanced Clustering Techniques

### Gaussian Mixture Models
```python
from sklearn.mixture import GaussianMixture

def fit_gaussian_mixture(X, n_components_range=range(1, 11)):
    """Fit Gaussian Mixture Models with different components"""
    
    bic_scores = []
    aic_scores = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
    
    # Plot information criteria
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bic_scores, 'o-', label='BIC')
    plt.plot(n_components_range, aic_scores, 'o-', label='AIC')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion')
    plt.title('Model Selection for Gaussian Mixture')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Return best model based on BIC
    best_n_components = n_components_range[np.argmin(bic_scores)]
    best_gmm = GaussianMixture(n_components=best_n_components, random_state=42)
    best_gmm.fit(X)
    
    return best_gmm
```

### Spectral Clustering
```python
from sklearn.cluster import SpectralClustering

def apply_spectral_clustering(X, n_clusters, affinity='rbf'):
    """Apply spectral clustering for non-convex clusters"""
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        random_state=42
    )
    
    labels = spectral.fit_predict(X)
    return labels
```

## Dimensionality Reduction Comparison

### Choosing the Right Method
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def compare_dimensionality_reduction(X, labels=None):
    """Compare different dimensionality reduction techniques"""
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [('PCA', X_pca), ('t-SNE', X_tsne), ('UMAP', X_umap)]
    
    for i, (method, X_reduced) in enumerate(methods):
        if labels is not None:
            scatter = axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=axes[i])
        else:
            axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
        
        axes[i].set_title(f'{method}')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return X_pca, X_tsne, X_umap
```

## Anomaly Detection Framework

### Multi-Method Comparison
```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

def compare_anomaly_detection(X, contamination=0.1):
    """Compare different anomaly detection methods"""
    
    methods = {
        'Isolation Forest': IsolationForest(contamination=contamination, random_state=42),
        'One-Class SVM': OneClassSVM(nu=contamination),
        'Local Outlier Factor': LocalOutlierFactor(contamination=contamination)
    }
    
    results = {}
    
    for name, method in methods.items():
        if name == 'Local Outlier Factor':
            # LOF only predicts on training data
            y_pred = method.fit_predict(X)
        else:
            y_pred = method.fit(X).predict(X)
        
        # Convert to binary (1: normal, 0: anomaly)
        y_binary = (y_pred == 1).astype(int)
        
        results[name] = {
            'predictions': y_pred,
            'binary': y_binary,
            'n_anomalies': np.sum(y_pred == -1)
        }
        
        print(f"{name}: {results[name]['n_anomalies']} anomalies detected")
    
    return results
```

## Real-World Applications

### Business Intelligence
- **Customer Segmentation**: RFM analysis, behavioral clustering
- **Market Research**: Survey data clustering, preference mapping
- **Fraud Detection**: Credit card transactions, insurance claims
- **Quality Control**: Manufacturing defect detection

### Scientific Research
- **Gene Expression**: Clustering genes by expression patterns
- **Astronomy**: Galaxy classification, exoplanet detection
- **Climate Science**: Weather pattern recognition
- **Social Networks**: Community detection, influence analysis

### Technology Applications
- **Recommendation Systems**: Item clustering, collaborative filtering
- **Computer Vision**: Image segmentation, object grouping
- **Natural Language Processing**: Topic modeling, document clustering
- **Cybersecurity**: Intrusion detection, malware classification

## Assessment Checklist

- [ ] Can apply K-means clustering and choose optimal k
- [ ] Understands hierarchical clustering and dendrograms
- [ ] Successfully uses DBSCAN for density-based clustering
- [ ] Applies PCA for dimensionality reduction and visualization
- [ ] Uses t-SNE and UMAP for non-linear visualization
- [ ] Implements various anomaly detection methods
- [ ] Evaluates unsupervised learning results appropriately
- [ ] Completed at least 3 practical unsupervised learning projects
- [ ] Can interpret and explain unsupervised learning results

## Estimated Time

**Total Duration:** 4-5 days (12-16 hours)

## Common Challenges and Solutions

### Challenge 1: Choosing Number of Clusters
**Solutions**: Elbow method, silhouette analysis, gap statistic, domain knowledge

### Challenge 2: High-Dimensional Data
**Solutions**: Dimensionality reduction, feature selection, distance metric selection

### Challenge 3: Interpreting Results
**Solutions**: Visualization, cluster profiling, domain expert consultation

### Challenge 4: Scalability
**Solutions**: Mini-batch algorithms, sampling, distributed computing

## Next Steps

After mastering unsupervised learning, you'll move to Module 5: Computer Vision, where you'll apply many of these concepts to image data, including clustering similar images and using PCA for image compression and feature extraction!