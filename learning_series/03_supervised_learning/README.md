# Module 3: Supervised Learning

This module covers the core supervised learning algorithms for both classification and regression tasks. You'll learn to build, evaluate, and optimize models that predict outcomes based on labeled training data.

## Learning Objectives

By the end of this module, you will:
- Understand the difference between classification and regression
- Build and evaluate linear and logistic regression models
- Work with tree-based algorithms (Decision Trees, Random Forests)
- Apply Support Vector Machines for various tasks
- Use k-Nearest Neighbors for classification and regression
- Master hyperparameter tuning and model selection
- Apply cross-validation for robust model evaluation
- Handle class imbalance and real-world challenges

## Prerequisites

- Completion of Modules 0-2
- Understanding of data preprocessing and EDA
- Basic knowledge of linear algebra and statistics
- Familiarity with scikit-learn basics

## Module Contents

1. **Linear Regression** (`01_linear_regression.ipynb`)
   - Simple and multiple linear regression
   - Assumptions and diagnostics
   - Regularization (Ridge, Lasso, Elastic Net)
   - Polynomial regression and feature interactions

2. **Logistic Regression** (`02_logistic_regression.ipynb`)
   - Binary and multiclass classification
   - Sigmoid function and odds ratios
   - Regularization in classification
   - Interpreting coefficients and probabilities

3. **Decision Trees** (`03_decision_trees.ipynb`)
   - Tree construction and splitting criteria
   - Handling categorical and numerical features
   - Pruning and avoiding overfitting
   - Visualizing decision boundaries

4. **Ensemble Methods** (`04_ensemble_methods.ipynb`)
   - Random Forests and Extra Trees
   - Bagging and boosting concepts
   - Gradient Boosting and XGBoost
   - Using the existing `ensemble_model.py` from the repository

5. **Support Vector Machines** (`05_support_vector_machines.ipynb`)
   - Linear and non-linear SVM
   - Kernel trick and different kernels
   - Soft margin and regularization parameter C
   - SVM for regression (SVR)

6. **k-Nearest Neighbors** (`06_k_nearest_neighbors.ipynb`)
   - Distance metrics and choosing k
   - Lazy learning and curse of dimensionality
   - KNN for classification and regression
   - Weighted KNN and optimization techniques

7. **Model Selection and Hyperparameter Tuning** (`07_model_selection.ipynb`)
   - Grid search and random search
   - Cross-validation strategies
   - Learning curves and validation curves
   - Advanced optimization with Optuna

8. **Advanced Topics** (`08_advanced_supervised_learning.ipynb`)
   - Handling imbalanced datasets
   - Multi-output and multi-label classification
   - Calibrating prediction probabilities
   - Feature selection with supervised learning

9. **Real-world Project** (`09_supervised_learning_project.ipynb`)
   - End-to-end supervised learning pipeline
   - Model comparison and selection
   - Business impact evaluation
   - Deployment considerations

## Key Algorithms Overview

### Regression Algorithms
- **Linear Regression**: Best linear unbiased estimator
- **Ridge Regression**: L2 regularization for multicollinearity
- **Lasso Regression**: L1 regularization for feature selection
- **Elastic Net**: Combined L1 and L2 regularization

### Classification Algorithms
- **Logistic Regression**: Linear classifier with probabilistic output
- **Decision Trees**: Interpretable non-linear classifier
- **Random Forest**: Ensemble of decision trees
- **SVM**: Maximum margin classifier with kernel trick
- **KNN**: Instance-based learning algorithm

## Leveraging Existing Repository Tools

This module extensively uses the `ensemble_model.py` script:

### EnsembleFeatureImportance Class
```python
from ensemble_model import EnsembleFeatureImportance

# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Define multiple models for comparison
models = [
    ('RandomForest', RandomForestRegressor(random_state=42), 
     {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
    ('XGBoost', xgb.XGBRegressor(random_state=42),
     {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}),
    ('Lasso', Lasso(random_state=42),
     {'alpha': [0.001, 0.01, 0.1]})
]

# Initialize ensemble with configuration
ensemble = EnsembleFeatureImportance(
    data=data,
    target_column='target',
    models=models,
    top_n=10,
    scoring='neg_mean_squared_error'
)

# Run complete ensemble analysis
ensemble.run_ensemble()

# Get results
print("Feature Importances:", ensemble.feature_importances)
print("Best Model:", ensemble.best_final_model)
```

## Practical Projects

### Project 1: House Price Prediction
Build a regression model to predict house prices:
- Data exploration and feature engineering
- Compare linear models with tree-based models
- Handle categorical variables and outliers
- Evaluate using appropriate regression metrics

### Project 2: Customer Churn Classification
Predict customer churn for a subscription service:
- Handle class imbalance
- Feature importance analysis
- Model interpretation for business insights
- Cost-sensitive learning

### Project 3: Multi-class Text Classification
Classify news articles into categories:
- Text preprocessing and feature extraction
- Compare different algorithms
- Handle large feature spaces
- Evaluate with multi-class metrics

### Project 4: Time Series Forecasting
Predict sales using supervised learning approach:
- Feature engineering for time series
- Lag features and rolling statistics
- Model selection for temporal data
- Walk-forward validation

## Model Evaluation Framework

### Classification Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """Comprehensive classification evaluation"""
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
        print(f"\nAUC Score: {auc:.4f}")
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression_model(y_true, y_pred):
    """Comprehensive regression evaluation"""
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
```

## Advanced Techniques

### Hyperparameter Optimization
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna

def optimize_hyperparameters(model, param_grid, X, y, method='grid'):
    """Advanced hyperparameter optimization"""
    
    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    elif method == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=5, n_iter=50)
    elif method == 'optuna':
        # Optuna implementation for advanced optimization
        pass
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
```

### Feature Selection Pipeline
```python
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.pipeline import Pipeline

def create_feature_selection_pipeline(estimator, selection_method='rfe'):
    """Create pipeline with feature selection"""
    
    if selection_method == 'rfe':
        selector = RFE(estimator, n_features_to_select=10)
    elif selection_method == 'univariate':
        selector = SelectKBest(k=10)
    
    pipeline = Pipeline([
        ('selector', selector),
        ('estimator', estimator)
    ])
    
    return pipeline
```

## Real-World Considerations

### Handling Imbalanced Data
- **Resampling**: SMOTE, ADASYN, random over/under-sampling
- **Cost-sensitive learning**: Class weights, cost matrices
- **Evaluation metrics**: Precision, recall, F1-score, AUC
- **Threshold optimization**: ROC curves, precision-recall curves

### Model Interpretability
- **Feature importance**: Built-in importance scores
- **SHAP values**: Advanced model explanation
- **LIME**: Local interpretable explanations
- **Partial dependence plots**: Feature effect visualization

### Production Considerations
- **Model serialization**: Pickle, joblib, ONNX
- **Prediction pipelines**: Consistent preprocessing
- **Model monitoring**: Performance degradation detection
- **A/B testing**: Gradual model deployment

## Assessment Checklist

- [ ] Can build and evaluate linear regression models
- [ ] Understands logistic regression for classification
- [ ] Successfully implements decision trees and random forests
- [ ] Applies SVM for different types of problems
- [ ] Uses KNN effectively with proper distance metrics
- [ ] Performs hyperparameter tuning using multiple methods
- [ ] Handles imbalanced datasets appropriately
- [ ] Completed at least 3 practical supervised learning projects
- [ ] Can interpret and explain model results to stakeholders

## Estimated Time

**Total Duration:** 6-7 days (18-22 hours)

## Common Challenges and Solutions

### Challenge 1: Overfitting
**Solutions**: Cross-validation, regularization, feature selection, ensemble methods

### Challenge 2: Poor Performance
**Solutions**: Feature engineering, algorithm selection, hyperparameter tuning, more data

### Challenge 3: Imbalanced Classes
**Solutions**: Resampling techniques, cost-sensitive learning, appropriate metrics

### Challenge 4: High Dimensionality
**Solutions**: Feature selection, dimensionality reduction, regularization

## Next Steps

After mastering supervised learning, you'll be ready for Module 4: Unsupervised Learning, where you'll learn to find patterns in data without labeled examples, including clustering and dimensionality reduction techniques!