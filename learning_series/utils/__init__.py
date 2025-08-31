"""
Utility functions for the Machine Learning Learning Series

This module provides helper functions that are used across multiple
modules in the learning series. It includes data generation, 
visualization utilities, and common ML operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def generate_classification_dataset(n_samples=1000, n_features=20, n_informative=10, 
                                  n_redundant=5, n_classes=2, random_state=42):
    """
    Generate a synthetic classification dataset for practice.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=20
        Total number of features
    n_informative : int, default=10
        Number of informative features
    n_redundant : int, default=5
        Number of redundant features
    n_classes : int, default=2
        Number of classes
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    feature_names : list
        List of feature names
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return X, y, feature_names

def generate_regression_dataset(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """
    Generate a synthetic regression dataset for practice.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=10
        Number of features
    noise : float, default=0.1
        Standard deviation of gaussian noise added to output
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    feature_names : list
        List of feature names
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return X, y, feature_names

def generate_clustering_dataset(n_samples=300, centers=3, n_features=2, 
                              cluster_std=1.0, random_state=42):
    """
    Generate a synthetic clustering dataset for practice.
    
    Parameters:
    -----------
    n_samples : int, default=300
        Number of samples to generate
    centers : int, default=3
        Number of cluster centers
    n_features : int, default=2
        Number of features
    cluster_std : float, default=1.0
        Standard deviation of clusters
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Cluster labels
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    return X, y

def plot_learning_curves(estimator, X, y, cv=5, scoring='accuracy', 
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curves to visualize model performance vs training set size.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        The ML model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='accuracy'
        Scoring metric
    train_sizes : array-like, default=np.linspace(0.1, 1.0, 10)
        Training set sizes to evaluate
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes,
        n_jobs=-1, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel(f'{scoring.title()} Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names=None, normalize=False):
    """
    Plot a heatmap of the confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of classes for labeling
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_feature_importance(feature_names, importance_scores, top_n=10):
    """
    Plot feature importance scores.
    
    Parameters:
    -----------
    feature_names : list
        Names of features
    importance_scores : array-like
        Importance scores for each feature
    top_n : int, default=10
        Number of top features to display
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

def create_model_comparison_plot(model_results):
    """
    Create a comparison plot of multiple models' performance.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and scores as values
        Example: {'Random Forest': 0.85, 'SVM': 0.82, 'Logistic Regression': 0.78}
    """
    models = list(model_results.keys())
    scores = list(model_results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_model_and_results(model, results, model_name, save_path='./'):
    """
    Save trained model and results to files.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to save
    results : dict
        Dictionary containing model results and metrics
    model_name : str
        Name for the saved files
    save_path : str, default='./'
        Directory to save files
    """
    import pickle
    import json
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_file = os.path.join(save_path, f'{model_name}_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results_file = os.path.join(save_path, f'{model_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Model saved to: {model_file}")
    print(f"✅ Results saved to: {results_file}")

def load_sample_dataset(dataset_name):
    """
    Load a sample dataset for practice.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('iris', 'boston', 'wine', etc.)
        
    Returns:
    --------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix
    y : numpy.ndarray or pandas.Series
        Target vector
    feature_names : list
        List of feature names
    """
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
    
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'digits': load_digits
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not available. "
                        f"Choose from: {list(datasets.keys())}")
    
    data = datasets[dataset_name]()
    return data.data, data.target, data.feature_names

def print_dataset_info(X, y, dataset_name="Dataset"):
    """
    Print comprehensive information about a dataset.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    dataset_name : str, default="Dataset"
        Name of the dataset for display
    """
    print(f"📊 {dataset_name} Information:")
    print(f"   • Shape: {X.shape}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Samples: {X.shape[0]}")
    print(f"   • Target classes: {len(np.unique(y))}")
    print(f"   • Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    if hasattr(X, 'dtypes'):  # If it's a pandas DataFrame
        print(f"   • Data types: {X.dtypes.value_counts().to_dict()}")
        print(f"   • Missing values: {X.isnull().sum().sum()}")

class ProgressTracker:
    """
    Simple progress tracker for learning exercises.
    """
    
    def __init__(self, module_name):
        self.module_name = module_name
        self.exercises = []
        self.completed = []
    
    def add_exercise(self, exercise_name, description=""):
        """Add an exercise to track."""
        self.exercises.append({
            'name': exercise_name,
            'description': description,
            'completed': False
        })
    
    def complete_exercise(self, exercise_name):
        """Mark an exercise as completed."""
        for exercise in self.exercises:
            if exercise['name'] == exercise_name:
                exercise['completed'] = True
                self.completed.append(exercise_name)
                print(f"✅ Completed: {exercise_name}")
                return
        print(f"❌ Exercise '{exercise_name}' not found")
    
    def show_progress(self):
        """Display current progress."""
        total = len(self.exercises)
        completed = sum(1 for ex in self.exercises if ex['completed'])
        
        print(f"\n📚 {self.module_name} Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        print("=" * 50)
        
        for i, exercise in enumerate(self.exercises, 1):
            status = "✅" if exercise['completed'] else "⏳"
            print(f"{i:2d}. {status} {exercise['name']}")
            if exercise['description'] and not exercise['completed']:
                print(f"     {exercise['description']}")
        
        if completed == total:
            print(f"\n🎉 Congratulations! You've completed all exercises in {self.module_name}!")
        else:
            remaining = total - completed
            print(f"\n💪 Keep going! {remaining} exercise{'s' if remaining > 1 else ''} remaining.")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing utility functions...")
    
    # Test dataset generation
    X, y, features = generate_classification_dataset(n_samples=100, n_features=5)
    print(f"Generated classification dataset: {X.shape}")
    
    # Test dataset info
    print_dataset_info(X, y, "Test Classification Dataset")
    
    # Test progress tracker
    tracker = ProgressTracker("Test Module")
    tracker.add_exercise("Learn Python", "Basic Python programming")
    tracker.add_exercise("Data Analysis", "Pandas and NumPy")
    tracker.complete_exercise("Learn Python")
    tracker.show_progress()
    
    print("\n✅ All utility functions working correctly!")