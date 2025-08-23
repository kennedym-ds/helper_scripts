# Module 1: Machine Learning Fundamentals

This module introduces core machine learning concepts, terminology, and the overall workflow. You'll learn the theoretical foundations that underpin all machine learning applications.

## Learning Objectives

By the end of this module, you will:
- Understand what machine learning is and how it differs from traditional programming
- Know the main types of machine learning (supervised, unsupervised, reinforcement)
- Understand the ML workflow from data to deployed model
- Recognize overfitting, underfitting, and the bias-variance tradeoff
- Know how to evaluate model performance using appropriate metrics
- Understand cross-validation and its importance

## Prerequisites

- Completion of Module 0 (Setup)
- Basic understanding of statistics and linear algebra
- Familiarity with Python programming

## Module Contents

1. **What is Machine Learning?** (`01_what_is_ml.ipynb`)
   - Traditional programming vs. machine learning
   - Examples of ML in everyday life
   - When to use machine learning
   - Limitations and ethical considerations

2. **Types of Machine Learning** (`02_types_of_ml.ipynb`)
   - Supervised learning (classification and regression)
   - Unsupervised learning (clustering, dimensionality reduction)
   - Reinforcement learning basics
   - Semi-supervised and self-supervised learning

3. **The Machine Learning Workflow** (`03_ml_workflow.ipynb`)
   - Problem definition and data collection
   - Data exploration and preprocessing
   - Model selection and training
   - Evaluation and validation
   - Deployment and monitoring

4. **Training, Validation, and Testing** (`04_train_val_test.ipynb`)
   - The importance of splitting data
   - Training set, validation set, test set
   - Cross-validation techniques
   - Holdout validation vs. k-fold cross-validation

5. **Overfitting and Underfitting** (`05_overfitting_underfitting.ipynb`)
   - Bias-variance tradeoff
   - Model complexity and generalization
   - Detecting overfitting
   - Regularization techniques

6. **Performance Metrics** (`06_performance_metrics.ipynb`)
   - Classification metrics (accuracy, precision, recall, F1-score)
   - Regression metrics (MSE, MAE, R²)
   - ROC curves and AUC
   - Confusion matrices

7. **Hands-on Exercise** (`07_first_ml_project.ipynb`)
   - Complete end-to-end ML project
   - Iris dataset classification
   - Applying all concepts learned

## Key Concepts Summary

### Machine Learning Types
- **Supervised Learning**: Learning from labeled examples
  - Classification: Predicting categories (spam/not spam)
  - Regression: Predicting continuous values (house prices)
- **Unsupervised Learning**: Finding patterns in unlabeled data
  - Clustering: Grouping similar items
  - Dimensionality Reduction: Simplifying data while preserving information
- **Reinforcement Learning**: Learning through interaction and feedback

### The Bias-Variance Tradeoff
- **High Bias (Underfitting)**: Model is too simple, misses patterns
- **High Variance (Overfitting)**: Model is too complex, memorizes noise
- **Sweet Spot**: Balanced model that generalizes well

### Model Evaluation
- Always use separate test data
- Cross-validation for robust evaluation
- Choose metrics appropriate for your problem
- Consider business impact, not just statistical metrics

## Exercises

1. **Concept Quiz**: Test your understanding of key terms
2. **Identify ML Types**: Categorize real-world problems
3. **Overfitting Detection**: Analyze learning curves
4. **Metrics Selection**: Choose appropriate evaluation metrics
5. **Complete Project**: Build your first ML classifier

## Estimated Time

**Total Duration:** 3-4 days (8-12 hours)

## Assessment Checklist

- [ ] Can explain what machine learning is in simple terms
- [ ] Can identify supervised vs. unsupervised learning problems
- [ ] Understands the importance of train/validation/test splits
- [ ] Can recognize overfitting and underfitting in plots
- [ ] Knows when to use different performance metrics
- [ ] Completed the hands-on project successfully

## Next Steps

After mastering these fundamentals, you'll be ready for Module 2: Data Preprocessing and EDA, where you'll learn to prepare real-world data for machine learning!