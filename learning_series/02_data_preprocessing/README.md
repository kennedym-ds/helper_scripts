# Module 2: Data Preprocessing and Exploratory Data Analysis

This module teaches you how to clean, explore, and prepare real-world data for machine learning. You'll learn to handle missing values, outliers, and feature engineering while leveraging automated tools for comprehensive data analysis.

## Learning Objectives

By the end of this module, you will:
- Master data cleaning and preprocessing techniques
- Perform comprehensive exploratory data analysis (EDA)
- Handle missing values and outliers effectively
- Engineer meaningful features from raw data
- Use automated EDA tools for faster insights
- Prepare data pipelines for machine learning
- Understand data quality assessment and validation

## Prerequisites

- Completion of Modules 0-1
- Understanding of Pandas and NumPy
- Basic statistics knowledge
- Familiarity with matplotlib and seaborn

## Module Contents

1. **Data Quality Assessment** (`01_data_quality_assessment.ipynb`)
   - Identifying data quality issues
   - Data profiling and validation
   - Duplicate detection and handling
   - Data type optimization

2. **Handling Missing Values** (`02_missing_values.ipynb`)
   - Types of missing data (MCAR, MAR, MNAR)
   - Missing value visualization
   - Imputation strategies (mean, median, mode, advanced methods)
   - Missing value indicators

3. **Outlier Detection and Treatment** (`03_outlier_detection.ipynb`)
   - Statistical methods (Z-score, IQR)
   - Visualization techniques for outlier identification
   - Robust statistical measures
   - Outlier treatment strategies

4. **Feature Engineering Fundamentals** (`04_feature_engineering.ipynb`)
   - Creating new features from existing ones
   - Polynomial features and interactions
   - Datetime feature extraction
   - Categorical encoding techniques

5. **Automated EDA with Helper Tools** (`05_automated_eda.ipynb`)
   - Using the existing `auto_eda.py` module
   - Generating comprehensive data reports
   - Statistical summaries and visualizations
   - Feature importance analysis

6. **Data Transformation and Scaling** (`06_data_transformation.ipynb`)
   - Normalization vs. standardization
   - Feature scaling techniques
   - Handling skewed distributions
   - Power transformations

7. **Advanced Preprocessing Techniques** (`07_advanced_preprocessing.ipynb`)
   - Principal Component Analysis (PCA) for dimensionality reduction
   - Feature selection methods
   - Handling imbalanced datasets
   - Text data preprocessing basics

8. **Real-world Data Pipeline** (`08_data_pipeline_project.ipynb`)
   - Building end-to-end preprocessing pipelines
   - Pipeline automation and reusability
   - Data validation and monitoring
   - Integration with scikit-learn pipelines

## Leveraging Existing Tools

This module makes extensive use of the `auto_eda.py` script already in the repository:

### GeneralEDA Class Features
```python
from auto_eda import GeneralEDA

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Initialize EDA
eda = GeneralEDA(df)

# Perform comprehensive analysis
eda.validate_data_integrity()
eda.data_info()
eda.statistical_summary()
eda.handle_missing_values(strategy='mean')
eda.handle_duplicates()
eda.handle_outliers(method='zscore', threshold=3)
eda.feature_engineering()
eda.feature_scaling(method='standard')

# Generate automated reports
eda.generate_pandas_profiling_report(
    output_file="data_profile_report.html",
    title="Dataset Analysis Report"
)

# Detect anomalies
eda.detect_and_visualize_anomalies(contamination=0.05)

# Save comprehensive report
eda.save_report(filepath="eda_summary.txt")
```

### Key Features of the Auto-EDA Tool

1. **Data Integrity Validation**
   - Automatic data type detection
   - Missing value analysis
   - Duplicate identification
   - Memory usage optimization

2. **Statistical Analysis**
   - Descriptive statistics for all variables
   - Distribution analysis
   - Correlation matrices
   - Variance inflation factor (VIF) calculation

3. **Automated Feature Engineering**
   - Interaction term creation
   - Datetime feature extraction
   - Categorical encoding
   - Polynomial feature generation

4. **Visualization Generation**
   - Automated plotting for different data types
   - Distribution plots and histograms
   - Correlation heatmaps
   - Box plots for outlier detection

## Hands-on Projects

### Project 1: Real Estate Data Analysis
Using a real estate dataset, perform complete EDA:
```python
# Load real estate data
real_estate_df = pd.read_csv("datasets/real_estate.csv")

# Initialize EDA
eda = GeneralEDA(real_estate_df)

# Complete analysis workflow
eda.validate_data_integrity()
eda.handle_missing_values(strategy='median')  # For numerical features
eda.handle_outliers(method='iqr')
eda.feature_engineering()
eda.feature_scaling(method='robust')

# Generate insights
transformed_data = eda.get_dataframe()
print("Original shape:", real_estate_df.shape)
print("Transformed shape:", transformed_data.shape)
```

### Project 2: Customer Behavior Analysis
Analyze customer transaction data:
- Handle missing customer information
- Engineer features from transaction timestamps
- Detect anomalous purchase patterns
- Create customer segmentation features

### Project 3: Time Series Data Preparation
Prepare time series data for forecasting:
- Handle irregular time intervals
- Create lag features and rolling statistics
- Detect and handle seasonal patterns
- Engineer datetime-based features

## Advanced Preprocessing Techniques

### Handling Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

# Different encoding strategies for different scenarios
def encode_categorical_features(df, target_column=None):
    """Apply appropriate encoding based on cardinality and context"""
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        unique_values = df[col].nunique()
        
        if unique_values == 2:
            # Binary encoding for binary categories
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col])
        elif unique_values <= 10:
            # One-hot encoding for low cardinality
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
        else:
            # Target encoding for high cardinality (if target available)
            if target_column:
                encoder = TargetEncoder()
                df[col + '_target_encoded'] = encoder.fit_transform(df[col], df[target_column])
    
    return df
```

### Feature Selection Pipeline
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def feature_selection_pipeline(X, y, method='rf_importance'):
    """Comprehensive feature selection"""
    
    if method == 'univariate':
        selector = SelectKBest(score_func=f_classif, k=10)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
    elif method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=10)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
    elif method == 'rf_importance':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(10)['feature'].tolist()
        X_selected = X[top_features]
        selected_features = top_features
    
    return X_selected, selected_features
```

### Data Validation Framework
```python
def validate_preprocessed_data(df_original, df_processed):
    """Validate that preprocessing didn't introduce errors"""
    
    validation_results = {}
    
    # Check for data leakage
    if df_processed.shape[0] != df_original.shape[0]:
        validation_results['row_count_mismatch'] = True
    
    # Check for extreme outliers introduced
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        if col in df_original.columns:
            original_range = df_original[col].quantile(0.99) - df_original[col].quantile(0.01)
            processed_range = df_processed[col].quantile(0.99) - df_processed[col].quantile(0.01)
            
            if processed_range > 10 * original_range:
                validation_results[f'{col}_extreme_scaling'] = True
    
    # Check for missing values introduced
    original_missing = df_original.isnull().sum().sum()
    processed_missing = df_processed.isnull().sum().sum()
    
    if processed_missing > original_missing:
        validation_results['new_missing_values'] = True
    
    return validation_results
```

## Integration with Scikit-learn Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """Create reusable preprocessing pipeline"""
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor
```

## Data Quality Metrics

### Comprehensive Data Quality Assessment
```python
def assess_data_quality(df):
    """Generate comprehensive data quality report"""
    
    quality_metrics = {}
    
    # Completeness
    quality_metrics['completeness'] = (1 - df.isnull().sum() / len(df)) * 100
    
    # Uniqueness (for categorical columns)
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        quality_metrics[f'{col}_uniqueness'] = unique_ratio
    
    # Validity (for numerical columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        # Check for infinite values
        inf_count = np.isinf(df[col]).sum()
        quality_metrics[f'{col}_infinite_values'] = inf_count
        
        # Check for extreme outliers (beyond 3 standard deviations)
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        extreme_outliers = (z_scores > 3).sum()
        quality_metrics[f'{col}_extreme_outliers'] = extreme_outliers
    
    return quality_metrics
```

## Real-World Challenges and Solutions

### Challenge 1: Mixed Data Types
```python
def handle_mixed_data_types(df):
    """Handle columns with mixed data types"""
    
    mixed_type_columns = []
    
    for col in df.columns:
        # Check if column has mixed types
        sample_values = df[col].dropna().head(1000)
        types = [type(val).__name__ for val in sample_values]
        
        if len(set(types)) > 1:
            mixed_type_columns.append(col)
            # Convert to string and handle later
            df[col] = df[col].astype(str)
    
    print(f"Mixed type columns found: {mixed_type_columns}")
    return df, mixed_type_columns
```

### Challenge 2: Large Dataset Processing
```python
def process_large_dataset(file_path, chunk_size=10000):
    """Process large datasets in chunks"""
    
    # Initialize EDA processor
    first_chunk = True
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if first_chunk:
            # Analyze first chunk to understand data structure
            eda = GeneralEDA(chunk)
            eda.data_info()
            first_chunk = False
        
        # Process chunk
        eda_chunk = GeneralEDA(chunk)
        eda_chunk.handle_missing_values()
        eda_chunk.handle_outliers()
        
        # Save processed chunk
        processed_chunk = eda_chunk.get_dataframe()
        # Save to database or file
        
        print(f"Processed chunk of size: {len(chunk)}")
```

## Assessment Checklist

- [ ] Can identify and assess data quality issues
- [ ] Successfully handles missing values using appropriate strategies
- [ ] Detects and treats outliers effectively
- [ ] Engineers meaningful features from raw data
- [ ] Uses automated EDA tools efficiently
- [ ] Builds reusable preprocessing pipelines
- [ ] Validates preprocessing results
- [ ] Completed practical projects on real datasets

## Estimated Time

**Total Duration:** 4-5 days (12-15 hours)

## Next Steps

After mastering data preprocessing and EDA, you'll be ready for Module 3: Supervised Learning, where you'll apply these data preparation skills to build and evaluate classification and regression models!

The clean, well-prepared datasets you create using these techniques will significantly improve your model performance in subsequent modules.