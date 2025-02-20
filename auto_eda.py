import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ydata_profiling import ProfileReport  # Use 'from ydata_profiling import ProfileReport' if installed via ydata-profiling

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class GeneralEDA:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.report = {}
        sns.set(style="whitegrid")

    def generate_pandas_profiling_report(self, output_file="pandas_profiling_report.html", sample_fraction=0.1, **kwargs):
        """
        Generates a pandas_profiling (ydata_profiling) report on a sampled version of the data.

        Parameters:
        - output_file (str): Path to save the profiling report.
        - sample_fraction (float): Fraction of data to sample for profiling to speed up the process.
        - **kwargs: Additional keyword arguments to pass to ProfileReport.
        """
        print("Generating pandas_profiling report on sampled data...")
        # Sample the data to make profiling faster
        sample_df = self.df.sample(frac=sample_fraction, random_state=42)
        # Use reduced settings to make the profiling faster
        profile = ProfileReport(
            sample_df,
            minimal=True,  # Use minimal profiling to speed up the process
            explorative=True,
            **kwargs
        )
        profile.to_file(output_file=output_file)
        print(f"Pandas Profiling report saved to {output_file}")
        self.report['pandas_profiling'] = output_file

    
    def data_info(self):
        print("DataFrame Shape:", self.df.shape)
        print("\nData Types:\n", self.df.dtypes)
        print("\nMissing Values:\n", self.df.isnull().sum())
        self.report['data_info'] = {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
    
    def statistical_summary(self):
        desc = self.df.describe(include='all').to_dict()
        print("\nStatistical Summary:\n", self.df.describe(include='all'))
        self.report['statistical_summary'] = desc

    def list_non_numeric_columns(self):
        non_numeric = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Non-numeric columns ({len(non_numeric)}): {non_numeric}")
        return non_numeric

    def correlation_matrix(self, figsize=(12,10), save=False, filename="correlation_matrix.png"):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 20:
            numeric_cols = numeric_cols[:20]  # Limit to 20 features for correlation matrix
        numeric_df = self.df[numeric_cols]

        if numeric_df.empty:
            print("No numeric columns available for correlation matrix.")
            return
        
        corr = numeric_df.corr()
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        if save:
            plt.savefig(filename)
            print(f"Correlation matrix saved as {filename}")
        plt.close()  # Close the plot to free resources
        self.report['correlation_matrix'] = corr.to_dict()

    def pairplot(self, vars=None, hue=None, save=False, filename="pairplot.png"):
        sample_df = self.df.sample(frac=0.1, random_state=42)  # Use only 10% of data for pairplot
        sns.pairplot(sample_df, vars=vars, hue=hue)
        if save:
            plt.savefig(filename)
            print(f"Pairplot saved as {filename}")
        plt.close()  # Close the plot to free resources
    
    def univariate_analysis(self, column, plot_type='hist', bins=30, save=False, filename=None):
        plt.figure(figsize=(8,6))
        if plot_type == 'hist':
            sns.histplot(self.df[column].dropna(), bins=bins, kde=True)
        elif plot_type == 'box':
            sns.boxplot(x=self.df[column])
        elif plot_type == 'count':
            sns.countplot(x=self.df[column])
        else:
            print("Unsupported plot type")
            plt.close()
            return
        plt.title(f'Univariate Analysis of {column}')
        if save and filename:
            plt.savefig(filename)
            print(f"Univariate plot saved as {filename}")
        plt.close()  # Close the plot to free resources

    def validate_data_integrity(self):
        print("Validating Data Integrity...")
    
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print("\nMissing Values:\n", missing_values[missing_values > 0])
    
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        print(f"\nNumber of Duplicate Rows: {duplicate_count}")
    
        # Check data types
        print("\nData Types:\n", self.df.dtypes)

        # Check value ranges for numerical columns
        for col in self.df.select_dtypes(include=[np.number]).columns:
            print(f"\nRange for {col}: Min={self.df[col].min()}, Max={self.df[col].max()}")
    
        # Check unique values for categorical columns
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            print(f"\nUnique values in {col}:\n", self.df[col].unique())
    
        # Custom business rules (example: no negative age values)
        if 'age' in self.df.columns:
            invalid_ages = self.df[self.df['age'] < 0]
            print(f"\nInvalid ages found:\n{invalid_ages}")
    
        print("Data Integrity Validation Completed.")

    def bivariate_analysis(self, x, y, kind='scatter', hue=None, save=False, filename=None):
        plt.figure(figsize=(8,6))
        if kind == 'scatter':
            sns.scatterplot(data=self.df, x=x, y=y, hue=hue)
        elif kind == 'line':
            sns.lineplot(data=self.df, x=x, y=y, hue=hue)
        elif kind == 'reg':
            sns.regplot(data=self.df, x=x, y=y)
        else:
            print("Unsupported kind")
            plt.close()
            return
        plt.title(f'Bivariate Analysis: {x} vs {y}')
        if save and filename:
            plt.savefig(filename)
            print(f"Bivariate plot saved as {filename}")
        plt.close()  # Close the plot to free resources

    def diagnostic_checks(self):
        print("\n--- Diagnostic Checks ---")
        print("DataFrame Info:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print("\nData Types:")
        print(self.df.dtypes)
        print("-------------------------\n")

    def handle_duplicates(self):
        duplicate_count = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicate_count}")
        self.df.drop_duplicates(inplace=True)
        print("Duplicate rows removed.")
        self.report['duplicates'] = f"Removed {duplicate_count} duplicate rows."

    def handle_outliers(self, method='zscore', threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if method == 'zscore':
            for col in numeric_cols:
                outliers = np.abs(stats.zscore(self.df[col])) > threshold
                outlier_count = outliers.sum()
                self.df.loc[outliers, col] = self.df[col].median()  # Replace outliers with median
                print(f"Replaced {outlier_count} outliers in {col} using Z-score.")
        elif method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
                outlier_count = outliers.sum()
                self.df.loc[outliers, col] = self.df[col].median()  # Replace outliers with median
                print(f"Replaced {outlier_count} outliers in {col} using IQR method.")
        self.report['outliers_handling'] = "Outliers handled using {} method.".format(method)

    def feature_engineering(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical columns before encoding: {categorical_cols}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if i < 2 and j < 2:  # Limit to a few combinations for performance
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    new_col = f"{col1}_x_{col2}"
                    self.df[new_col] = self.df[col1] * self.df[col2]
                    print(f"Created feature: {new_col}")

        datetime_cols = self.df.select_dtypes(include=['datetime64[ns]', 'object']).columns.tolist()
        for col in datetime_cols:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.df[col+'_year'] = self.df[col].dt.year
                self.df[col+'_month'] = self.df[col].dt.month
                self.df[col+'_day'] = self.df[col].dt.day
                print(f"Extracted year, month, day from {col}")
            else:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.df[col+'_year'] = self.df[col].dt.year
                    self.df[col+'_month'] = self.df[col].dt.month
                    self.df[col+'_day'] = self.df[col].dt.day
                    print(f"Converted and extracted year, month, day from {col}")
                except:
                    print(f"Column {col} is not a datetime type and cannot be converted.")

        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical columns after datetime handling: {categorical_cols}")
        for col in categorical_cols:
            self.df[col] = self.df[col].astype('category')
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(col, axis=1, inplace=True)
            print(f"Encoded categorical feature: {col}")

        self.report['feature_engineering'] = "Performed interaction terms, datetime extraction, and categorical encoding."

    def handle_missing_values(self, strategy='mean'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        print(f"Handling missing values with strategy='{strategy}'")
        print(f"Number of numerical columns: {len(numeric_cols)}")
        print(f"Number of categorical columns: {len(categorical_cols)}")

        if len(numeric_cols) > 0:
            imputer_num = SimpleImputer(strategy=strategy)
            self.df[numeric_cols] = imputer_num.fit_transform(self.df[numeric_cols])
            print(f"Imputed numerical columns with {strategy}")
        else:
            print("No numerical columns to impute.")
        
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            self.df[categorical_cols] = imputer_cat.fit_transform(self.df[categorical_cols])
            print("Imputed categorical columns with most frequent value")
        else:
            print("No categorical columns to impute.")
        
        self.report['missing_values_handling'] = f"Imputed numerical with {strategy} and categorical with most frequent."

    def feature_scaling(self, method='standard'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            print("Unsupported scaling method")
            return
        self.df.loc[:, numeric_cols] = scaler.fit_transform(self.df[numeric_cols].values).astype(np.float64)
        print(f"Scaled numerical features using {method} scaling.")
        self.report['feature_scaling'] = f"Scaled numerical features using {method} scaling."
    
    def save_report(self, filepath="eda_report.txt"):
        with open(filepath, 'w') as f:
            f.write("EDA Report\n")
            f.write(f"Generated on: {datetime.now()}\n\n")
            f.write("Data Information:\n")
            f.write(str(self.report.get('data_info', {})) + "\n\n")
            f.write("Statistical Summary:\n")
            f.write(str(self.report.get('statistical_summary', {})) + "\n\n")
            f.write("Correlation Matrix:\n")
            f.write(str(self.report.get('correlation_matrix', {})) + "\n\n")
            f.write("Feature Engineering:\n")
            f.write(str(self.report.get('feature_engineering', '')) + "\n\n")
            f.write("Missing Values Handling:\n")
            f.write(str(self.report.get('missing_values_handling', '')) + "\n\n")
            f.write("Feature Scaling:\n")
            f.write(str(self.report.get('feature_scaling', '')) + "\n\n")
            f.write("Pandas Profiling Report:\n")
            f.write(str(self.report.get('pandas_profiling', '')) + "\n\n")
        print(f"Report saved to {filepath}")

    def get_dataframe(self):
        return self.df

    def detect_and_visualize_anomalies(self, contamination=0.05):
        print("Detecting anomalies using Isolation Forest...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print("No numeric columns available for anomaly detection.")
            return

        # Fit Isolation Forest to detect anomalies
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.df['anomaly'] = iso_forest.fit_predict(self.df[numeric_cols])
        anomalies = self.df[self.df['anomaly'] == -1]
        print(f"Number of anomalies detected: {len(anomalies)}")

        # Visualize anomalies
        plt.figure(figsize=(12, 8))
        for col in numeric_cols:
            plt.scatter(self.df.index, self.df[col], c=self.df['anomaly'], cmap='coolwarm', alpha=0.6)
            plt.xlabel('Index')
            plt.ylabel(col)
            plt.title(f'Anomaly Detection in {col}')
            plt.show()
            plt.close()
        
        self.df.drop(columns=['anomaly'], inplace=True)

# Load the dataset
df = pd.read_csv("iris_dataset.csv")

# Initialize EDA
eda = GeneralEDA(df)

# Validate data integrity
eda.validate_data_integrity()

# Initial EDA
eda.data_info()
eda.handle_missing_values(strategy='mean')
eda.statistical_summary()
eda.handle_duplicates()
eda.handle_outliers(method='zscore', threshold=3)

# Feature Engineering
eda.feature_engineering()
eda.diagnostic_checks()

# Scaling features after feature engineering
eda.feature_scaling(method='standard')

# Get the transformed dataframe for further analysis
transformed_df = eda.get_dataframe()

# Generate Pandas Profiling Report on transformed data
eda_transformed = GeneralEDA(transformed_df)
# Generate Pandas Profiling Report on transformed data with sampling and faster settings
eda_transformed.generate_pandas_profiling_report(
    output_file="transformed_pandas_profiling_report.html",
    sample_fraction=0.05,  # Profile only 5% of the dataset
    title="Transformed Flight Dataset Profiling Report"
)

# Detect and visualize anomalies in the data
eda_transformed.detect_and_visualize_anomalies(contamination=0.05)

# Save the EDA report
eda_transformed.save_report(filepath="eda_report.txt")

# Display the transformed dataframe for modeling
print(transformed_df.head())

