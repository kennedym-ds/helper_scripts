import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Union, Callable
import pickle
import os
from datetime import datetime
from tqdm import tqdm  # For progress bars
import yaml


class EnsembleFeatureImportance:
    """
    Calculates feature importance using multiple models, ensembles them, and builds a refined model.

    Args:
        data (pd.DataFrame): The input data.
        target_column (str): The name of the target variable column.
        models (List[Tuple[str, Any, Dict]], optional): List of models to use. Each tuple should contain
            (model_name, model_instance, hyperparameter_grid).  If None, uses a default set of models.
        top_n (int, optional): Number of top features to use in the final model. Defaults to 10.
        model_params_save_path (str, optional): Path to save model parameters. Defaults to "model_params.csv".
        best_model_save_path (str, optional): Path to save the best model. Defaults to "best_model.pkl".
        load_previous_best_models (bool, optional): flag to load any and all previously saved best models.
        scoring (Union[str, Callable, List, Dict], optional): Scoring metric(s) for model evaluation.
            Defaults to 'neg_root_mean_squared_error' and 'r2'.
        categorical_features (List[str], optional): List of categorical feature column names. If None,
            attempts to infer automatically.
        numerical_features (List[str], optional): List of numerical feature column names. If None,
            attempts to infer automatically.
        categorical_encoder (str, optional):  Encoding strategy for categorical features.
            Can be 'ordinal' (default), 'onehot', or 'target'.
        imputation_strategy (str, optional): Imputation strategy for missing values ('mean', 'median', 'most_frequent').
            Defaults to 'mean'.
        final_model (Any, optional):  The model to use for the final multivariate model.
            Defaults to GradientBoostingRegressor.
        final_model_param_grid (Dict, optional): Hyperparameter grid for the final model. Defaults to a basic grid.
        scaler_type (str, optional): Type of scaler for numerical features ('standard', 'minmax', or None). Defaults to 'standard'.
        config_file (str, optional): Path to a YAML configuration file.  If provided, settings in the config file
            will override other arguments.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.


    Attributes:
        (Same as in the original class, plus attributes for storing best models, hyperparameters, etc.)
    """

    def __init__(self, data: pd.DataFrame, target_column: str,
                 models: List[Tuple[str, Any, Dict]] = None,
                 top_n: int = 10,
                 model_params_save_path: str = "model_params.csv",
                 best_model_save_path: str = "best_model.pkl",
                 load_previous_best_models: bool = False,
                 scoring: Union[str, Callable, List, Dict] = None,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 categorical_encoder: str = 'ordinal',
                 imputation_strategy: str = 'mean',
                 final_model=None,
                 final_model_param_grid: Dict = None,
                 scaler_type: str = 'standard',
                 config_file: str = None,
                 random_state:int = 42):

        # Load configuration from file if provided
        if config_file:
            self._load_config(config_file)
        else:
            self.config = {}  # Initialize config as an empty dictionary

        self.data = data.copy()
        self.target_column = self._get_config_value('target_column', target_column)
        self.random_state = self._get_config_value('random_state', random_state)
        self.feature_columns = [col for col in self.data.columns if col != self.target_column]
        self.top_n = self._get_config_value('top_n', top_n)
        self.model_params_save_path = self._get_config_value('model_params_save_path', model_params_save_path)
        self.best_model_save_path = self._get_config_value('best_model_save_path', best_model_save_path)
        self.scoring = self._get_config_value('scoring', scoring or ['neg_root_mean_squared_error', 'r2'])  # Default scoring
        self.categorical_features = self._get_config_value('categorical_features', categorical_features)
        self.numerical_features = self._get_config_value('numerical_features', numerical_features)
        self.categorical_encoder = self._get_config_value('categorical_encoder', categorical_encoder)
        self.imputation_strategy = self._get_config_value('imputation_strategy', imputation_strategy)
        self.scaler_type = self._get_config_value('scaler_type', scaler_type)


        self._detect_data_types()  # Detect data types if not explicitly provided
        self.preprocessor = self._create_preprocessor()
        self.X, self.y = self._prepare_data()
        self.feature_importances = pd.DataFrame(index=self.X.columns)
        self.best_models = {}  # Store best models from each type
        self.best_hyperparams = {} # Store best hyperparameters

        # Model definitions
        default_models = [
            ('F_Test', f_regression, {}),  # No hyperparams for f_regression
            ('Mutual_Info', mutual_info_regression, {}), # No hyperparams for mutual_info_regression
            ('Lasso', Lasso(random_state=self.random_state), {'alpha': [0.01, 0.1, 1.0]}),
            ('Linear_Regression', LinearRegression(), {}), # No hyperparams needed
            ('Random_Forest', RandomForestRegressor(random_state=self.random_state),
             {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}),
            ('Extra_Trees', ExtraTreesRegressor(random_state=self.random_state),
             {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}),
            ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state),
             {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}),
        ]
        self.models = self._get_config_value('models', models or default_models)

        # Final Model
        if final_model is None:
          self.final_model = GradientBoostingRegressor(random_state=self.random_state)
          default_final_model_param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5]}
          self.final_model_param_grid = self._get_config_value('final_model_param_grid', final_model_param_grid or default_final_model_param_grid)
        else:
          self.final_model = final_model
          self.final_model_param_grid = self._get_config_value('final_model_param_grid',final_model_param_grid) #could be none

        self.previous_best_models = []
        if load_previous_best_models:
          self.load_all_previous_best_models()

    def _get_config_value(self, key, default_value):
        """Helper function to get values from config file, else use default."""
        return self.config.get(key, default_value)

    def _load_config(self, config_file: str) -> None:
        """Loads configuration from a YAML file."""
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def _detect_data_types(self) -> None:
        """Infers numerical and categorical features if not provided."""
        if self.numerical_features is None:
            self.numerical_features = self.data[self.feature_columns].select_dtypes(include=np.number).columns.tolist()
        if self.categorical_features is None:
            self.categorical_features = self.data[self.feature_columns].select_dtypes(exclude=np.number).columns.tolist()

        # Sanity Check in case user provided lists dont match available columns
        self.numerical_features = [col for col in self.numerical_features if col in self.feature_columns]
        self.categorical_features = [col for col in self.categorical_features if col in self.feature_columns]


    def _create_preprocessor(self) -> ColumnTransformer:
        """Creates a ColumnTransformer to handle numerical and categorical features."""

        transformers = []

        if self.numerical_features:
            if self.scaler_type == 'standard':
                num_transformer = StandardScaler()
            elif self.scaler_type == 'minmax':
                num_transformer = MinMaxScaler()
            else:
                num_transformer = 'passthrough'  # No scaling

            if self.imputation_strategy:
                num_pipeline = ('num', Pipeline([('imputer', SimpleImputer(strategy=self.imputation_strategy)),
                                              ('scaler', num_transformer)]), self.numerical_features)
            else:
                num_pipeline = ('num', num_transformer, self.numerical_features)
            transformers.append(num_pipeline)


        if self.categorical_features:
            if self.categorical_encoder == 'onehot':
                cat_transformer = OneHotEncoder(handle_unknown='ignore')
            elif self.categorical_encoder == 'target':
                #  Requires a TargetEncoder (not part of core sklearn) - you'd need to import it.
                #  cat_transformer = TargetEncoder()
                raise NotImplementedError("TargetEncoder not yet implemented. Use 'ordinal' or 'onehot'.")
            else:  # Default to ordinal
                cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

            if self.imputation_strategy:
                cat_pipeline = ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), # Categorical often uses most_frequent
                                        ('encoder', cat_transformer)]), self.categorical_features)
            else:
                cat_pipeline = ('cat', cat_transformer, self.categorical_features)
            transformers.append(cat_pipeline)

        return ColumnTransformer(transformers=transformers, remainder='passthrough')


    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the data, handling different data types."""
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]

        # Check for missing values BEFORE preprocessing
        if X.isnull().any().any():
            if self.imputation_strategy is None:
                raise ValueError("Missing values found in data, but no imputation strategy specified.")
            else:
                print(f"Missing values found. Applying imputation strategy: {self.imputation_strategy}")

        try:
            X = self.preprocessor.fit_transform(X)
            feature_names = self._get_preprocessor_feature_names()
            # Handle sparse matrices (from OneHotEncoder)
            if isinstance(X, np.ndarray):
                 X = pd.DataFrame(X, columns=feature_names)
            else:
                X = pd.DataFrame.sparse.from_spmatrix(X, columns=feature_names)


        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise

        return X, y

    def _get_preprocessor_feature_names(self):
        """Retrieves feature names after preprocessing, handling one-hot encoding."""
        feature_names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                feature_names.extend(trans.get_feature_names_out(cols))
            elif name == 'num':
                feature_names.extend(cols)  # Numeric features
            elif name == 'cat':
                if isinstance(trans, Pipeline):  # Handle pipelines
                  if isinstance(trans[-1], OneHotEncoder):
                    feature_names.extend(trans[-1].get_feature_names_out(cols))
                  else:
                      feature_names.extend(cols)
                elif isinstance(trans, OneHotEncoder):
                    feature_names.extend(trans.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)  # For ordinal and others
            elif name == 'remainder' and trans == 'passthrough':
                all_cols = self.data[self.feature_columns].columns.tolist()
                feature_names.extend([col for col in all_cols if col not in feature_names])
        return feature_names



    def _calculate_feature_importance(self, model: Any, model_name: str, param_grid: Dict) -> None:
        """Calculates feature importance for a given model, including hyperparameter tuning."""

        print(f"Calculating feature importance for {model_name}...")

        if callable(model):  # Function (like f_regression)
            scores = model(self.X, self.y)
            if isinstance(scores, tuple):
                scores = scores[0]
            self.feature_importances[model_name] = scores
            self._evaluate_model(model, model_name)  # Evaluate (handles function case)
            return

        # --- Hyperparameter Tuning ---
        if param_grid:
            if len(param_grid) > 5: # Use Randomized search for a large number of hyper parameters
              grid_search = RandomizedSearchCV(model, param_grid, scoring=self.scoring, cv=3, refit=False, n_jobs=-1, verbose=0, n_iter=10) # Reduced n_iter for speed
            else:
              grid_search = GridSearchCV(model, param_grid, scoring=self.scoring, cv=3, refit=False, n_jobs=-1, verbose=0)  # Reduced cv for speed

            # Handle multiple scoring metrics
            if isinstance(self.scoring, list) or isinstance(self.scoring, dict):
                grid_search.fit(self.X, self.y)
                # Select best model based on the *first* scoring metric.  This is a simplification.
                best_index = np.argmax(grid_search.cv_results_[f'mean_test_{self.scoring[0]}'])
                best_params = grid_search.cv_results_['params'][best_index]
                best_model = model.set_params(**best_params) # Use the best parameters
            else: # Single scoring metric
                grid_search.fit(self.X, self.y)
                best_params = grid_search.best_params_
                best_model = grid_search.best_estimator_

            best_model.fit(self.X, self.y)  # Refit with best params on full data
            self.best_models[model_name] = best_model  # Store best model
            self.best_hyperparams[model_name] = best_params # store hyper parameters

        else:  # No hyperparameter tuning
            best_model = model
            best_model.fit(self.X, self.y)
            self.best_models[model_name] = best_model # Store the model.
            self.best_hyperparams[model_name] = {} # Empty dict

        # --- Feature Importance Calculation ---
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importances[model_name] = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            self.feature_importances[model_name] = np.abs(best_model.coef_)
        else:
            print(f"Warning: Model {model_name} does not provide direct feature importances.")
            self.feature_importances[model_name] = np.nan # Or

# demo.py
import pandas as pd
import numpy as np
from ensemble_feature_importance_v2 import EnsembleFeatureImportance  #  file name
import yaml


def create_synthetic_data(num_samples=1000):
    """Creates synthetic data for demonstration, including missing values."""
    np.random.seed(42)
    data = pd.DataFrame({
        'Numerical_1': np.random.rand(num_samples) * 10,
        'Numerical_2': np.random.randn(num_samples) * 5 + 2,
        'Categorical_1': np.random.choice(['A', 'B', 'C', 'D'], size=num_samples), # More categories
        'Categorical_2': np.random.choice(['X', 'Y', 'Z', 'W'], size=num_samples),
        'Noise_1': np.random.rand(num_samples),
        'Noise_2': np.random.choice(['P', 'Q'], size=num_samples)
    })
    # Create a target variable with some relationships to the features
    data['Target'] = (
        2 * data['Numerical_1']
        + 0.5 * data['Numerical_2']
        + (data['Categorical_1'] == 'B').astype(int) * 3
        + (data['Categorical_1'] == 'D').astype(int) * -2  # Added interaction with 'D'
        + (data['Categorical_2'] == 'Y').astype(int) * -2
        + np.random.randn(num_samples) * 0.5  # Add some noise
    )

    # Introduce some missing values (NaNs)
    for col in ['Numerical_1', 'Categorical_2', 'Noise_1']:
        data.loc[data.sample(frac=0.05).index, col] = np.nan

    return data

def create_config_file(filename="config.yaml"):
    """Creates a sample configuration file."""
    config = {
        'target_column': 'Target',
        'top_n': 5,
        'model_params_save_path': 'model_params_config.csv',
        'best_model_save_path': 'best_model_config.pkl',
        'scoring': ['neg_root_mean_squared_error', 'r2'],
        'categorical_encoder': 'onehot',  # Use one-hot encoding
        'imputation_strategy': 'mean',
        'scaler_type': 'standard',  # Use standard scaling
        'models': [
            ('Lasso', {'name': 'sklearn.linear_model.Lasso', 'params': {'alpha': [0.01, 0.1, 1.0]}}),  # Use dictionaries
            ('Random_Forest', {'name': 'sklearn.ensemble.RandomForestRegressor', 'params': {'n_estimators': [50, 100], 'max_depth': [None, 5]}}),
            ('XGBoost', {'name': 'xgboost.XGBRegressor', 'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}}),
        ],
        'final_model': {'name': 'sklearn.ensemble.GradientBoostingRegressor'},
        'final_model_param_grid': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
         'random_state': 42
    }
    with open(filename, 'w') as f:
        yaml.dump(config, f)


def main():
    """Demonstrates the usage of the EnsembleFeatureImportance class."""

    # Create synthetic data
    data = create_synthetic_data()

    # Create a configuration file (optional, but demonstrates the functionality)
    create_config_file()

    # --- Using the config file ---
    print("-" * 30, "Using config file", "-" * 30)
    ensemble_config = EnsembleFeatureImportance(data, target_column='Target', config_file="config.yaml") # Load all settings.
    ensemble_config.run_ensemble()
    print("\nFeature Importances (from config):\n", ensemble_config.feature_importances)



    # --- Manual Initialization (without config file) ---
    print("-" * 30, "Manual Initialization", "-" * 30)
    # Define models with hyperparameter grids (using correct format)
    my_models = [
        ('Lasso', Lasso(random_state=42), {'alpha': [0.001, 0.01, 0.1, 1.0]}), # More alphas
        ('Ridge', LinearRegression(), {}),  # Add Ridge (no hyperparameters needed)
        ('Random_Forest', RandomForestRegressor(random_state=42), {'n_estimators': [20, 50, 100], 'max_depth': [None, 3, 6, 10]}),  # More max_depth options
        ('Extra_Trees', ExtraTreesRegressor(random_state=42), {'n_estimators': [20, 50, 100], 'max_depth': [None, 3, 6, 10]}),
        ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror', random_state=42), {'n_estimators': [20, 50, 100], 'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7]}), # More learning rates
    ]
    final_model_grid =  {'n_estimators': [150, 250], 'learning_rate': [0.02, 0.08], 'max_depth': [2, 4]}

    ensemble_manual = EnsembleFeatureImportance(
        data,
        target_column='Target',
        models=my_models,
        top_n=6,
        model_params_save_path='model_results_manual.csv',
        best_model_save_path='best_model_manual.pkl',
        load_previous_best_models=False,
        scoring='neg_mean_squared_error',  # Use a single scoring metric for simplicity
        categorical_features=['Categorical_1', 'Categorical_2'],  # Explicitly specify
        numerical_features=['Numerical_1', 'Numerical_2', 'Noise_1'],
        categorical_encoder='ordinal',  # Try ordinal encoding
        imputation_strategy='median',  # Use median imputation
        final_model=GradientBoostingRegressor(random_state=42), # Specify the final model
        final_model_param_grid= final_model_grid,
        scaler_type='minmax'  # Try MinMax scaling
    )

    ensemble_manual.run_ensemble() # Run all
    print("\nFeature Importances (manual):\n", ensemble_manual.feature_importances)


    # Demonstrate accessing best models and hyperparameters
    print("\nBest Hyperparameters (Manual Run):")
    for model_name, params in ensemble_manual.best_hyperparams.items():
        print(f"- {model_name}: {params}")

    # You would normally access the best final model like this:
    # best_final_model = ensemble_manual.best_final_model

    # Demonstrate loading previous best models
    # Create a dummy model and save
    ensemble_dummy = EnsembleFeatureImportance(data, target_column='Target', best_model_save_path='dummy.pkl', load_previous_best_models=False)
    ensemble_dummy.best_model = LinearRegression()
    ensemble_dummy.save_best_model()

    # Create a new one and load the previous models.
    ensemble_loader = EnsembleFeatureImportance(data, target_column='Target', best_model_save_path='best.pkl', load_previous_best_models=True)
    print(f"Loaded {len(ensemble_loader.previous_best_models)} previous models.")


if __name__ == "__main__":
    main()