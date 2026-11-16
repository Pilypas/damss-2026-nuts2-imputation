"""
================================================================================
XGBOOST IMPUTATION MODEL
================================================================================
Imputation of missing economic indicator values at the NUTS 2 level using the
XGBoost (Extreme Gradient Boosting) algorithm (xgboost XGBRegressor).

A separate XGBoost model is trained for each economic indicator with
missing values. The other indicators (and the categorical identifiers 'geo'
and 'year') are used as predictors.

RULES IMPLEMENTED BY THE CODE:
------------------------------
1. Structural zeros (0) are NEVER imputed - only NaN values are imputed.
2. Zero values are used as TRAINING data (real information), but NOT as TEST data.
3. Categorical predictors ('geo', 'year') contain no missing values - they are only encoded.
4. Test data without leakage: 20% of the known (non-zero) target values form the TEST set.
5. Predictor imputation (mean) ignores zero values.
6. The sMAPE metric is used (it is defined for zero values).

CODE STRUCTURE:
---------------
1. IMPORTS
2. HELPER CLASS: ZeroIgnoringImputer
3. MAIN CLASS: XGBoostImputer
   3.1. Initialization (__init__)
   3.2. Public API
   3.3. Data preparation
   3.4. Column imputation
   3.5. Train/test split
   3.6. Model training, cross-validation and hyperparameter optimization
   3.7. Feature transformations
   3.8. Final imputation
   3.9. Metric calculation and storage
================================================================================
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("The xgboost library is not installed. Install it with: pip install xgboost") from e


# ==============================================================================
# 2. HELPER CLASS: ZeroIgnoringImputer
# ==============================================================================
# Used internally to fill NaN values of the predictors.
# Computes the column mean while ignoring zero and NaN values.

class ZeroIgnoringImputer:
    """
    Mean imputer that ignores zero values when computing the mean.

    Structural zeros should not influence predictor imputation, because they
    often mean "no data" or "not applicable".

    Methods:
        fit(X): computes the mean ignoring NaN and 0
        transform(X): fills NaN with the computed means
        fit_transform(X): fit and transform in one step
    """

    def __init__(self):
        self.statistics_ = None

    def fit(self, X):
        """Computes the mean of each column ignoring NaN and 0 values."""
        X_array = self._to_array(X)
        self.statistics_ = self._compute_nonzero_means(X_array)
        return self

    def transform(self, X):
        """Fills NaN values with the computed means."""
        X_array = self._to_array(X)
        return self._impute_nans(X_array)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    @staticmethod
    def _to_array(X):
        """Converts a DataFrame to a numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values.copy()
        return np.array(X, copy=True)

    def _compute_nonzero_means(self, X_array):
        """Computes the mean of each column ignoring 0 and NaN."""
        means = []
        for col_idx in range(X_array.shape[1]):
            col = X_array[:, col_idx]
            valid_mask = ~np.isnan(col) & (col != 0)

            if np.any(valid_mask):
                means.append(np.mean(col[valid_mask]))
            else:
                # If all values are NaN or 0, 0 is used as the default
                means.append(0.0)

        return np.array(means)

    def _impute_nans(self, X_array):
        """Fills NaN values with the column means."""
        for col_idx in range(X_array.shape[1]):
            nan_mask = np.isnan(X_array[:, col_idx])
            X_array[nan_mask, col_idx] = self.statistics_[col_idx]
        return X_array


# ==============================================================================
# 3. MAIN CLASS: XGBoostImputer
# ==============================================================================

class XGBoostImputer:
    """
    XGBoost imputation of economic indicators.

    A separate XGBRegressor model is trained for each column with missing values.
    Test data: 20% of the known target values -> TEST (only rows where target != 0).
    After evaluation, the final model is retrained on 100% of the known target values.

    Usage:
        imputer = XGBoostImputer(cv_folds=5)
        df_imputed = imputer.fit_and_impute(df)
        imputer.print_cv_results()
    """

    # ==========================================================================
    # 3.1. INITIALISATION
    # ==========================================================================

    def __init__(
        self,
        n_estimators=200,
        random_state=42,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        categorical_cols=None,
        exclude_columns=None,
        cv_folds=2,
        use_hyperopt=False,
        hyperopt_n_iter=30,
        hyperopt_cv=3,
    ):
        """
        Initializes the XGBoost imputer.

        Args:
            n_estimators: number of boosting iterations (trees)
            random_state: random number generator seed
            learning_rate: learning rate (eta)
            max_depth: maximum tree depth
            subsample: row subsampling ratio for each iteration
            colsample_bytree: column subsampling ratio for each tree
            min_child_weight: minimum sum of instance weight in a child
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            categorical_cols: list of categorical columns
            exclude_columns: columns that will not be imputed
            cv_folds: number of cross-validation folds
            use_hyperopt: whether hyperparameter optimization is used
            hyperopt_n_iter: number of hyperparameter search iterations
            hyperopt_cv: number of CV folds used in the hyperparameter search
        """
        # ----- Baseline XGBoost hyperparameters -----
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        # ----- Data configuration -----
        self.categorical_cols = categorical_cols or ['geo', 'year']
        self.exclude_columns = exclude_columns or []

        # ----- Cross-validation -----
        self.cv_folds = cv_folds

        # ----- Hyperparameter optimization -----
        self.use_hyperopt = use_hyperopt
        self.hyperopt_n_iter = hyperopt_n_iter
        self.hyperopt_cv = hyperopt_cv

        # ----- Results -----
        self.models = {}                # trained model for each indicator
        self.feature_importance = {}    # feature importance values
        self.model_metrics = {}         # test set metrics (R², nRMSE, nMAE, sMAPE)
        self.test_predictions = {}      # test set predictions (actual vs predicted)
        self.cv_scores = {}             # cross-validation results
        self.best_params = {}           # best hyperparameters (if hyperopt is enabled)

    # ==========================================================================
    # 3.2. PUBLIC API
    # ==========================================================================

    def fit_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method: trains the models and imputes the missing values.

        IMPORTANT: the ORIGINAL values are used as features, not already imputed
        values. This avoids the "cascade error" problem, where the errors of one
        column propagate to other columns. https://arxiv.org/pdf/1105.0828

        Steps:
            1. Prepares the DataFrame (copy, type conversion)
            2. Validates the categorical columns
            3. Stores the original values for use as features
            4. Imputes every column that contains NaN values

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame with imputed values (only NaN, not 0)
        """
        # Step 1: prepare the DataFrame
        df_work = self._prepare_dataframe(df)

        # Step 2: validate the categorical columns
        self._validate_categorical_columns(df_work)

        # Step 3: store the original values used as features, so that each column
        # is imputed ONLY from original values, not from values imputed for other columns
        self._df_original = df_work.copy()

        # Step 4: impute every column with NaN values
        for target_col in df_work.columns:
            if self._should_impute_column(df_work, target_col):
                self._impute_column(df_work, target_col)

        # Step 5: free memory
        del self._df_original

        return df_work

    def get_feature_importance(self):
        """Returns {indicator: {feature: importance}}."""
        return self.feature_importance

    def get_model_metrics(self):
        """Returns the test set metrics {indicator: {nrmse, r2, nmae, smape, ...}}."""
        return self.model_metrics

    def get_test_predictions(self, df=None):
        """Returns the test set predictions {indicator: {y_true, y_pred, test_indices, ...}}."""
        return self.test_predictions

    def get_cv_scores(self):
        """Returns the cross-validation results (means and standard deviations)."""
        return self.cv_scores

    def get_params(self):
        """Returns the model hyperparameters as a dictionary."""
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'cv_folds': self.cv_folds,
            'use_hyperopt': self.use_hyperopt,
        }
        if self.use_hyperopt:
            params['hyperopt_n_iter'] = self.hyperopt_n_iter
            params['hyperopt_cv'] = self.hyperopt_cv
        return params

    def get_best_params(self):
        """Returns the best hyperparameters for each indicator after optimization."""
        return self.best_params

    def get_best_params_summary(self):
        """
        Returns a summary of the selected hyperparameter values across indicators.

        Returns:
            dict: statistics of the best parameters
        """
        if not self.best_params:
            return {}

        summary = {
            'n_estimators': [],
            'learning_rate': [],
            'max_depth': [],
            'subsample': [],
            'colsample_bytree': [],
            'min_child_weight': [],
            'reg_alpha': [],
            'reg_lambda': []
        }

        for params in self.best_params.values():
            for key in summary.keys():
                if key in params:
                    summary[key].append(params[key])

        result = {}
        for key, values in summary.items():
            if values:
                result[key] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'median': np.median(values)
                }

        return result

    def print_params(self):
        """Prints the model hyperparameters."""
        params = self.get_params()
        print("XGBoost hyperparameters:")
        for name, value in params.items():
            print(f"  - {name}: {value}")

        if self.use_hyperopt:
            print("\nHyperparameter optimization ENABLED (RandomizedSearchCV)")
            print(f"  - Number of iterations: {self.hyperopt_n_iter}")
            print(f"  - CV folds used in the search: {self.hyperopt_cv}")

    def print_best_params(self):
        """Prints the best hyperparameters found by the optimization."""
        if not self.best_params:
            print("Hyperparameter optimization was not performed.")
            return

        print("=" * 80)
        print("BEST HYPERPARAMETERS FOUND")
        print("=" * 80)

        summary = self.get_best_params_summary()
        if summary:
            print("\nSUMMARY (across all indicators):")
            print("-" * 50)
            for param, stats in summary.items():
                print(f"  {param}: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                      f"mean={stats['mean']:.4f}, median={stats['median']:.4f}")

        print(f"\nNumber of optimized indicators: {len(self.best_params)}")
        print("=" * 80)

    def get_cv_metrics_df(self):
        """
        Returns the CV metrics as a DataFrame (mean and standard deviation across folds).

        Returns:
            pd.DataFrame: CV metrics, also formatted as 'mean ± std'
        """
        if not self.cv_scores:
            return pd.DataFrame()

        metrics_list = []
        for indicator, scores in self.cv_scores.items():
            if scores['cv_folds'] > 0:
                metrics_list.append({
                    'Indicator': indicator,
                    'R²': scores['r2_mean'],
                    'R²_std': scores['r2_std'],
                    'R²_formatted': f"{scores['r2_mean']:.4f} ± {scores['r2_std']:.4f}",
                    'nRMSE': scores['nrmse_mean'],
                    'nRMSE_std': scores['nrmse_std'],
                    'nRMSE_formatted': f"{scores['nrmse_mean']:.4f} ± {scores['nrmse_std']:.4f}",
                    'nMAE': scores['nmae_mean'],
                    'nMAE_std': scores['nmae_std'],
                    'nMAE_formatted': f"{scores['nmae_mean']:.4f} ± {scores['nmae_std']:.4f}",
                    'sMAPE (%)': scores['smape_mean'],
                    'sMAPE_std': scores['smape_std'],
                    'sMAPE_formatted': f"{scores['smape_mean']:.2f} ± {scores['smape_std']:.2f}",
                    'CV_folds': scores['cv_folds'],
                    'Sample size': scores['sample_size']
                })

        return pd.DataFrame(metrics_list)

    def get_cv_summary(self):
        """
        Returns the overall CV summary (means across all indicators).

        The '± std' values are the standard deviations of the per-indicator CV means.

        Returns:
            dict: overall CV means and standard deviations
        """
        if not self.cv_scores:
            return {}

        all_r2 = [s['r2_mean'] for s in self.cv_scores.values()
                  if s['cv_folds'] > 0 and not np.isnan(s.get('r2_mean', float('nan')))]
        all_nrmse = [s['nrmse_mean'] for s in self.cv_scores.values()
                     if s['cv_folds'] > 0 and not np.isnan(s.get('nrmse_mean', float('nan')))]
        all_nmae = [s['nmae_mean'] for s in self.cv_scores.values()
                    if s['cv_folds'] > 0 and not np.isnan(s.get('nmae_mean', float('nan')))]
        all_smape = [s['smape_mean'] for s in self.cv_scores.values()
                     if s['cv_folds'] > 0 and not np.isnan(s.get('smape_mean', float('nan')))]

        return {
            'r2_mean': float(np.mean(all_r2)) if all_r2 else float('nan'),
            'r2_std': float(np.std(all_r2)) if all_r2 else float('nan'),
            'nrmse_mean': float(np.mean(all_nrmse)) if all_nrmse else float('nan'),
            'nrmse_std': float(np.std(all_nrmse)) if all_nrmse else float('nan'),
            'nmae_mean': float(np.mean(all_nmae)) if all_nmae else float('nan'),
            'nmae_std': float(np.std(all_nmae)) if all_nmae else float('nan'),
            'smape_mean': float(np.mean(all_smape)) if all_smape else float('nan'),
            'smape_std': float(np.std(all_smape)) if all_smape else float('nan'),
            'n_indicators': len(all_r2)
        }

    def print_cv_results(self):
        """Prints the cross-validation results."""
        cv_df = self.get_cv_metrics_df()
        summary = self.get_cv_summary()

        if cv_df.empty:
            print("No CV results available.")
            return

        print("=" * 90)
        print(f"CROSS-VALIDATION RESULTS ({self.cv_folds}-fold CV)")
        print("=" * 90)

        print("\nOVERALL MEANS (across all indicators):")
        print("-" * 50)
        print(f"  R²:    {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
        print(f"  nRMSE: {summary['nrmse_mean']:.4f} ± {summary['nrmse_std']:.4f}")
        print(f"  nMAE:  {summary['nmae_mean']:.4f} ± {summary['nmae_std']:.4f}")
        print(f"  sMAPE: {summary['smape_mean']:.2f}% ± {summary['smape_std']:.2f}%")

        print("\nCV RESULTS BY INDICATOR:")
        print("-" * 90)
        for _, row in cv_df.iterrows():
            print(f"\n{row['Indicator']}:")
            print(f"  R²:    {row['R²_formatted']}")
            print(f"  nRMSE: {row['nRMSE_formatted']}")
            print(f"  nMAE:  {row['nMAE_formatted']}")
            print(f"  sMAPE: {row['sMAPE_formatted']}%")
        print("\n" + "=" * 90)

    # ==========================================================================
    # 3.3. DATA PREPARATION (PRIVATE)
    # ==========================================================================

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the DataFrame for imputation.

        Steps:
            1. Creates a copy of the DataFrame
            2. Converts non-categorical 'object' columns to numeric
        """
        df_work = df.copy()

        for col in df_work.columns:
            if (df_work[col].dtype == 'object') and (col not in self.categorical_cols):
                df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        return df_work

    def _validate_categorical_columns(self, df: pd.DataFrame):
        """
        Checks that the categorical columns contain no NaN values
        ('geo' and 'year' are always present in the dataset).

        Raises:
            ValueError: if a categorical column contains NaN
        """
        for col in self.categorical_cols:
            if col in df.columns and df[col].isna().any():
                raise ValueError(
                    f"Categorical column '{col}' contains missing values, "
                    "which is not expected for this dataset."
                )

    def _should_impute_column(self, df: pd.DataFrame, col: str) -> bool:
        """
        A column is imputed if it:
            1. is not categorical ('geo' or 'year'),
            2. is not in the exclusion list,
            3. contains at least one NaN value.
        """
        return (
            col not in self.categorical_cols and
            col not in self.exclude_columns and
            df[col].isna().any()
        )

    # ==========================================================================
    # 3.4. COLUMN IMPUTATION
    # ==========================================================================

    def _impute_column(self, df_work: pd.DataFrame, target_col: str):
        """
        Imputes a single column with missing values.

        IMPORTANT: self._df_original (original values) is used for the features
        to avoid "cascade error".

        Steps:
            1. Creates the test data (20%) without zero values
            2. Trains and evaluates the model (test set + k-fold CV)
            3. Retrains the model on 100% of the known values
            4. Imputes the NaN values

        Args:
            df_work: DataFrame in which the NaN values are filled
            target_col: column to impute
        """
        # ORIGINAL values are used for the features
        df_features = self._df_original.copy()

        feature_cols = [c for c in df_features.columns if c != target_col]
        known_mask = df_features[target_col].notna()

        # Fallback to the mean if there are too few known values
        if known_mask.sum() < 5:
            self._fallback_mean_impute(df_work, target_col, known_mask)
            return

        # Prepare the data
        X_all_raw = df_features.loc[known_mask, feature_cols].copy()
        y_all = df_features.loc[known_mask, target_col].copy()

        # Train/test split
        train_mask, test_mask = self._create_train_test_split(y_all)

        # Model training
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            self._train_without_test(X_all_raw, y_all, target_col, feature_cols)
        else:
            self._train_with_test(
                X_all_raw, y_all, train_mask, test_mask,
                target_col, feature_cols
            )

        # Final imputation
        self._perform_final_imputation(df_work, df_features, target_col, feature_cols)

        # Store feature importance
        self._save_feature_importance(target_col)

    def _fallback_mean_impute(self, df: pd.DataFrame, target_col: str, known_mask):
        """
        Fallback for columns with fewer than 5 known values: mean + year-based variation.
        (Not triggered for the dataset used in this study.)
        """
        known_values = df.loc[known_mask, target_col]
        fill_value = known_values.mean()
        fill_std = known_values.std() if len(known_values) > 1 else abs(fill_value) * 0.1

        if fill_std < 1e-10:
            fill_std = abs(fill_value) * 0.1 if fill_value != 0 else 1.0

        missing_mask = ~known_mask

        if 'year' in df.columns and missing_mask.any():
            years = df.loc[missing_mask, 'year'].values
            mean_year = int(np.median(df['year'].values))

            imputed_values = []
            for i, year in enumerate(years):
                geo = df.loc[missing_mask].iloc[i]['geo'] if 'geo' in df.columns else 'unknown'
                seed = hash(f"{geo}_{year}_{target_col}_fallback_v4") % 100000
                np.random.seed(seed)

                jitter_pct = 0.08 + (year - 2000) * 0.003
                jitter_base = fill_std * min(jitter_pct, 0.15)
                year_offset = (year - mean_year) * fill_std * 0.02
                random_jitter = np.random.normal(0, jitter_base)
                total_variation = random_jitter + year_offset

                imputed_values.append(fill_value + total_variation)

            df.loc[missing_mask, target_col] = imputed_values
        else:
            df.loc[missing_mask, target_col] = fill_value

        self.model_metrics[target_col] = {
            'nrmse': float('nan'),
            'r2': float('nan'),
            'nmae': float('nan'),
            'smape': float('nan'),
            'model_type': 'insufficient_data_mean_impute_v4',
            'sample_size': int(known_mask.sum()),
            'total_samples': int(known_mask.sum())
        }

    # ==========================================================================
    # 3.5. TRAIN/TEST SPLIT
    # ==========================================================================

    def _create_train_test_split(self, y_all: pd.Series):
        """
        Creates the train/test split; zero values are excluded from the test set.

        Zero values are returned to the training set (real information) but are
        not used for testing (to avoid bias). https://arxiv.org/pdf/2106.04525

        Returns:
            tuple: (train_mask, test_mask) boolean arrays
        """
        all_indices = y_all.index.to_numpy()

        # Initial 80/20 split
        train_mask, test_mask = self._split_indices(all_indices, test_frac=0.2)

        # Remove zero values from the test set
        zero_mask = (y_all.values == 0)
        test_mask = np.logical_and(test_mask, ~zero_mask)
        train_mask = np.logical_or(train_mask, zero_mask)

        return train_mask, test_mask

    def _split_indices(self, indices, test_frac=0.2):
        """
        Randomly assigns the indices to train/test.

        Args:
            indices: array of indices
            test_frac: test proportion (default 0.2 = 20%)

        Returns:
            tuple: (train_mask, test_mask)
        """
        rng = np.random.RandomState(self.random_state)

        if len(indices) == 0:
            return np.array([], dtype=bool), np.array([], dtype=bool)

        n_test = max(1, int(len(indices) * test_frac))
        n_test = min(n_test, len(indices))

        test_idx = rng.choice(indices, size=n_test, replace=False)
        test_mask = pd.Index(indices).isin(test_idx)
        train_mask = ~test_mask

        return train_mask, test_mask

    # ==========================================================================
    # 3.6. MODEL TRAINING, CROSS-VALIDATION AND HYPERPARAMETER OPTIMISATION
    # ==========================================================================

    def _train_without_test(self, X_all_raw, y_all, target_col, feature_cols):
        """
        Trains the model without test set evaluation
        (used when there are not enough non-zero values for a test set).
        """
        num_cols, cat_cols = self._get_column_groups(X_all_raw, feature_cols)

        num_imputer, encoder = self._fit_transformers(X_all_raw, num_cols, cat_cols)
        X_all = self._apply_transformations(X_all_raw, num_cols, cat_cols, num_imputer, encoder)

        if self.use_hyperopt:
            best_params = self._perform_hyperopt(X_all, y_all.values, target_col)
        else:
            best_params = None

        model = self._create_model_with_params(best_params)
        model.fit(X_all, y_all.values)

        self._save_model(target_col, model, num_cols, cat_cols, num_imputer, encoder)

        self.model_metrics[target_col] = {
            'nrmse': float('nan'),
            'r2': float('nan'),
            'nmae': float('nan'),
            'smape': float('nan'),
            'model_type': 'no_test_nonzero',
            'sample_size': int(len(y_all)),
            'total_samples': int(len(y_all))
        }

    def _train_with_test(
        self, X_all_raw, y_all, train_mask, test_mask,
        target_col, feature_cols
    ):
        """
        Trains the model with test set evaluation.

        Steps:
            1. Splits the data into train/test
            2. Fits the transformers on ALL known data
            3. Performs hyperparameter optimization (if enabled)
            4. Trains the evaluation model on TRAIN
            5. Evaluates it on TEST
            6. Performs k-fold cross-validation on TRAIN
            7. Trains the final model on 100% of the known data
        """
        # Step 1: split the data
        X_train_raw = X_all_raw.loc[train_mask].copy()
        X_test_raw = X_all_raw.loc[test_mask].copy()
        y_train = y_all.loc[train_mask].copy()
        y_test = y_all.loc[test_mask].copy()

        num_cols, cat_cols = self._get_column_groups(X_all_raw, feature_cols)

        # Step 2: transformers fitted on ALL known data
        num_imputer_full, encoder_full = self._fit_transformers(X_all_raw, num_cols, cat_cols)
        X_all = self._apply_transformations(X_all_raw, num_cols, cat_cols, num_imputer_full, encoder_full)

        # Step 3: hyperparameter optimization (if enabled)
        if self.use_hyperopt:
            best_params = self._perform_hyperopt(X_all, y_all.values, target_col)
        else:
            best_params = None

        # Step 4: evaluation transformers (fitted on TRAIN only)
        num_imputer_eval, encoder_eval = self._fit_transformers(X_train_raw, num_cols, cat_cols)
        X_train = self._apply_transformations(X_train_raw, num_cols, cat_cols, num_imputer_eval, encoder_eval)
        X_test = self._apply_transformations(X_test_raw, num_cols, cat_cols, num_imputer_eval, encoder_eval)

        # Step 5: train the evaluation model
        model_eval = self._create_model_with_params(best_params)
        model_eval.fit(X_train, y_train.values)
        y_pred = model_eval.predict(X_test)

        # Step 6: test set metrics
        self._save_test_metrics(target_col, y_test, y_pred, len(y_all))

        # Step 7: k-fold cross-validation
        self._perform_cross_validation(X_train, y_train.values, target_col, best_params)

        # Step 8: final model on 100% of the known data
        model_full = self._create_model_with_params(best_params)
        model_full.fit(X_all, y_all.values)

        # Step 9: store the final model
        self._save_model(target_col, model_full, num_cols, cat_cols, num_imputer_full, encoder_full)

    def _perform_cross_validation(self, X, y, target_col, params=None):
        """
        Performs k-fold cross-validation and stores the results.

        All four metrics are computed in each fold: R², nRMSE, nMAE, sMAPE.
        nRMSE and nMAE are normalized by the range (max - min) of the actual
        values in the validation fold.

        Args:
            X: transformed feature matrix (numpy array)
            y: target values (numpy array)
            target_col: column name
            params: hyperparameters (if None, the baseline values are used)
        """
        cv_folds = min(self.cv_folds, len(y))
        if cv_folds < 2:
            self.cv_scores[target_col] = {
                'r2_scores': [], 'r2_mean': float('nan'), 'r2_std': float('nan'),
                'nrmse_scores': [], 'nrmse_mean': float('nan'), 'nrmse_std': float('nan'),
                'nmae_scores': [], 'nmae_mean': float('nan'), 'nmae_std': float('nan'),
                'smape_scores': [], 'smape_mean': float('nan'), 'smape_std': float('nan'),
                'cv_folds': 0, 'sample_size': len(y)
            }
            return

        r2_scores, nrmse_scores, nmae_scores, smape_scores = [], [], [], []

        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kfold.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            model_cv = self._create_model_with_params(params)
            model_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = model_cv.predict(X_val_cv)

            # Metrics
            r2 = r2_score(y_val_cv, y_pred_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
            mae = np.mean(np.abs(y_val_cv - y_pred_cv))
            smape = self._calculate_smape(y_val_cv, y_pred_cv)

            # Normalization by the range of the actual values
            y_range = float(np.max(y_val_cv) - np.min(y_val_cv))
            nrmse = rmse / y_range if y_range > 0 else rmse
            nmae = mae / y_range if y_range > 0 else mae

            r2_scores.append(r2)
            nrmse_scores.append(nrmse)
            nmae_scores.append(nmae)
            smape_scores.append(smape)

        self.cv_scores[target_col] = {
            'r2_scores': r2_scores, 'r2_mean': float(np.mean(r2_scores)), 'r2_std': float(np.std(r2_scores)),
            'nrmse_scores': nrmse_scores, 'nrmse_mean': float(np.mean(nrmse_scores)), 'nrmse_std': float(np.std(nrmse_scores)),
            'nmae_scores': nmae_scores, 'nmae_mean': float(np.mean(nmae_scores)), 'nmae_std': float(np.std(nmae_scores)),
            'smape_scores': smape_scores, 'smape_mean': float(np.mean(smape_scores)), 'smape_std': float(np.std(smape_scores)),
            'cv_folds': cv_folds, 'sample_size': len(y)
        }

    def _create_model_with_params(self, params=None):
        """
        Creates an XGBoost model with the given hyperparameters.

        Args:
            params: dict of hyperparameters or None (baseline values are used)

        Returns:
            XGBRegressor: created (untrained) model
        """
        if params is not None:
            return xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', self.n_estimators),
                random_state=self.random_state,
                learning_rate=params.get('learning_rate', self.learning_rate),
                max_depth=params.get('max_depth', self.max_depth),
                subsample=params.get('subsample', self.subsample),
                colsample_bytree=params.get('colsample_bytree', self.colsample_bytree),
                min_child_weight=params.get('min_child_weight', self.min_child_weight),
                reg_alpha=params.get('reg_alpha', self.reg_alpha),
                reg_lambda=params.get('reg_lambda', self.reg_lambda),
                verbosity=0,
                n_jobs=2,
                eval_metric='rmse'
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                verbosity=0,
                n_jobs=2,
                eval_metric='rmse'
            )

    def _get_param_distributions(self):
        """
        Returns the hyperparameter search space for RandomizedSearchCV.
        """
        return {
            'n_estimators': randint(50, 301),
            'learning_rate': uniform(0.01, 0.29),
            'max_depth': randint(3, 16),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 11),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }

    def _perform_hyperopt(self, X, y, target_col):
        """
        Performs hyperparameter optimization with RandomizedSearchCV (scoring = R²).

        Args:
            X: feature matrix (numpy array)
            y: target values (numpy array)
            target_col: column name

        Returns:
            dict: best hyperparameters found
        """
        base_model = xgb.XGBRegressor(
            random_state=self.random_state,
            verbosity=0,
            n_jobs=2,
            eval_metric='rmse'
        )

        param_distributions = self._get_param_distributions()

        cv_folds = min(self.hyperopt_cv, len(y))
        if cv_folds < 2:
            return {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda
            }

        n_iter = min(self.hyperopt_n_iter, 50) if len(y) < 100 else self.hyperopt_n_iter

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',
            random_state=self.random_state,
            n_jobs=2,
            verbose=0,
            return_train_score=False
        )

        random_search.fit(X, y)

        best_params = random_search.best_params_
        self.best_params[target_col] = best_params

        return best_params

    # ==========================================================================
    # 3.7. FEATURE TRANSFORMATIONS
    # ==========================================================================

    def _get_column_groups(self, X: pd.DataFrame, feature_cols):
        """
        Splits the feature columns into numeric and categorical columns.

        Returns:
            tuple: (num_cols, cat_cols)
        """
        num_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(X[c]) and c not in self.categorical_cols
        ]
        cat_cols = [c for c in self.categorical_cols if c in feature_cols]
        return num_cols, cat_cols

    def _fit_transformers(self, X_raw, num_cols, cat_cols):
        """
        Creates and fits the transformers.

        Returns:
            tuple: (num_imputer, encoder)
        """
        # Numeric column imputer (ignores zeros)
        num_imputer = None
        if num_cols:
            num_imputer = ZeroIgnoringImputer()
            num_imputer.fit(X_raw[num_cols])

        # Categorical column encoder
        encoder = None
        if cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            encoder.fit(X_raw[cat_cols])

        return num_imputer, encoder

    def _apply_transformations(self, X_raw, num_cols, cat_cols, num_imputer, encoder):
        """
        Applies the transformations and returns a numpy array.

        Returns:
            np.ndarray: transformed feature matrix
        """
        if num_cols and num_imputer:
            X_num = num_imputer.transform(X_raw[num_cols])
        else:
            X_num = np.empty((len(X_raw), 0))

        if cat_cols and encoder:
            X_cat = encoder.transform(X_raw[cat_cols])
        else:
            X_cat = np.empty((len(X_raw), 0))

        return np.hstack([X_num, X_cat])

    # ==========================================================================
    # 3.8. FINAL IMPUTATION
    # ==========================================================================

    def _perform_final_imputation(self, df: pd.DataFrame, df_features: pd.DataFrame,
                                  target_col: str, feature_cols: list):
        """
        Fills the NaN values of the target column with the predictions of the
        final model. Zero values are NOT imputed - they remain 0.
        """
        missing_mask = df[target_col].isna()

        if not missing_mask.any():
            return

        X_missing_raw = df_features.loc[missing_mask, feature_cols].copy()
        model_info = self.models[target_col]

        X_missing = self._apply_transformations(
            X_missing_raw,
            model_info['num_cols'],
            model_info['cat_cols'],
            model_info['num_imputer'],
            model_info['encoder']
        )

        predictions = model_info['model'].predict(X_missing)
        df.loc[missing_mask, target_col] = predictions

    # ==========================================================================
    # 3.9. METRIC CALCULATION AND STORAGE
    # ==========================================================================

    def _save_test_metrics(self, target_col: str, y_test, y_pred, total_samples: int):
        """
        Computes and stores the test set metrics (R², nRMSE, nMAE, sMAPE)
        and the test set predictions.
        """
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(np.mean(np.abs(y_test.values - y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        smape = self._calculate_smape(y_test.values, y_pred)

        y_range = float(y_test.max() - y_test.min())
        if y_range > 0:
            nrmse = rmse / y_range
            nmae = mae / y_range
        else:
            nrmse = rmse
            nmae = mae

        self.model_metrics[target_col] = {
            'nrmse': nrmse,
            'r2': r2,
            'nmae': nmae,
            'smape': smape,
            'model_type': 'synthetic_test_no_zeros',
            'sample_size': int(len(y_test)),
            'total_samples': int(total_samples)
        }

        self.test_predictions[target_col] = {
            'y_true': y_test.values,
            'y_pred': y_pred,
            'test_indices': y_test.index.tolist(),
            'r2': r2,
            'nmae': nmae,
            'nrmse': nrmse,
            'smape': smape
        }

    @staticmethod
    def _calculate_smape(y_true, y_pred):
        """
        Computes the symmetric mean absolute percentage error (sMAPE) [%].
        A small constant (1e-9) keeps the denominator non-zero.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-9
        return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denominator))

    def _save_model(self, target_col, model, num_cols, cat_cols, num_imputer, encoder):
        """Stores the model and its transformers."""
        self.models[target_col] = {
            'model': model,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'num_imputer': num_imputer,
            'encoder': encoder
        }

    def _save_feature_importance(self, target_col: str):
        """Stores the feature importance of the final model (sorted in descending order)."""
        model_info = self.models[target_col]
        feature_names = (model_info['num_cols'] or []) + (model_info['cat_cols'] or [])
        importances = model_info['model'].feature_importances_

        k = min(len(feature_names), len(importances))
        importance_dict = {
            feature_names[i]: float(importances[i])
            for i in range(k)
        }

        self.feature_importance[target_col] = dict(
            sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        )
