"""
preprocessing._data.py contains a set of classes designed to perform essential preprocessing
tasks that prepare the initial dataframe for detailed analysis and modeling.

Classes:

- outlierdetector: A class designed to process a DataFrame by detecting and replacing `outliers`
                    in the numerical features specified by the `features` parameter.
- multioptbinning: A class for transforming continuous features into categorical ones using optimal
                   binning strategies across various feature types and target configurations. For Features
                   where no significant relationship with the target is detected, an unsupervised clustering
                   algorithm, `HDBSCAN`, is used.
- logisticregressionpreparer: A class for preparing data for logistic regression modeling.

"""

# Importing librairies
import pandas as pd
from optbinning import (
    ContinuousOptimalBinning,
    MulticlassOptimalBinning,
    OptimalBinning,
)
from hdbscan import HDBSCAN
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm
from typing import List, Union, Dict, Optional


class outlierdetector:
    """
    A class designed to process a DataFrame by detecting and replacing `outliers` in the numerical features specified
    by the `columns` parameter. It supports two methods for outlier detection: the Interquartile Range (IQR) and the Z-score method.
    Additionally, it supports multiple methods for replacing outliers: constant value, mean, median,...
    """

    def __init__(
        self,
        features: List[str],
        method: str = "IQR",
        replacement_method: Union[str, Dict[str, str]] = "constant",
        replacement_value: Union[float, Dict[str, float]] = -999.001,
        z_threshold: float = 3,
    ) -> None:
        """
        Initialization of the outlierdetector class.

        Attributes:
        features (list of str): List of features to be processed for outlier detection.
        method (str, optional (default='IQR')): Method for detecting outliers ('IQR' or 'z-score').
        replacement_method (str or dict, optional (default='constant')): Method for replacing outliers ('constant', 'mean',
                                        'median', 'mode', 'std_dev' and 'capping_flooring'). If dict, specify method per column.
        replacement_value (float or dict, optional): Value to replace outliers if replacement method is 'constant'.
                                        Default is -999.001. If dict, specify value per column. Ignored for other methods.
        z_threshold (float, optional (default=3)): Z-score threshold for outlier detection.
        bounds (dict): Dictionary to store lower and upper bounds for IQR method for each column.
        stats (dict): Dictionary to store mean and standard deviation for z-score method for each column.
        """
        # Initialization of class attributes
        self.features = features
        self.method = method
        self.replacement_method = replacement_method
        self.replacement_value = replacement_value
        self.z_threshold = z_threshold
        self.bounds: Dict[str, Dict[str, Optional[float]]] = {column: {"lower": None, "upper": None} for column in features}
        self.stats: Dict[str, Dict[str, Optional[float]]] = {column: {"mean": None, "std_dev": None} for column in features}

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Fits the outlier detection method to the given DataFrame based on the defined method,
        determining threshold values for outliers in each specified column.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit on.
        """
        for column in self.features:
            # Case of `Interquartile Range` method
            if self.method == "IQR":
                # Calculate the first quartile (25th percentile)
                Q1 = df[column].quantile(0.25)
                # Calculate the third quartile (75th percentile)
                Q3 = df[column].quantile(0.75)
                # Calculate the Interquartile Range (IQR)
                IQR = Q3 - Q1
                # Define the lower bound for outlier detection
                self.bounds[column]["lower"] = Q1 - 1.5 * IQR
                # Define the upper bound for outlier detection
                self.bounds[column]["upper"] = Q3 + 1.5 * IQR
            # Case of `z-score` method
            elif self.method == "z-score":
                # Calculate the mean of the column
                self.stats[column]["mean"] = df[column].mean()
                # Calculate the standard deviation of the column
                self.stats[column]["std_dev"] = df[column].std()

    def _is_outlier(self, value: float, column: str) -> bool:
        """
        Checks if a value is an outlier based on the defined method.

        Parameters:
        value (float): The value in the column.
        column (str): The name of the column.

        Returns:
        bool: True if the value is an outlier, False otherwise.
        """
        # Case of `Interquartile Range` method
        if self.method == "IQR":
            # Lower bound for IQR method
            lower = self.bounds[column]["lower"]
            # upper bound for IQR method
            upper = self.bounds[column]["upper"]
            return value < lower or value > upper
        
        # Case of `z-score` method
        elif self.method == "z-score":
            # mean for z-score method
            mean = self.stats[column]["mean"]
            # standard deviation for z-score method
            std_dev = self.stats[column]["std_dev"]
            return abs(value - mean) / std_dev > self.z_threshold
        else:
            return False

    def _replace_outlier(self, value: float, column: str, df: pd.DataFrame) -> float:
        """
        Function to replace outliers in a column based on the calculated bounds or statistics.

        Parameters:
        value (float): The value in the column.
        column (str): The name of the column.
        df (pandas.DataFrame): The DataFrame for reference in certain replacement methods.

        Returns:
        float: The replaced value.
        """
        # Checking if the value is an outlier
        if self._is_outlier(value, column):
            # Selecting the replacement method based on the column if specified as a dictionary, otherwise using the global method
            if isinstance(self.replacement_method, dict):
                # If it is a dictionary, retrieve the replacement method for the current column.
                # Default to 'constant' if the column is not specified in the dictionary.
                replacement_method = self.replacement_method.get(column, "constant")
            else:
                # If replacement_method is not a dictionary, we will use the same replacement method for all features.
                replacement_method = self.replacement_method

            # Applying the specified replacement method
            if replacement_method == "mean":
                # Replacing with the column's mean
                return df[column].mean()
            elif replacement_method == "median":
                # Replacing with the column's median
                return df[column].median()
            elif replacement_method == "mode":
                # Replacing with the column's mode
                return df[column].mode()[0]
            elif replacement_method == "std_dev":
                # Replacing with the column's standard deviation
                return df[column].std()
            elif replacement_method == "capping_flooring":
                # Defining the upper and lower bounds for capping and flooring
                upper_bound = df[column].quantile(0.95)
                lower_bound = df[column].quantile(0.05)

                # Applying capping and flooring
                # First, caping the value if it's above the upper bound
                value = min(value, upper_bound)
                # Then, flooring the value if it's below the lower bound
                value = max(value, lower_bound)
                return value
            else:
                # If none of the above methods are specified, using a constant value for replacement
                ## Check if replacement_value is a dictionary and fetch the value for the current column if it is
                if isinstance(self.replacement_value, dict):
                    return self.replacement_value.get(column, -999.001)  
                else:
                    # If replacement_value is not a dictionary, we will use the same constant value for all features.
                    return self.replacement_value
        # If the value is not an outlier, return it unchanged
        return value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the calculated outlier thresholds to the given DataFrame for each column and replaces outliers using apply method.

        Parameters:
        df (pandas.DataFrame): The DataFrame on which to apply the outlier thresholds.

        Returns:
        pandas.DataFrame: The DataFrame with outliers replaced in the specified features.
        """
        # Creating a copy of the input DataFrame
        df_copy = df.copy()
        for column in self.features:
            # Create a filtered DataFrame where the values that are not outliers are retained.
            # This step isolates the non-outlier data, which can be used to estimate replacement values for the outliers.
            filtered_df = df_copy[
                df_copy[column].apply(lambda x: not self._is_outlier(x, column))
            ]
            # replacing outliers with "special_value'
            # Here, the method "_replace_outlier" uses the filtered non-outlier data for methods that require estimation
            df_copy[column] = df_copy[column].apply(
                lambda value: self._replace_outlier(value, column, filtered_df)
            )

        return df_copy

    def fit_transform(self, df: pd.DataFrame, y = None) -> pd.DataFrame:
        """
        Fits outilerdetection method and transforms the specified features in one step.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit and transform.

        Returns:
        pandas.DataFrame: The DataFrame with the specified features transformed.
        """
        # Fitting the outlier detection methods to the DataFrame
        self.fit(df)
        # Transforming the DataFrame by replacing any identified outliers according to the specified replacement method 
        return self.transform(df)


class multioptbinning:
    """
    A class for transforming continuous features into categorical ones using optimal binning strategies
    across various feature types and target configurations. For Features where no significant relationship
    with the target is detected, an unsupervised clustering algorithm, HDBSCAN, is used.
    """

    def __init__(
        self,
        num_features: list,
        cat_features: list,
        target: str,
        target_dtype: str = "binary",
        solver: str = "cp",
        prebinning_method: str = "cart",
        divergence: str = "iv",
        min_n_bins: int = 2,
        monotonic_trend: str = "auto",
        min_event_rate_diff: float = 0.02,
        special_value: float = -999.001,
        additional_optb_params: dict = None,
        hdbscan_params: dict = None,
    ):
        """
        Initializes the multioptbinning class.

        Parameters:
        num_features (list of str): A list of numerical features names to process for optimal binning.
        cat_features (list of str): A list of categorical features names to process for optimal binning.
        target ( str): Name of target column.
        target_dtype (str, optional (default="binary")) - The data type of the target variable. Supported types are "binary", "continuous", and "multiclass".
        
        solver (str, optional (default="cp")) – The optimizer to solve the optimal binning problem. Supported solvers are “mip” to choose a
                                                mixed-integer programming solver, “cp” to choose a constrained programming solver or “ls” to choose LocalSolver.

        prebinning_method (str, optional (default="cart")) – The pre-binning method. Supported methods are “cart” for a CART decision tree,
                                                             “mdlp” for Minimum Description Length Principle (MDLP), “quantile” to generate
                                                              prebins with approximately same frequency and “uniform” to generate prebins with
                                                               equal width. Method “cart” uses

        divergence (str, optional (default="iv")) –    The divergence measure in the objective function to be maximized. Supported divergences
                                                        are “iv” (Information Value or Jeffrey’s divergence), “js” (Jensen-Shannon),
                                                        “hellinger” (Hellinger divergence) and “triangular” (triangular discrimination).

        min_n_bins (int or None, optional (default=2)) – The minimum number of bins. If None, then min_n_bins is a value in [0, max_n_prebins].

        monotonic_trend (str or None, optional (default="auto")) – The event rate monotonic trend. Supported trends are “auto”, “auto_heuristic”
                                                                    and “auto_asc_desc” to automatically determine the trend maximizing IV using a
                                                                    machine learning classifier, “ascending”, “descending”, “concave”, “convex”,
                                                                    “peak” and “peak_heuristic” to allow a peak change point, and “valley” and “valley_heuristic”
                                                                    to allow a valley change point. Trends “auto_heuristic”, “peak_heuristic” and “valley_heuristic”
                                                                    use a heuristic to determine the change point, and are significantly faster for large size instances (max_n_prebins > 20).
                                                                    Trend “auto_asc_desc” is used to automatically select the best monotonic trend between “ascending” and “descending”. If None,
                                                                     then the monotonic constraint is disabled.

        min_event_rate_diff (float, optional (default=0)) – The minimum event rate difference between consecutives bins. For solver “ls”, this option currently only applies when monotonic_trend
                                                             is “ascending”, “descending”, “peak_heuristic” or “valley_heuristic”.

        additional_optb_params (dict, optional): Additional parameters for OptimalBinning. For a full list of parameters, see the OptimalBinning documentation at [https://gnpalencia.org/optbinning/].

        hdbscan_params (dict, optional): parameters for hdbscan clustering algorithm. For a full list of parameters, see the HDBSCAN documentation at [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html]).

        special_value (float, optional (default=-999.001)): A special value designated for outliers, for which a distinct category will be created.
        """
        # Initialization of class attributes
        self.num_features = num_features
        self.cat_features = cat_features
        self.target = target
        self.target_dtype = target_dtype
        self.solver = solver
        self.prebinning_method = prebinning_method
        self.divergence = divergence
        self.min_n_bins = min_n_bins
        self.monotonic_trend = monotonic_trend
        self.min_event_rate_diff = min_event_rate_diff
        self.additional_optb_params = additional_optb_params or {}
        self.hdbscan_params = hdbscan_params or {"min_samples": 2}
        self.special_value = special_value
        self.optb_models = {}
        self.features= self.num_features + self.cat_features

    def fit(self, df: pd.DataFrame, y=None) -> None:
        """
        Fits OptimalBinning models on the specified features of the provided DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit on.
        """
        # Creating a LabelEncoder object for the target
        le = LabelEncoder()
        # Encoding the target
        encoded_target = le.fit_transform(df[self.target].values)

        # Seting default min_cluster_size if not provided in hdbscan_params
        default_min_cluster_size = max(int(0.05 * len(df)), 2)  
        self.hdbscan_params.setdefault('min_cluster_size', default_min_cluster_size)
        
    
        for variable in tqdm(self.features, desc="Fitting OptimalBinning Models"):
            if variable in self.num_features:
                dtype = "numerical"
            elif variable in self.cat_features:
                dtype = "categorical"
            else:
                raise ValueError(f"Variable '{variable}' not found in num_features or cat_features lists")

            # Creating a dictionary of parameters for initializing the OptimalBinning object
            optb_params = {
                "name": variable,
                "dtype": dtype,
                "solver": self.solver,
                "prebinning_method": self.prebinning_method,
                "divergence": self.divergence,
                "min_n_bins": self.min_n_bins,
                "monotonic_trend": self.monotonic_trend,
                "min_event_rate_diff": self.min_event_rate_diff,
                "special_codes": [self.special_value],
                **self.additional_optb_params,
            }
            # Creating an instance of OptimalBinning for binary targets
            if self.target_dtype == "binary":
                optb = OptimalBinning(**optb_params)
            # Creating an instance of OptimalBinning for continuous targets
            elif self.target_dtype == "continuous":
                optb = ContinuousOptimalBinning(
                    **{
                        key: value
                        for key, value in optb_params.items()
                        if key not in ["solver", "divergence", "min_event_rate_diff"]
                    }
                )
            # Creating an instance of OptimalBinning for multiclass targets
            elif self.target_dtype == "multiclass":
                optb = MulticlassOptimalBinning(
                    **{
                        key: value
                        for key, value in optb_params.items()
                        if key not in ["dtype", "divergence"]
                    }
                )
            else:
                raise ValueError(
                    "Unsupported target_dtype: {}".format(self.target_dtype)
                )
            
            # Fiting the OptimalBinning model
            optb.fit(df[variable].values, encoded_target)
            # Check if OptimalBinning found at least one split (meaning at least two bins)
            if len(optb.splits) >= 1:
                self.optb_models[variable] = ('optimalbinning', optb)
            else:
                if dtype == "numerical":
                    # Apply HDBSCAN for numerical variables if fewer than two bins are found
                    hdbscan_model = HDBSCAN(**self.hdbscan_params).fit(df[variable].values.reshape(-1, 1))
                    self.optb_models[variable] = ('HDBSCAN', hdbscan_model)
                    print(f"HDBSCAN was applied to the numerical variable '{variable}' due to the lack of significant splits.")
                elif dtype == "categorical":
                    # For categorical variables with fewer than two bins, record that no fitting was applied
                    self.optb_models[variable] = ('no_binning_applied', None)
                    print(f"No significant splits were found for the categorical variable '{variable}'; no binning was applied.")

    

    def transform(self, df):
        """
        Transforms the specified features of the given DataFrame using the fitted OptimalBinning models.

        Parameters:
        df (pandas.DataFrame): The DataFrame on which to apply the transformations.

        Returns:
        pandas.DataFrame: The DataFrame with the specified features transformed.
        """
        # Creating a copy of the initial dataframe
        transformed_df = df.copy()
        # Identify binary features
        binary_features = [
            col for col in df.columns if len(df[col].dropna().unique()) == 2
        ]
        transformed_df[binary_features] = transformed_df[binary_features].applymap(
            lambda x: "Special" if x == self.special_value else x
        )

        # Transforming each variable
        for variable in [v for v in self.features if v not in binary_features]:
            if variable in self.optb_models:
                model_type, model = self.optb_models[variable]

                # Check the type of the model and perform corresponding actions
                if model_type == 'optimalbinning':
                        # Using the standard transformation if more than one bin
                        transformed_df[variable] = model.transform(
                            transformed_df[variable].values, metric="bins"
                        )
                elif model_type == 'HDBSCAN':
                    # Applying HDBSCAN labels
                    # Creating mask for missing values in the dataset
                    nan_mask = transformed_df[variable].isna()
                    # Predicting labels using the HDBSCAN model
                    labels = model.labels_
                    # Customizing labels
                    # Grouping data by the cluster labels
                    unique_values = df[variable].groupby(labels).apply(lambda x: [-1] if x.name == -1 else x.unique())
                    # Characterizing clusters with intervals for varied values or unique values for constant clusters
                    custom_labels = unique_values.apply(lambda x: 'outliers' if -1 in x else f"[{x.min()},{x.max()}]" if x.min() != x.max() else f"{x.max()}")
                   
                    # Maping customized labels to the transformed data
                    label_map = {idx: label for idx, label in zip(unique_values.index, custom_labels)}
                    transformed_df[variable] = [label_map[label] for label in labels]
                    
                    # Replacing nan values with 'missing'
                    transformed_df.loc[nan_mask, variable] = 'missing'
                elif model_type == 'no_binning_applied':
                    # Keeping original values for categorical variables where no binning was applied
                    transformed_df[variable] = df[variable]    
            else:
                raise ValueError(f"Model not fitted for variable: {variable}")
        # Returns the DataFrame with the transformed features.
        return transformed_df

    def fit_transform(self, df, y=None):
        """
        Fits OptimalBinning models and transforms the specified features in one step.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit and transform.

        Returns:
        pandas.DataFrame: The DataFrame with the specified features transformed.
        """
        # Fitting the optbinning methods to the DataFrame
        self.fit(df)
        # Returns the DataFrame with the transformed features.
        return self.transform(df)


class refcatencoder:
    """
    A class designed to prepare data for logistic regression. It applies one-hot encoding to specified columns
    of a DataFrame, with an option to selectively drop categories based on a provided dictionary.
    """

    def __init__(self, columns=None, column_dict={}, target=None):
        """
        Initialize the logisticregressionpreparer with a list of columns and a column dictionary.

        Parameters:
        columns (list, optional): List of column names to be processed.
        column_dict (dict): A dictionary where keys are column names and values are the categories to drop.
                            If a column is not in this dictionary, the first category (based on nunique()) will be dropped.
        """
        # Initialization of class attributes
        self.columns = columns 
        self.column_dict = column_dict
        self.encoders = {}
        self.target = target

    def fit(self, df, y=None):
        """
        Fit the data preparer to the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to fit the preparer on.
        """
        # If columns is None, all columns in the DataFrame are used
        if self.columns is None:
            self.columns = [col for col in df.columns.tolist() if col not in self.target]

        for col in self.columns:
            # category to drop for each column
            categories_to_drop = (
                [self.column_dict[col]]
                if col in self.column_dict
                else [df[col].dropna().unique()[0]]
            )
            # Initializing OneHotEncoder object with column ang category to drop
            self.encoders[col] = OneHotEncoder(
                categories="auto", drop=categories_to_drop
            )
            # Fitting the encoder to the column
            self.encoders[col].fit(df[[col]])

    def transform(self, df):
        """
        Transform the DataFrame using the fitted encoders.

        Parameters:
        df (pd.DataFrame): The DataFrame to transform.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        df_transformed = df[
            [col for col in df.columns if col not in self.columns]
        ].copy()
        for col, encoder in self.encoders.items():
            # transforming the column
            encoded_data = encoder.transform(df[[col]]).toarray()
            # Determining reference category
            reference_category = (
                self.column_dict[col]
                if col in self.column_dict
                else df[col].dropna().unique()[0]
            )
            # new column names for the encoded data
            col_names = [
                f"{col}_{cat} (vs {reference_category})"
                for cat in encoder.categories_[0]
                if cat != reference_category 
            ]
            # New DataFrame with encoded column
            df_encoded = pd.DataFrame(encoded_data, columns=col_names, index=df.index)
            # Dropping the original column
            df_transformed = pd.concat([df_encoded, df_transformed], axis=1)
        return df_transformed

    def fit_transform(self, df, y=None):
        """
        Fit and transform the DataFrame in one step.

        Parameters:
        df (pd.DataFrame): The DataFrame to fit and transform.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df)
