"""
preprocessing._data.py contain a set of classes to perform some preprocessing tasks.

Classes:

outlierdetector : a class for detecting and replacing outliers.
multioptbinning : a class for optimal binning.
logisticregressionpreparer : a class for preparing data for logistic regression.

"""

#Importing librairies
from optbinning import OptimalBinning, ContinuousOptimalBinning, MulticlassOptimalBinning
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class outlierdetector:
    """
    A class designed for detecting and replacing outliers in a dataset. It supports two methods for outlier detection: 
    the Interquartile Range (IQR) and the Z-score method. Additionally, it supports multiple methods for replacing outliers:
    constant value, mean, median,... 
    """
    def __init__(self, columns, method='IQR', replacement_method='constant', replacement_value=-999.001, z_threshold=3):
        """
        Initialization of the outlierdetector class.

        Attributes:
        columns (list of str): List of columns to be processed for outlier detection.
        method (str, optional (default='IQR')): Method for detecting outliers ('IQR' or 'z-score').
        replacement_method (str or dict, optional (default='constant')): Method for replacing outliers ('constant', 'mean', 
                                        'median', 'mode', 'std_dev' and 'capping_flooring'). If dict, specify method per column.
        replacement_value (float or dict, optional): Value to replace outliers if replacement method is 'constant'. 
                                        Default is -999.001. Ignored for other methods.
        z_threshold (float, optional (default=3)): Z-score threshold for outlier detection.
        bounds (dict): Dictionary to store lower and upper bounds for IQR method for each column.
        stats (dict): Dictionary to store mean and standard deviation for z-score method for each column.
        """
        # Initialization of class attributes
        self.columns = columns
        self.method = method
        self.replacement_method = replacement_method
        self.replacement_value = replacement_value
        self.z_threshold = z_threshold
        self.bounds = {column: {'lower': None, 'upper': None} for column in columns}
        self.stats = {column: {'mean': None, 'std_dev': None} for column in columns}  

    def fit(self, df, y=None):
        """
        Fits the outlier detection method to the given DataFrame, determining threshold values for outliers in each specified column.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit on.
        """
        for column in self.columns:
            if self.method == 'IQR':
                # Calculate the first quartile (25th percentile)
                Q1 = df[column].quantile(0.25)
                # Calculate the third quartile (75th percentile)
                Q3 = df[column].quantile(0.75)
                # Calculate the Interquartile Range (IQR)
                IQR = Q3 - Q1
                # Define the lower bound for outlier detection
                self.bounds[column]['lower'] = Q1 - 1.5 * IQR
                # Define the upper bound for outlier detection
                self.bounds[column]['upper'] = Q3 + 1.5 * IQR

            elif self.method == 'z-score':
                # Calculate the mean of the column
                self.stats[column]['mean'] = df[column].mean()
                # Calculate the standard deviation of the column
                self.stats[column]['std_dev'] = df[column].std()

    def _is_outlier(self, value, column):
        """
        Checks if a value is an outlier based on the defined method.

        Parameters:
        value (float): The value in the column.
        column (str): The name of the column.

        Returns:
        bool: True if the value is an outlier, False otherwise.
        """
        if self.method == 'IQR':
            # Lower bound for IQR method
            lower = self.bounds[column]['lower']
            # upper bound for IQR method
            upper = self.bounds[column]['upper']
            return value < lower or value > upper
        elif self.method == 'z-score':
            # mean for z-score method
            mean = self.stats[column]['mean']
            # standard deviation for z-score methodgg
            std_dev = self.stats[column]['std_dev']
            return abs(value - mean) / std_dev > self.z_threshold
        else:
            return False            

    def _replace_outlier(self, value, column, df):
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
                replacement_method = self.replacement_method.get(column, 'constant')
            else:
                replacement_method = self.replacement_method

            # Applying the specified replacement method
            if replacement_method == 'mean':
                # Replacing with the column's mean
                return df[column].mean()
            elif replacement_method == 'median':
                # Replacing with the column's median
                return df[column].median()
            elif replacement_method == 'mode':
                # Replacing with the column's mode
                return df[column].mode()[0]
            elif replacement_method == 'std_dev':
                # Replacing with the column's standard deviation
                return df[column].std()
            elif replacement_method == 'capping_flooring':
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
                return self.replacement_value
        # If the value is not an outlier, return it unchanged
        return value


    def transform(self, df):
        """
        Applies the calculated outlier thresholds to the given DataFrame for each column and replaces outliers using apply method.

        Parameters:
        df (pandas.DataFrame): The DataFrame on which to apply the outlier thresholds.

        Returns:
        pandas.DataFrame: The DataFrame with outliers replaced in the specified columns.
        """
        # Creating a copy of the input DataFrame
        df_copy = df.copy()
        for column in tqdm(self.columns, desc="Replacing outliers"):
            filtered_df = df_copy[df_copy[column].apply(lambda x: not self._is_outlier(x, column))]
            #replacing outliers with "outlier_value'
            df_copy[column] = df_copy[column].apply(lambda value: self._replace_outlier(value, column, filtered_df))
        
        return df_copy

    def fit_transform(self, df, y=None):
        """
        Fits outilerdetection method and transforms the specified variables in one step.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit and transform.

        Returns:
        pandas.DataFrame: The DataFrame with the specified variables transformed.
        """
        self.fit(df)
        return self.transform(df)




class multioptbinning:
    """
    A class for transforming a list of continuous variables into categorical variables using optimal binning.
    """
    def __init__(self, variables, target, target_dtype, dtype="numerical", solver = "cp", prebinning_method="cart", divergence="iv", min_n_bins=2, monotonic_trend= "auto", min_event_rate_diff=0.02, outlier_value=-999.001,  additional_optb_params=None):
        """
        Initializes the multioptbinning class.

        Parameters:
        variables (list of str): A list of column names to process for optimal binning.
        target ( str): Name of target column.
        target_dtype (str, optional (default="binary")) - The data type of the target variable. Supported types are "binary", "continuous", and "multiclass".
        dtype (str, optional (default="numerical")) – The variable data type. Supported data types are “numerical” for continuous and ordinal
                                                      variables and “categorical” for categorical and nominal variables.

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

        outlier_value (float, optional (default=-999.001)): The value of outliers.
        """
        # Initialization of class attributes
        self.variables = variables
        self.target = target
        self.target_dtype = target_dtype
        self.dtype = dtype 
        self.solver = solver
        self.prebinning_method = prebinning_method
        self.divergence = divergence
        self.min_n_bins = min_n_bins
        self.monotonic_trend = monotonic_trend
        self.min_event_rate_diff = min_event_rate_diff
        self.additional_optb_params = additional_optb_params or {}
        self.outlier_value = outlier_value
        self.optb_models = {}

    def fit(self, df, y=None):
        """
        Fits OptimalBinning models on the specified variables of the provided DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit on.
        """
        for variable in tqdm(self.variables, desc="Fitting OptimalBinning Models"):
            # Creating a dictionary of parameters for initializing the OptimalBinning object
            optb_params = {
                "name": variable,
                "dtype": self.dtype,
                "solver": self.solver,
                "prebinning_method" : self.prebinning_method,
                "divergence" : self.divergence,
                "min_n_bins" : self.min_n_bins,
                "monotonic_trend" : self.monotonic_trend,
                "min_event_rate_diff" : self.min_event_rate_diff,
                "special_codes": [self.outlier_value],
                **self.additional_optb_params
            }
            # Creating an instance of OptimalBinning for binary targets
            if self.target_dtype == "binary":
                optb = OptimalBinning(**optb_params)
            # Creating an instance of OptimalBinning for continuous targets
            elif self.target_dtype == "continuous":
                optb = ContinuousOptimalBinning(**{key: value for key, value in optb_params.items() if key not in  ["solver","divergence","min_event_rate_diff"]})
            # Creating an instance of OptimalBinning for multiclass targets
            elif self.target_dtype == "multiclass":
                optb = MulticlassOptimalBinning(**{key: value for key, value in optb_params.items() if key not in  ["dtype","divergence"]})
            else:
                raise ValueError("Unsupported target_dtype: {}".format(self.target_dtype))
            # Fiting the OptimalBinning model
            optb.fit(df[variable].values, df[self.target].values)
            # Storing the fitted OptimalBinning model in a dictionary
            self.optb_models[variable] = optb

    def transform(self, df):
        """
        Transforms the specified variables of the given DataFrame using the fitted OptimalBinning models.

        Parameters:
        df (pandas.DataFrame): The DataFrame on which to apply the transformations.

        Returns:
        pandas.DataFrame: The DataFrame with the specified variables transformed.
        """
        # Creating a copy of the initial dataframe
        transformed_df = df.copy()
        # Identify binary columns
        binary_columns = [col for col in df.columns if len(df[col].dropna().unique()) == 2]
        # Transforming each variable
        for variable in [v for v in self.variables if v not in binary_columns]:
            if variable in self.optb_models:
                # Check the number of bins created for the variable
                n_bins = len(self.optb_models[variable].splits) + 1
                if n_bins > 1:
                    # using the standard transformation if more than one bin
                    transformed_df[variable] = self.optb_models[variable].transform(transformed_df[variable].values, metric="bins")
                else:
                    # If only one bin, manually split the variable into two uniform intervals using quantiles
                    quantiles = np.nanpercentile(transformed_df[variable], np.linspace(0, 100, 5))
                    transformed_df[variable] = pd.cut(transformed_df[variable], bins=quantiles, include_lowest=True, duplicates='drop')
            else:
                raise ValueError(f"Model not fitted for variable: {variable}")
        # Returns the DataFrame with the transformed variables.
        return transformed_df 

    def fit_transform(self, df, y=None):
        """
        Fits OptimalBinning models and transforms the specified variables in one step.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to fit and transform.

        Returns:
        pandas.DataFrame: The DataFrame with the specified variables transformed.
        """
        self.fit(df)
        # Returns the DataFrame with the transformed variables.
        return self.transform(df)


class logisticregressionpreparer:
    """
    A class designed to prepare data for logistic regression. It applies one-hot encoding to specified columns
    of a DataFrame, with an option to selectively drop categories based on a provided dictionary. 
    """
    def __init__(self, columns, column_dict):
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

    def fit(self, df):
        """
        Fit the data preparer to the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to fit the preparer on.
        """
        for col in self.columns:
            # category to drop for each column
            categories_to_drop = [self.column_dict[col]] if col in self.column_dict else [df[col].dropna().unique()[0]]
            # Initializing OneHotEncoder object with column ang category to drop 
            self.encoders[col] = OneHotEncoder(categories='auto', drop=categories_to_drop)
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
        df_transformed = df.copy()
        for col, encoder in self.encoders.items():
            # transforming the column
            encoded_data = encoder.transform(df[[col]]).toarray()
            # Determining reference category
            reference_category = self.column_dict[col] if col in self.column_dict else df[col].dropna().unique()[0]
            # new column names for the encoded data
            col_names = [f"{col}_{cat} (vs {reference_category})" for cat in encoder.categories_[0] if cat != reference_category]
            # New DataFrame with encoded column
            df_encoded = pd.DataFrame(encoded_data, columns=col_names, index=df.index)
            # Dropping the original column
            df_transformed = df_transformed.drop(col, axis=1).join(df_encoded)

        return df_transformed

    def fit_transform(self, df):
        """
        Fit and transform the DataFrame in one step.

        Parameters:
        df (pd.DataFrame): The DataFrame to fit and transform.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df)
    

    