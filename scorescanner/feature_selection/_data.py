"""
feature_selection._data.py is used to perform variable selection tasks.

Functions:

select_uncorrelated_features : Selects variables for a model while avoiding multicollinearity.

"""

# importing librairies
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scorescanner.utils.statistical_metrics import cramers_v

class variableselector:
    """
    A class designed to select variables for modeling by evaluating their correlations and multicollinearity.
    It supports Pearson correlation and Cramér's V for correlation measurement and VIF for multicollinearity.
    """

    def __init__(
        self,
        target,
        corr_threshold=0.3,
        metric="cramers_v",
        use_vif=False,
        vif_threshold=1.5,
        json_dict_name = "eliminated_variables_info.json",
        sample_size=None
    ):
        """
        Initialization of the VariableSelector class.
        Attributes:
        - target (str): The name of the target variable in the DataFrame.
        - corr_threshold (float): The threshold for correlation above which variables are considered highly correlated.
        - metric (str): The metric used for calculating correlation ('pearson' for continuous variables or 'cramers_v' for categorical variables).
        - use_vif (bool): A flag indicating whether to use VIF for selecting variables.
        - vif_threshold (float): The VIF threshold above which variables are considered to have multicollinearity.
        - selected_variables (list): The list of selected variables after fitting the model.
        - eliminated_variables_info (dict): Information on variables eliminated based on correlation or VIF, including the reason for their elimination.
        """
        self.target = target
        self.corr_threshold = corr_threshold
        self.metric = metric
        self.use_vif = use_vif
        self.vif_threshold = vif_threshold
        self.selected_variables = []
        self.eliminated_variables_info = {}
        self.json_dict_name = json_dict_name
        self.sample_size = sample_size

    def fit(self, df, y=None):
        """
        Fits the selector to the DataFrame, performing correlation checks or VIF checks to select variables for modeling.
        """
        if self.sample_size is not None and self.sample_size < len(df):
            df = df.sample(n=self.sample_size)
            
        correlation_function = {
            "pearson": lambda x, y: x.corr(y),
            "cramers_v": cramers_v,
        }.get(self.metric, None)

        if correlation_function is None:
            raise ValueError("Invalid metric. Choose 'pearson' or 'cramers_v'.")

        # Creating a LabelEncoder object for the target
        le = LabelEncoder()
        df[self.target] = le.fit_transform(df[self.target])
        correlation_with_target = df.drop(columns=[self.target]).apply(
            lambda x: correlation_function(x, df[self.target])
        )
        self.selected_variables = []
        self.eliminated_variables_info = {}

        for var in correlation_with_target.abs().sort_values(ascending=False).index:
            self._process_variable(df, var, correlation_function)

        with open(self.json_dict_name, "w") as file:
            json.dump(self.eliminated_variables_info, file)

        return self

    def _process_variable(self, df, var, correlation_function):
        """
        Evaluate and select variables based on correlation or VIF.
        """
        if self.use_vif:
            # Directly add the first variable or if VIF check is not used
            if not self.selected_variables:
                self.selected_variables.append(var)
            else:
                # Prepare DataFrame for VIF calculation, including the current variable and previously selected variables
                temp_df = sm.add_constant(df[self.selected_variables + [var]])

                # Calculate the VIF for the current variable
                vif = variance_inflation_factor(
                    temp_df.values, temp_df.columns.get_loc(var)
                )

                # Append the variable to the selected list if its VIF is below the threshold; otherwise, record its VIF
                if vif < self.vif_threshold:
                    self.selected_variables.append(var)
                else:
                    self.eliminated_variables_info[var] = vif
        elif not self.use_vif:
            correlations = df[self.selected_variables].apply(
                lambda x: correlation_function(x, df[var])
            )
            if correlations.empty or correlations.abs().max() < self.corr_threshold:
                self.selected_variables.append(var)
                self.eliminated_variables_info[var] = []
            else:
                # Find the key variable which the current variable is most correlated with among the selected
                key_var = correlations.abs().idxmax()
                # Add the variable to the group of the key variable
                self.eliminated_variables_info.setdefault(key_var, []).append(var)

    def transform(self, df):
        """
        Returns a new DataFrame containing only the variables selected by the fit method.
        """
        return df[self.selected_variables + [self.target]]

    def fit_transform(self, df, y=None):
        """
        Fits variableselector method and transforms the DataFrame in one step.
        """
        self.fit(df)
        return self.transform(df)