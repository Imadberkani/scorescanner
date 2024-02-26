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

    def cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate the Cramér's V coefficient between two pd.Series.

        Parameters:
        x, y (pd.Series): Two pandas Series.

        Returns:
        float: Calculated Cramér's V coefficient.

        Note: This version will be used exclusively for Cramér's V correlation matrix calculations.
        """

        # Create a confusion matrix from the two Series
        confusion_matrix = pd.crosstab(x, y)

        # Calculate the chi-squared statistic from the confusion matrix
        chi2 = stats.chi2_contingency(confusion_matrix)[0]

        # Calculate the total number of observations
        n = confusion_matrix.sum().sum()

        # Calculate Phi2 (the chi-squared ratio to n)
        phi2 = chi2 / n

        # Get the number of rows and columns of the confusion matrix
        r, k = confusion_matrix.shape

        # Calculate Phi2 corrected by subtracting a correction for matrix dimensions
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))

        # Calculate corrections for the number of rows and columns
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        # Calculate the Cramér's V coefficient
        cramers_v_value = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        return cramers_v_value

    def fit(self, df):
        """
        Fits the selector to the DataFrame, performing correlation checks or VIF checks to select variables for modeling.
        """
        correlation_function = {
            "pearson": lambda x, y: x.corr(y),
            "cramers_v": self.cramers_v,
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

        with open("eliminated_variables_info.json", "w") as file:
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
        return df[self.selected_variables]

    def fit_transform(self, df):
        """
        Fits variableselector method and transforms the DataFrame in one step.
        """
        self.fit(df)
        return self.transform(df)
