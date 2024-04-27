"""
The statistical_metrics_tests.py file contains integration tests for the functions defined in the statistical_metrics.py module.
"""


# Importing librairies
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
import unittest
import pandas as pd
import numpy as np
from scorescanner.utils.statistical_metrics import (
    cramers_v,
    calculate_cramers_v_matrix,
    calculate_js_distances,
    cluster_corr_matrix,
    logistic_regression_summary,
    one_vs_rest_woe,
    univariate_category_importance,
    univariate_feature_importance,
)

class testcramers_v(unittest.TestCase):
    def setUp(self):
        """
        Initializing the data for testing the cramers_v function.
        """
        np.random.seed(0)  

        col1 = np.random.choice(['A', 'B', 'C'], size=100)
        col2 = np.random.choice(['X', 'Y'], size=100)

        data = {
            'col1': col1,
            'col2': col2
        }
        self.df = pd.DataFrame(data)

    def test_cramers_v_calculation(self):
        """
        Test that Cramér's V is calculated correctly.
        """
        result = cramers_v(self.df['col1'], self.df['col2'])
        # Ensure result is a float
        self.assertIsInstance(result, float)
        # Check for a valid Cramér's V value (0 <= V <= 1)
        self.assertTrue(0 <= result <= 1)

    def test_identical_columns(self):
        """
        Test that Cramér's V is approximately 1 for identical columns.
        """
        self.df['col3'] = self.df['col1']
        result = cramers_v(self.df['col1'], self.df['col3'])
        self.assertTrue(0.99 <= result <= 1)

class testunivariatefeatureimportance(unittest.TestCase):
    def setUp(self):
        """
        Initializing the data for testing univariate_feature_importance function.
        """
        data = {
            'col1': [25, 30, 35, 40, 45],
            'col2': [50, 600, 700, 8, 900],
            'col3': ['0', '1', '0', '0', '1']
        }
        self.df = pd.DataFrame(data)
        self.features = ['col1', 'col2']
        self.target_var = 'col3'

    def test_univariate_feature_importance_cramerv(self):
        """
        Test with Cramér's V method.
        """
        result = univariate_feature_importance(
            df=self.df, 
            features=self.features, 
            target_var=self.target_var, 
            method='cramers_v'
        )
        self.assertTrue(not result.empty)
        
    def test_univariate_feature_importance_ppscore(self):
        """
        Test with Predictive Power Score (PPS) method.
        """
        result = univariate_feature_importance(
            df=self.df, 
            features=self.features, 
            target_var=self.target_var, 
            method='ppscore'
        )
        self.assertTrue(not result.empty)

    def test_invalid_method(self):
        """
        Test with an invalid method.
        """
        with self.assertRaises(ValueError):
            univariate_feature_importance(
                df=self.df, 
                features=self.features, 
                target_var=self.target_var, 
                method='Pearson'
            )

class testcalculatejsdistances(unittest.TestCase):
    def setUp(self):
        """
        Initializing the data for testing calculate_js_distances function.
        """
        data = {
            'col1': ["A", "B", "B", "A", "A"],
            'col3': ['0', '1', '0', '0', '1']
        }
        self.df = pd.DataFrame(data)
        self.feature = 'col1'
        self.target_var = 'col3'
        
    def test_js_distances(self):
        """
        Test that Jensen-Shannon distances are calculated correctly.
        """
        result = calculate_js_distances(
            df=self.df, 
            feature=self.feature, 
            target_var=self.target_var
        )
        # Check that the result is not empty
        self.assertTrue(not result.empty)
        # Check that the result contains the expected columns
        self.assertIn('Category', result.columns)
        self.assertIn('Jensen-Shannon Distance', result.columns)
        

if __name__ == '__main__':
    unittest.main()