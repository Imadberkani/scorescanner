"""
The test_preprocessing.py file contains integration tests for various classes within the preprocessing module. 
These tests ensure that the classes function correctly when integrated with typical use cases and data flows.

Testoutlierdetector:
    This class conducts integration tests specifically for the `outlierdetector` class. 
    It checks the functionality of detecting and replacing outliers under various scenarios to ensure that the class 
    accurately identifies and handles outliers according to the defined methods and parameters.

Testmultioptbinning:
    A class dedicated to conducting integration tests for the `multioptBinning` class. 
    It verifies that the binning process functions correctly for different target variable scenarios as well as for various
    scenarios of the strength of the relationship between the explanatory variables and the target.


"""

# Importing librairies
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
import pytest 
import pandas as pd
import numpy as np
from scorescanner.preprocessing import (
    outlierdetector,
    multioptbinning,
    refcatencoder,
)


class TestOutlierDetector:
    @pytest.fixture
    def setup_dataframe(self):
        """Create a sample dataframe to use in tests."""
        data = {
            'A': [1, 2, 3, 4, 4, 4, 1, 3, 5, 3, 100],  
            'B': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],   
            'C': [1, 2, 2, 3, 5, 1, 2, 2, 3, 5, 50],   
            'D': [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 10600]  
        }
        return pd.DataFrame(data)
    
    @pytest.mark.parametrize("method", ["IQR", "z-score"])
    def test_iqr_constant_replacement(self, setup_dataframe, method):
        """Test outlier detection with IQR and z-score methods with constant value replacement"""
        df = setup_dataframe
        detector = outlierdetector(features=['A', 'B', 'C'], method=method, replacement_method='constant', replacement_value=-999.001)
        transformed_df = detector.fit_transform(df)

        # Check that the last value in column 'A' (expected to be an outlier) has been replaced with -999.001
        assert transformed_df.iloc[-1]['A'] == -999.001, "Outlier at the last index of column A should be replaced with -999.001"

        # Check that the last value in column 'C' (expected to be a mild outlier) has been replaced with -999.001
        assert transformed_df.iloc[-1]['C'] == -999.001, "Outlier at the last index of column C should be replaced with -999.001"

        # Check that no non-outlier values have been mistakenly replaced
        assert all(transformed_df['B'] == df['B']), "No changes should be made to column B since there are no outliers"

    @pytest.mark.parametrize("method", ["IQR", "z-score"])
    def test_iqr_constant_dict_replacement(self, setup_dataframe, method):
        """Test outlier detection using both IQR and z-score methods with dictionary-based constant value replacement"""
        df = setup_dataframe
        replacement_methods = {'A': 'constant', 'C': 'constant', 'D': 'mean'}  
        replacement_values = {'A': -999.001, 'C': -999.001, 'D': None}  

        detector = outlierdetector(
            features=['A', 'B', 'C', 'D'],
            method=method,
            replacement_method=replacement_methods,
            replacement_value=replacement_values
        )
        transformed_df = detector.fit_transform(df)

        # Check that the last value in column 'A' has been replaced with -999.001
        assert transformed_df.iloc[-1]['A'] == -999.001, "Outlier at the last index of column A should be replaced with -999.001"

        # Check that the last value in column 'C' has been replaced with -999.001
        assert transformed_df.iloc[-1]['C'] == -999.001, "Outlier at the last index of column C should be replaced with -999.001"

        # Check that column 'D' uses the mean of the non-outlier values for replacement
        expected_mean_D = df['D'][:-1].mean()  # Calculate mean excluding the potential outlier
        assert transformed_df.iloc[-1]['D'] == pytest.approx(expected_mean_D), "Outlier in column D should be replaced with the mean of non-outliers"

        # Check that no changes were made to column 'B' since there are no outliers
        assert all(transformed_df['B'] == df['B']), "No changes should be made to column B since there are no outliers"    


class Testmultioptbinning:
    @pytest.fixture
    def setup_data(self):
        """Generate a single sample dataframe for all classification types."""
        np.random.seed(0)
        data = {
            'numerical1': np.random.normal(loc=10, scale=3, size=100),
            'numerical2': np.random.uniform(low=0, high=1, size=100),
            'binary_target': np.random.choice([0, 1], size=100),
            'continuous_target': np.random.normal(loc=50, scale=10, size=100),
            'multiclass_target': np.random.choice([0, 1, 2], size=100),  
            'categorical1': np.random.choice(['A', 'B', 'C'], size=100)  
        }
        return pd.DataFrame(data)

    @pytest.mark.parametrize("target_column, target_dtype, num_features, cat_features", [
        ('binary_target', 'binary', ['numerical1', 'numerical2'], ['categorical1']),
        ('continuous_target', 'continuous', ['numerical1', 'numerical2'], ['categorical1']),
        ('multiclass_target', 'multiclass', ['numerical1', 'numerical2'], [])
    ])
    def test_multioptbinning(self, setup_data, target_column, target_dtype, num_features, cat_features):
        """Test multioptbinning across different target scenarios."""
        df = setup_data
        target = target_column
       

        binning = multioptbinning(
            num_features=num_features,
            cat_features=cat_features,
            target=target,
            target_dtype=target_dtype,
        )
        binning.fit(df)
        transformed_df = binning.transform(df)

       
        assert not transformed_df.isnull().values.any(), "No nulls should be in the transformed dataframe"
        assert set(transformed_df.columns).issuperset(num_features + cat_features), "All original features should be present in the transformed dataframe"
        for num_feature in num_features:
            assert transformed_df[num_feature].nunique() >= 2, f"{num_feature} should have at least 2 unique values post-binning"
            

           
