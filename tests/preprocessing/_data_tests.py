"""
The _data_tests.py file contains integration tests for the preprocessing module classes.

TestOutlierDetector:
    A class dedicated to conducting integration tests for the OutlierDetector class.

TestMultiOptBinning:
    A class dedicated to conducting integration tests for the multioptbinning class.    

TestLogisticRegressionPreparer:
    A class to conduct tests for the LogisticRegressionPreparer class.
        
"""

# Importing librairies
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
import unittest
import pandas as pd
import numpy as np
from scorescanner.preprocessing import (
    outlierdetector,
    multioptbinning,
    logisticregressionpreparer,
)


class testoutlierdetector(unittest.TestCase):

    def setUp(self):
        """
        Initializing the data for testing OutlierDetector class.
        """
        # Seting the random seed
        np.random.seed(0)
        # Creating a synthetic DataFrame with 2 columns where the last 5 values are outliers.
        data = {
            "col1": np.concatenate(
                [np.random.normal(0, 1, 95), np.random.normal(10, 5, 5)]
            ),
            "col2": np.concatenate(
                [np.random.normal(0, 1, 95), np.random.normal(20, 5, 5)]
            ),
        }
        self.df_test = pd.DataFrame(data)
        # Columns that will be used in the tests
        self.columns = ["col1", "col2"]

    def test_iqr_constant(self):
        """
        Test using IQR method for outlier detection and constant replacement.
        """
        # Initializing the OutlierDetector with IQR method and constant replacement
        detector = outlierdetector(
            columns=self.columns,
            method="IQR",
            replacement_method="constant",
            replacement_value=-999,
        )

        # Fitting outliers detection method (Calculating the bounds for outliers)
        detector.fit(self.df_test)

        # Identifying outliers in each column
        outliers_mask = {
            col: self.df_test[col].apply(lambda x: detector._is_outlier(x, col))
            for col in self.columns
        }

        # Replacing the outliers with constant value
        df_transformed = detector.fit_transform(self.df_test)

        for col in self.columns:

            for i in outliers_mask[col][outliers_mask[col]].index:

                # Verifying that outliers have been correctly transformed.
                self.assertEqual(
                    df_transformed.loc[i, col],
                    -999,
                    f"Outlier not replaced in column {col}",
                )

                # Verifying that outliers have been correctly identified.
                outlier_value = self.df_test.loc[i, col]
                lower_bound, upper_bound = (
                    detector.bounds[col]["lower"],
                    detector.bounds[col]["upper"],
                )
                self.assertTrue(
                    outlier_value < lower_bound or outlier_value > upper_bound,
                    f"Identified outlier {outlier_value} in column {col} is within bounds",
                )

    def test_zscore_constant(self):
        """
        Test using Z-score method for outlier detection and constant replacement.
        """
        # Initializing the OutlierDetector with Z-score method and constant replacement
        detector = outlierdetector(
            columns=self.columns,
            method="z-score",
            replacement_method="constant",
            replacement_value=-999,
            z_threshold=3,
        )

        # Fitting outlier detection method (Calculating the mean and standard deviation for Z-score)
        detector.fit(self.df_test)

        # Identifying outliers in each column based on the Z-score
        outliers_mask = {
            col: self.df_test[col].apply(lambda x: detector._is_outlier(x, col))
            for col in self.columns
        }

        # Replacing the outliers with a constant value
        df_transformed = detector.fit_transform(self.df_test)

        for col in self.columns:

            for i in outliers_mask[col][outliers_mask[col]].index:
                # Verifying that outliers have been correctly transformed.
                self.assertEqual(
                    df_transformed.loc[i, col],
                    -999,
                    f"Outlier not replaced in column {col}",
                )

                # Verifying that outliers have been correctly identified.
                mean, std_dev = (
                    detector.stats[col]["mean"],
                    detector.stats[col]["std_dev"],
                )
                z_score = abs((self.df_test.loc[i, col] - mean) / std_dev)
                self.assertTrue(
                    z_score > detector.z_threshold,
                    f"Identified outlier {self.df_test.loc[i, col]} in column {col} has Z-score {z_score} within threshold",
                )

    def test_iqr_mean(self):
        """
        Test using IQR method for outlier detection and mean replacement.

        """
        # Initializing the OutlierDetector with IQR method and mean replacement
        detector = outlierdetector(
            columns=self.columns, method="IQR", replacement_method="mean"
        )

        # Fitting outlier detection method (Calculating the bounds for outliers)
        detector.fit(self.df_test)

        # Identifying outliers in each column
        outliers_mask = {
            col: self.df_test[col].apply(lambda x: detector._is_outlier(x, col))
            for col in self.columns
        }

        # Replacing the outliers with the mean of non-outlier values
        df_transformed = detector.fit_transform(self.df_test)

        for col in self.columns:

            # Calculate the mean of non-outlier values for verification
            mean_non_outliers = self.df_test[~outliers_mask[col]][col].mean()

            for i in outliers_mask[col][outliers_mask[col]].index:
                # Verifying that outliers have been correctly transformed to the mean of non-outlier values.
                self.assertAlmostEqual(
                    df_transformed.loc[i, col],
                    mean_non_outliers,
                    msg=f"Outlier in column {col} not replaced by mean",
                )

                # Verifying that outliers have been correctly identified.
                outlier_value = self.df_test.loc[i, col]
                lower_bound, upper_bound = (
                    detector.bounds[col]["lower"],
                    detector.bounds[col]["upper"],
                )
                self.assertTrue(
                    outlier_value < lower_bound or outlier_value > upper_bound,
                    f"Identified outlier {outlier_value} in column {col} is within bounds",
                )

    def test_iqr_capping_flooring(self):
        """
        Test using IQR method for outlier detection and capping/flooring for outlier replacement.
        """
        # Initializing the OutlierDetector with IQR method and capping/flooring replacement
        detector = outlierdetector(
            columns=self.columns, method="IQR", replacement_method="capping_flooring"
        )

        # Fitting outlier detection method (Calculating the bounds for outliers)
        detector.fit(self.df_test)

        # Identifying outliers in each column
        outliers_mask = {
            col: self.df_test[col].apply(lambda x: detector._is_outlier(x, col))
            for col in self.columns
        }

        # Replacing the outliers using capping and flooring
        df_transformed = detector.fit_transform(self.df_test)

        for col in self.columns:
            # Verifying that outliers have been correctly capped or floored
            for i in outliers_mask[col][outliers_mask[col]].index:
                transformed_value = df_transformed.loc[i, col]
                lower_bound, upper_bound = (
                    detector.bounds[col]["lower"],
                    detector.bounds[col]["upper"],
                )

                # Verifying that outliers are within the bounds after transformation
                self.assertTrue(
                    lower_bound <= transformed_value <= upper_bound,
                    f"Outlier in column {col} not correctly capped/floored",
                )

                # Verifying that outliers have been correctly identified
                outlier_value = self.df_test.loc[i, col]
                self.assertTrue(
                    outlier_value < lower_bound or outlier_value > upper_bound,
                    f"Identified outlier {outlier_value} in column {col} is within bounds before transformation",
                )


class TestMultiOptBinning(unittest.TestCase):

    def setUp(self):
        """
        Initializing the data for testing MultiOptBinning class.
        """
        # Setting the random seed
        np.random.seed(0)

        # Creating a synthetic DataFrame with 2 numerical columns
        n_samples = 100
        data = {
            "col1": np.random.normal(0, 1, n_samples),
            "col2": np.random.normal(0, 1, n_samples),
        }
        self.df_test = pd.DataFrame(data)

        ### Creating a binary target that has a relationship with features
        self.df_test["binary_target"] = (
            4 * self.df_test["col1"] - 3 * self.df_test["col1"] > 0
        ).astype(int)

        ### Creating a multiclass target variable that has a relationship with features
        combined_feature = 0.5 * (self.df_test["col1"] + self.df_test["col2"])
        first_tertile = np.percentile(combined_feature, 33)
        second_tertile = np.percentile(combined_feature, 66)
        self.df_test["multiclass_target"] = np.where(
            combined_feature <= first_tertile,
            0,
            np.where(combined_feature <= second_tertile, 1, 2),
        )

        ### Creating a continuous target variable that has a relationship with features
        self.df_test["continuous_target"] = (
            0.7 * self.df_test["col1"]
            + 0.3 * self.df_test["col2"]
            + np.random.normal(0, 1, n_samples)
        )

        # Initialize MultiOptBinning with some default parameters
        self.binner = multioptbinning(
            variables=["col1", "col2"],
            target="binary_target",
            target_dtype="binary",
            dtype="numerical",
            solver="cp",
            prebinning_method="cart",
            divergence="iv",
            min_n_bins=2,
            monotonic_trend="auto",
            min_event_rate_diff=0.02,
            outlier_value=-999.001,
            additional_optb_params=None,
        )

    def test_binary_target_binning(self):
        """
        Test the multioptbinning class with a binary target.
        """
        # Setting the binary target
        self.binner.target = "binary_target"
        self.binner.target_dtype = "binary"

        # Fitting the multioptbinning on the synthetic data
        self.binner.fit(self.df_test)

        # Transforming the features based on the fitted model
        transformed_df = self.binner.transform(self.df_test)

        # Verifying that the transformation produces meaningful output
        for variable in self.binner.variables:
            transformed_unique_values = transformed_df[variable].nunique()
            # Verifying that the result of transformation is of type 'category' or 'object'
            self.assertTrue(
                transformed_df[variable].dtype.name in ["category", "object"],
                f"The transformed data for {variable} is not of type 'category', indicating it may not be properly binned.",
            )
            self.assertTrue(
                transformed_unique_values >= self.binner.min_n_bins,
                f"The number of unique bins for {variable} is outside the expected range.",
            )

    def test_multiclass_target_binning(self):
        """
        Test the multioptbinning class with a multiclass target.
        """
        # Setting the target to the multiclass target for this test
        self.binner.target = "multiclass_target"
        self.binner.target_dtype = "multiclass"

        # Fitting the MultiOptBinning model on the synthetic data
        self.binner.fit(self.df_test)

        # Transforming the features based on the fitted model
        transformed_df = self.binner.transform(self.df_test)

        # Verifying that the transformation produces meaningful output
        for variable in self.binner.variables:
            transformed_unique_values = transformed_df[variable].nunique()

            # Verifying that the result of transformation is of type 'category' or 'object'
            self.assertTrue(
                transformed_df[variable].dtype.name in ["category", "object"],
                f"The transformed data for {variable} is not of type 'category', indicating it may not be properly binned.",
            )

            self.assertTrue(
                transformed_unique_values >= self.binner.min_n_bins,
                f"The number of unique bins for {variable} is outside the expected range.",
            )

    def test_continuous_target_binning(self):
        """
        Test the MultiOptBinning class with a continuous target.
        """
        # Setting the target to the continuous target for this test
        self.binner.target = "continuous_target"
        self.binner.target_dtype = "continuous"

        # Fitting the MultiOptBinning model on the synthetic data
        self.binner.fit(self.df_test)

        # Transforming the features based on the fitted model
        transformed_df = self.binner.transform(self.df_test)

        # Verifying that the transformation produces meaningful output
        for variable in self.binner.variables:
            transformed_unique_values = transformed_df[variable].nunique()
            # Verifying that the result of transformation is of type 'category'
            self.assertTrue(
                transformed_df[variable].dtype.name in ["category", "object"],
                f"The transformed data for {variable} is not of type 'category', indicating it may not be properly binned.",
            )
            self.assertTrue(
                transformed_unique_values >= self.binner.min_n_bins,
                f"The number of unique bins for {variable} is outside the expected range.",
            )


class TestLogisticRegressionPreparer(unittest.TestCase):
    """
    A class to conduct tests for the LogisticRegressionPreparer class.
    """

    def setUp(self):
        """
        Initializing the data for testing the LogisticRegressionPreparer class.
        """
        # Create a synthetic DataFrame for testing
        self.df_test = pd.DataFrame(
            {
                "feature1": ["A", "B", "C", "A", "B", "C", "A", "B"],
                "feature2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
                "numeric_feature": [1, 2, 3, 1, 2, 3, 1, 2],
            }
        )

        # Defining columns to be one-hot encoded and the categories to drop for each (if any)
        columns_to_encode = ["feature1", "feature2"]
        # Specifying categories to drop for one-hot encoding (ref categories)
        column_dict = {"feature1": "A", "feature2": "X"}

        # Initializing the LogisticRegressionPreparer with specified columns and column_dict
        self.preparer = logisticregressionpreparer(
            columns=columns_to_encode, column_dict=column_dict
        )

    def test_one_hot_encoding_transformation(self):
        """
        Test the one-hot encoding transformation of the LogisticRegressionPreparer class.
        """
        # Transform the DataFrame using the prepared LogisticRegressionPreparer instance
        transformed_df = self.preparer.fit_transform(self.df_test)

        # Verifying that the original categorical columns have been removed
        for col in self.preparer.columns:
            self.assertNotIn(
                col,
                transformed_df.columns,
                f"Column {col} was not removed after one-hot encoding.",
            )

        # Verifying that the total length equals the sum of unique categories minus the number of specified columns
        expected_columns = len(self.df_test.columns) - len(self.preparer.columns)
        for col in self.preparer.columns:
            expected_columns += len(self.df_test[col].unique()) - 1

        self.assertEqual(
            len(transformed_df.columns),
            expected_columns,
            "The total number of columns in the transformed DataFrame is incorrect.",
        )


if __name__ == "__main__":
    unittest.main()
