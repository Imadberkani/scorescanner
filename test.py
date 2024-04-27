import os
import pandas as pd
import scorescanner.data

adult_data = pd.read_csv(
    os.path.join(os.path.dirname(scorescanner.data.__file__), "adult_data.csv"),
    low_memory=False,
)
adult_data.head()


# Target
target = "income"
# Numerical features
num_features = [
    col for col in columns if adult_data[col].dtypes in ["int64"] and col not in target
]
# Value to replace outliers
outlier_value = -999.001


# Defining the pipeline steps
pipeline_steps = [
    (
        "outlier_detection",
        outlierdetector(
            columns=num_features,
            method="IQR",
            replacement_method="constant",
            replacement_value=outlier_value,
        ),
    ),
    (
        "optimal_binning",
        multioptbinning(
            variables=num_features,
            target=target,
            target_dtype="binary",
            outlier_value=outlier_value,
        ),
    ),
]

# Creating the pipeline
data_preprocessing_pipeline = Pipeline(steps=pipeline_steps)

# Fitting the pipeline on the data
data_preprocessing_pipeline.fit(adult_data)

# Transforming the data
adult_data_binned = data_preprocessing_pipeline.transform(adult_data)

# Overview of binned DataFrame
adult_data_binned.head()


from scorescanner.utils.statistical_metrics import (
    univariate_feature_importance,
    univariate_category_importance,
    calculate_cramers_v_matrix,
    cluster_corr_matrix,
)

# Target variable and features list
target = "income"
features = [col for col in columns if col not in target]

# Calculate univariate feature importance
univariate_importance = univariate_feature_importance(
    df=adult_data_binned, features=features, target_var=target, method="cramerv"
)

# Display the univariate feature importance
univariate_importance
