# ScoreScanner

**ScoreScanner** is a Python library designed to accelerate and simplify the process of understanding and quantifying the relationship between features and the target variable in the context of predictive Machine Learning modeling.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Tutorial](#quick-tutorial)

## Key Features

### Preprocessing
- **Outlier Identification & Replacement**: Automatically detecting and replacing outliers.
- **Supervised Binning of Continuous Variables**: Converting continuous variables into categorical ones using supervised binning techniques for better interpretability.

### Feature Analysis
- **Univariate Feature Importance**: Identifying the most impactful features on the target variable using statistical measures.
- **Divergent Category Identification**: Pinpoint the categories that deviate most from the target, providing deeper insights into data using Jensen-Shannon divergence.
- **Feature Clustering:** Clustering Cramers'v correlation matrix.

### Feature Selection
- **Multicollinearity Elimination**: Reducing multicollinearity to ensure that model's predictors are independent, enhancing the stability and interpretability of a model.
- **Identifying Correlated Variable Subgroups:** Automatically grouping correlated variables, facilitating a nuanced interpretation of feature importance through the mean of absolute Shapley values.

### Logistic Regression
- **Logistic Regression Report**: Generate detailed logistic regression reports, offering a clear view of how each independent variable influences the target.

## Installation

To install ScoreScanner, you can use pip:

```bash
pip install scorescanner
```

## Quick Tutorial

To start, let's import the "Adult" dataset from UCI, aimed at classifying individuals based on whether their income exceeds $50K/year.

```python

import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
adult_data = pd.read_csv(url, names=columns)

```
### Preprocessing

Now, we propose two preprocessing steps:
- First, identifying and replacing outliers with extreme value.
- Second, applying optimal binning of continuous variables, which includes creating unique categories for outliers and missing values.

We can incorporate both steps into a Scikit-learn pipeline:


```python

#Target
target='income'
#Numerical features
num_features=[col for col in columns if adult_data[col].dtypes in ['int64'] and col not in target]
#Value to replace outliers
outlier_value = -999.001


# Defining the pipeline steps
pipeline_steps = [
    ('outlier_detection', outlierdetector(
        columns=num_features,
        method="IQR",
        replacement_method="constant",
        replacement_value=outlier_value,
    )),
    ('optimal_binning', multioptbinning(
        variables=num_features,
        target=target,
        target_dtype="multiclass",
        outlier_value=outlier_value,
    ))
]

# Creating the pipeline
data_preprocessing_pipeline = Pipeline(steps=pipeline_steps)

# Fitting the pipeline on the data
data_preprocessing_pipeline.fit(adult_data)

# Transforming the data 
adult_data_binned = data_preprocessing_pipeline.transform(adult_data)

```

### Univariate Feature Importance

Now, we can identify the most impactful features on the target variable using the univariate importance method: 


```python

from scorescanner.utils.statistical_metrics import (
    univariate_feature_importance,
    univariate_category_importance,
    calculate_cramers_v_matrix,
    cluster_corr_matrix
)

# Target variable and features list
target = 'income'
features = [col for col in columns if col not in target]

# Calculate univariate feature importance
univariate_importance = univariate_feature_importance(
    df=adult_data_binned, features=features, target_var=target, method="cramerv"
)

# Display the univariate feature importance
univariate_importance


```
![Univariate Feature Importance](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/category_importance.png)

### Identifying Highly Divergent Categories from target
Now, we can identify the categories that diverge most from the target:

```python

univariate_category_importance(
    df=adult_data_binned, categorical_vars=features, target_var=target
)[0:30]

```
![Univariate Feature Importance](/_images/scorescanner/univariate_importance.png)
### Visualisation
Now, we can visualize the most important measures and statistical metrics of a variable in a bar plot:

```python
from scorescanner.utils.plotting import (
    generate_bar_plot,
    plot_woe,
    plot_js,
    plot_corr_matrix
)
```


```python

fig = generate_bar_plot(
    df=adult_data_binned,
    feature="relationship",
    target_var=target,
    cat_ref=None,
)
fig.show()

```



The right axis represents the percentage, allowing us to visualize the evolution of each target modality across all bins.

We can also focus on the Weight of Evidence or the Jensen-Shannon metrics.

```python

fig = plot_woe(
    df=adult_data_binned, feature="relationship", target_var=target, cat_ref=None
)
fig.show()

```

```python

fig = plot_js(
    df=adult_data_binned,feature="relationship",target_var= target
    )
fig.show()

```

### Feature Clustering


```python

corr_matrix = calculate_cramers_v_matrix(df=adult_data_binned, sampling=False)
corr_matrix_clustered = cluster_corr_matrix(corr_matrix=corr_matrix, threshold=1.7) 
plot_corr_matrix(corr_matrix_clustered)

```

