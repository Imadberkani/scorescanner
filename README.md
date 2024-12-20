

<p align="center">
  <img src="https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/logo_red.png" alt="ScoreScanner Logo" title="Logo de ScoreScanner">
</p>


## What is ScoreScanner ? 📋

**scorescanner** is a Python library designed to accelerate and simplify the process of understanding and quantifying the relationship between `features` and the `target variable`
 in the context of supervised predictive `Machine Learning` modeling on `tabular datasets`.

## Why and when to use ScoreScanner? 🤔

**Why use ScoreScanner?**

- **Efficiency:** Streamlines the exploration process, saving valuable time and effort ⏱️
- **Clarity:** Provides clear and quantifiable insights into the relationships between features and the target 🔆

**When to use ScoreScanner?**

when you aim to:

- Quickly identify the most significant features and gain a better understanding of their importance using statistical indicators and visualizations 📈
- Give meaning to missing values and outliers 🔎
- Perform optimal feature selection for interpretation ✔️
- Create an interpretable initial model 🧠
- Simplify communication with business teams 📢

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Tutorial](#quick-tutorial)

## Key Features

### ⚙️ Preprocessing 
- **Outlier Identification & Replacement**: Automatically detecting and replacing outliers.
- **Supervised Binning of Continuous Variables**: Converting continuous variables into categorical ones using supervised binning techniques for better interpretability. If no significant relationship with the target is detected, an unsupervised clustering algorithm, HDBSCAN, is used.

### 🕵️‍♂️ Feature Analysis 
- **Univariate Feature Importance**: Identifying the most impactful features on the target variable using statistical measures.
- **Divergent Category Identification**: Pinpoint the categories that deviate most from the target, providing deeper insights into data using `Jensen-Shannon` divergence.
- **Feature Clustering:** Clustering `Cramers'v` correlation matrix.

### ✔️ Feature Selection
- **Multicollinearity Elimination**: Reducing multicollinearity to ensure that model's predictors are independent, enhancing the stability and interpretability of a model.
- **Identifying Correlated Variable Subgroups:** Automatically grouping correlated variables, facilitating a nuanced interpretation of feature importance through the mean of absolute Shapley values.

### 🤖 Logistic Regression
- **Logistic Regression Report**: Generate detailed logistic regression reports, offering a clear view of how each independent variable influences the target.

## Installation

To install `scorescanner`, you can use pip:

```bash
pip install scorescanner
```

## Quick Tutorial

To start, let's import the "Adult" dataset from [UCI](https://archive.ics.uci.edu/dataset/2/adult), aimed at classifying individuals based on whether their income exceeds $50K/year.

```python
# Importing libraries
import os  
import pandas as pd  
import numpy as np  
import scorescanner.data  

# Loading the adult dataset 
adult_data = pd.read_csv(
    os.path.join(os.path.dirname(scorescanner.data.__file__), "adult_data.csv"),
    low_memory=False,  
)

# Setting a seed 
np.random.seed(42)

# Adding a random numerical feature from discrete uniform distribution 
adult_data['random_num_feature'] = np.random.randint(0, 100, size=len(adult_data)) 

# Adding a random categorical feature from discrete uniform distribution 
adult_data['random_category'] = np.random.choice(["cat1", "cat2", "cat3"], size=len(adult_data))


# Displaying first rows
adult_data.head()

```
![Adult DataFrame](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/adult_data_re.png)

### Preprocessing

Now, we propose two preprocessing steps:
- First, identifying and replacing `outliers` with `extreme value`.
- Second, applying `optimal binning` of `continuous variables`, which includes creating unique categories for `outliers` and `missing values`.

##### Preprocessing parameters

```python
# Target
target = "income"
# Numerical features
num_features = [
     col for col in adult_data.select_dtypes(include=np.number).columns.tolist() if adult_data[col].nunique() > 2
]
# Categorical features
cat_features = [
    "workclass",
    "education",
    "occupation"
]
# Value to replace outliers
outlier_value = -999.001
```

We can incorporate both steps into a **Scikit-learn pipeline**:


```python
from scorescanner.preprocessing import (
    multioptbinning,
    outlierdetector,
)
from sklearn.pipeline import Pipeline 
```


```python
# Defining the pipeline steps
pipeline_steps = [
    (
        "outlier_detection",
        outlierdetector(
            features=num_features,
            method="IQR",
            replacement_method="constant",
            replacement_value=outlier_value,
        ),
    ),
    (
        "optimal_binning",
        multioptbinning(
            num_features=num_features,
            cat_features=cat_features,
            target=target,
            target_dtype="binary",
            special_value=outlier_value,
            cat_features_info_json_file="cat_features_info_json_file.json"
        ),
    ),
]

# Creating the pipeline
data_preprocessing_pipeline = Pipeline(steps=pipeline_steps)

# Fitting the pipeline on the data
data_preprocessing_pipeline.fit(adult_data)

# Transforming the data 
adult_data_binned = data_preprocessing_pipeline.transform(adult_data)

#Overview of binned DataFrame
adult_data_binned.head()

```
![Binned DataFrame](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/binned_data_re.png)


### Univariate Feature Importance

Now, we can identify the most `impactful features` on the `target` variable using the univariate importance method: 


```python
from scorescanner.utils.statistical_metrics import (
    univariate_feature_importance,
    univariate_category_importance,
    calculate_cramers_v_matrix,
    cluster_corr_matrix
)

# Target variable and features list
target = 'income'
features = [col for col in adult_data.columns if col not in target]

# Calculate univariate feature importance
univariate_importance = univariate_feature_importance(
    df=adult_data_binned, features=features, target_var=target, method="cramers_v"
)

# Display the univariate feature importance
univariate_importance.style.bar(subset=["Univariate_Importance"], color="#5f8fd6")
```
![univariate importance](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/univariate_importance.png)

You can also use the Predictive Power Score (PPS) as an alternative metric for calculating univariate feature importance. For more detailed information, refer to the [documentation](https://pypi.org/project/ppscore/).



### Identifying Highly Divergent Categories from target
Now, we can identify the categories that `diverge` most from the target:

```python

univariate_category_importance(
    df=adult_data_binned, categorical_vars=features, target_var=target
)[0:30]

```
![Category divergence](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/category_importance.png)

The `Doctorate` category in the `education` feature shows the highest divergence from the overall distribution of the target feature, with a Jensen-Shannon distance of `0.36`, indicating that the distribution of the target among individuals with a doctorate differs significantly from the overall target distribution compared to other categories.


### Visualisation
Now, we can visualize the most important `measures` and statistical `metrics` of a variable in a **bar plot**:

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

![Bar plot](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/_bar_plot.png)

The right axis represents the percentage, allowing us to visualize the evolution of each `target modality` across all `bins`.

We can also focus on the **Weight of Evidence** or the **Jensen-Shannon metrics**.

```python

fig = plot_woe(
    df=adult_data_binned,
    feature="relationship",
    target_var=target,
    cat_ref=None
)
fig.show()

```
![woe plot](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/woe.png)

A positive value of `one-vs-rest WoE` indicates that the `reference category` dominates in the respective bin, meaning that the presence of this reference category in an observation increases the `likelihood` of the `reference category`.
```python

fig = plot_js(
    df=adult_data_binned,
    feature="relationship",
    target_var= target
    )
fig.show()

```
![js plot](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/js.png)

The larger the `Jensen-Shannon distance` between the `overall distribution` of the target feature and the `distribution of the target within each bin`, the more `significant` the differences are.

### Feature Clustering

Now, we propose calculating the Cramér's V correlation matrix and grouping variables by applying hierarchical clustering to the correlation matrix using Ward's method.

```python

corr_matrix = calculate_cramers_v_matrix(df=adult_data_binned, sampling=False)
corr_matrix_clustered = cluster_corr_matrix(corr_matrix=corr_matrix, threshold=1.7) 
plot_corr_matrix(corr_matrix_clustered)

```

![corr DataFrame](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/corr.png)

### Logistic Regression

##### Data Preparation

Now, we are going to use the `logisticregressionpreparer` class to prepare our data for training a logistic regression model. This class will enable us to perform the following data preparation steps:
- Perform One Hot Encoding for our categorical variables.
- Remove one variable for each categorical variable to reduce multicollinearity.
- Additionally, if we wish, we can define for each variable which category to remove, which will indirectly set our reference category. Otherwise, a category will be chosen randomly.

```python
from scorescanner.preprocessing import refcatencoder
```

```python
# Converting features to string
adult_data_binned[features] = adult_data_binned[features].astype(str)
# Dictionary for reference categories
column_dict = {
    "education-num": "(-inf, 8.50)",
    "capital-gain": "0.0",
    "education": " HS-grad",
    
}
# Initializing the DataPreparerForLogisticRegression
data_preparer = refcatencoder(
    columns=[col for col in features], column_dict=column_dict
)
# Applying the data preparation steps
prepared_df = data_preparer.fit_transform(adult_data_binned)
#Overview of prepared DataFrame
prepared_df.head()
```

![prepared_df](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/prepared_df.png)

##### Variable Selection (Pearson correlation method)
Given our focus on model robustness, a significant concern before training a logistic regression is the phenomenon of correlation and multicollinearity. These issues can substantially impact the interpretability of our model:

- Coefficients may seem insignificant, even when a significant relationship exists between the predictor and the response.
- Coefficients of strongly correlated predictors can vary considerably from one sample to another.
- When terms in a model are strongly correlated, removing one of these terms will significantly impact the estimated coefficients of the others. Coefficients of strongly correlated terms might even have the incorrect sign.

To address these concerns, we propose the following approach with the `select_uncorrelated_features` function:

- The function assesses each variable's correlation with the target, helping to identify the most relevant predictors.
- It then evaluates correlations among predictors, eliminating those that are strongly correlated with others (above a defined threshold).
- This process reduces multicollinearity, ensuring that remaining variables provide unique and valuable information for the model.

Another viable strategy to enhance model robustness, particularly against multicollinearity, involves prioritizing the most discriminative predictors while also rigorously assessing multicollinearity using the Variance Inflation Factor (VIF). This method can be summarized as follows:

- Initially, the model focuses on selecting predictors that have the highest discriminatory power with respect to the target variable.
- Subsequently, for the selected predictors, the VIF is calculated for each one. The VIF quantifies the extent of multicollinearity in regression analysis. It provides a measure of how much the variance of an estimated regression coefficient increases if your predictors are correlated.
- A common threshold from the literature is employed to decide which variables to retain. Typically, a VIF value exceeding 5 or 10 (depending on the specific criteria adopted) is indicative of substantial multicollinearity, warranting the removal of the corresponding predictor from the model.
- By employing this VIF-based approach, we ensure that the predictors included in the model are not only relevant but also minimally interdependent, thereby preserving the interpretability and stability of the model coefficients.

```python
from scorescanner.feature_selection import variableselector 
```

```python
# Initializing variableselector class
selector_pearson = variableselector(
    target="income", corr_threshold=0.2, metric="pearson", use_vif=False
)

# Fitting variableselector to data
selector_pearson.fit(prepared_df)

# Selected variables
print("Selected Variables:", selector_pearson.selected_variables)

# Filtering data to selected variables
selected_features_df = selector_pearson.transform(prepared_df)

selected_features_df.head()
```

We can open the JSON file to understand the flow of the algorithm. It contains key-value pairs where the key is the selected variable, and the value is a list of variables removed due to their strong correlation with the key variable.


```python
import json
with open("eliminated_variables_info.json", 'r') as file:
    data = json.load(file)
print(json.dumps(data, indent=3))
```

![json file](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/json_file.png)

##### Splitting the Training and Test Sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    prepared_df[selector_pearson.selected_variables],
    prepared_df["income"],
    test_size=0.1,
    random_state=42,
    stratify=prepared_df["income"],
)
```

##### Fitting the model

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver="newton-cholesky", random_state=42)
logreg.fit(X_train, y_train)
```

![fit](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/logisticregression_sk.png)

##### Logistic Regression Report 

```python
from scorescanner.utils.statistical_metrics import logistic_regression_summary

logistic_regression_report = logistic_regression_summary(
    model=logreg,
    X=X_train,
    columns=X_train.columns.tolist(),
    y=y_train,
    intercept=True,
    multi_class=False,
).reset_index()

logistic_regression_report


```

![model report](https://github.com/Imadberkani/scorescanner/blob/master/scorescanner/_images/logisticregression_report.png)