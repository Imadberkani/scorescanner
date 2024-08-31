"""
statistical_metrics.py file includes a collection of functions that will be used in 'scorescanner' to calculate statistical metrics.

Functions:
cramers_v : Calculate the Cramér's V coefficient to measure the association between two categorical variables of a ""DataFrame"".
univariate_feature_importance : Calculate and rank features based on their importance.
calculate_js_distances : Calculate the Jensen-Shannon distance for each category of an explanatory variable by
                          measuring its divergence from the target variable.
univariate_category_importance : Calculate the Jensen-Shannon distance for each category of each categorical variable 
                                    in a DataFrame.           
one_vs_rest_woe : Calculate the Weight of Evidence (WoE), considering a specific category as reference and grouping 
                   the rest.                
calculate_cramers_v_matrix : Calculate the Cramér's V correlation matrix for a DataFrame.
cluster_corr_matrix : Reorganizes a correlation matrix based on hierarchical clustering of columns.

"""

# Importing Libraries
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import ppscore as pps
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate the Cramér's V coefficient between two pd.Series.

    Parameters:
    x, y (pd.Series): Two pandas Series.

    Returns:
    float: Calculated Cramér's V coefficient.

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


def univariate_feature_importance(
    df: pd.DataFrame, features: list, target_var: str, method: str
) -> pd.DataFrame:
    """
    Calculate and rank features based on their importance using either Cramér's V or Predictive Power Score (PPS)
    in relation to a target variable, depending on the chosen method.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data.
    features (list): A list of feature names (str) to be evaluated.
    target_var (str): The name of the target variable.
    method (str): The method to be used for calculating importance ('cramers_v' or 'ppscore').

    Returns:
    pd.DataFrame: A DataFrame containing features and their corresponding importance scores,
    sorted in descending order of importance.
    """
    # Supported methods
    if method not in ["cramers_v", "ppscore"]:
        raise ValueError("Method must be 'cramers_v' or 'ppscore'")

    importance_scores = []
    # Calculating 'cramerv' or 'ppscore' for each feature
    for feature in features:
        if feature not in df.columns or target_var not in df.columns:
            raise ValueError("Specified variables must be in the DataFrame")

        if method == "cramers_v":
            score = cramers_v(df[feature], df[target_var])
        elif method == "ppscore":
            score = pps.score(df, feature, target_var)["ppscore"]

        importance_scores.append((feature, score))

    importance_df = pd.DataFrame(
        importance_scores, columns=["Feature", "Univariate_Importance"]
    )
    # Sorting the DataFrame by importance
    importance_df = importance_df.sort_values(
        by="Univariate_Importance", ascending=False
    )

    return importance_df


def calculate_js_distances(
    df: pd.DataFrame, feature: str, target_var: str
) -> pd.DataFrame:
    """
    Calculate the Jensen-Shannon distance for each category of an categorical feature by
    measuring its divergence from the target variable.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data.
    feature (str): The name of the feature.
    target_var (str): The name of the target variable.

    Returns:
    pd.DataFrame: A DataFrame containing the categories of the feature
    and their corresponding Jensen-Shannon distances to the target variable distribution.
    """

    if feature not in df.columns or target_var not in df.columns:
        raise ValueError("Specified variables must be in the DataFrame")

    # Calculating the probability distributions for each category of the feature
    prob_distributions = pd.crosstab(df[feature], df[target_var]).apply(
        lambda x: x / x.sum(), axis=1
    )

    # Calculating the global distribution of the target variable
    global_distribution = (
        df[target_var]
        .value_counts(normalize=True)
        .reindex(prob_distributions.columns, fill_value=0)
    )

    # Calculating the Jensen-Shannon distance for each category
    js_distances = prob_distributions.apply(
        lambda row: jensenshannon(row, global_distribution), axis=1
    )

    # Creating a DataFrame to display the results
    js_distances_df = pd.DataFrame(js_distances, columns=["Jensen-Shannon Distance"])
    js_distances_df.reset_index(inplace=True)
    js_distances_df.rename(columns={feature: "Category"}, inplace=True)

    return js_distances_df.sort_values(by="Jensen-Shannon Distance", ascending=False)


def univariate_category_importance(
    df: pd.DataFrame, categorical_vars: list, target_var: str
) -> pd.DataFrame:
    """
    Calculate the Jensen-Shannon distance for each category of each categorical variable
    in a DataFrame in relation to a target variable and return a combined DataFrame
    with all categories and their distances.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data.
    categorical_vars (list): A list of names of categorical variables.
    target_var (str): The name of the target variable.

    Returns:
    pd.DataFrame: A DataFrame containing the variable name, category, and their
    Jensen-Shannon distances to the target variable distribution.

    Raises:
    ValueError: If the specified variables are not found in the DataFrame.
    """

    all_distances = []
    # Calculating the Jensen-Shannon distance for each category of each categorical variable
    for cat_var in categorical_vars:
        if cat_var not in df.columns or target_var not in df.columns:
            raise ValueError("Specified variables must be in the DataFrame")

        js_distances_df = calculate_js_distances(df, cat_var, target_var)
        js_distances_df["Variable"] = cat_var
        all_distances.append(js_distances_df)

    # Concatenating all distance DataFrames
    category_importance_df = pd.concat(all_distances)
    category_importance_df.rename(columns={cat_var: "Category"}, inplace=True)

    return category_importance_df.sort_values(
        by="Jensen-Shannon Distance", ascending=False
    )


def one_vs_rest_woe(
    df: pd.DataFrame, feature: str, target_var: str, cat_ref=None
) -> pd.DataFrame:
    """
    Calculate the Weight of Evidence (WoE), considering a specific category
    as reference and grouping the rest.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data.
    feature (str): The name of the categorical explanatory variable.
    target_var (str): The name of the target variable.
    cat_ref: The reference category in the categorical variable.


    Returns:
    pd.DataFrame: A DataFrame containing the WoE for each category of the explanatory variable.
    """

    # Validation of input variables
    if target_var not in df.columns or feature not in df.columns:
        raise ValueError("Specified variables must be in the DataFrame")

    if cat_ref is not None and cat_ref not in df[target_var].unique():
        raise ValueError(
            f"The specified reference category '{cat_ref}' is not a valid category in the target variable '{target_var}'."
        )

    # Selecting a random category as reference if cat_ref is None
    if cat_ref is None:
        cat_ref = np.random.choice(df[target_var].unique())
        
    # Creating a copy of the DataFrame
    df_copy = df[[feature, target_var]].copy()

    # Identifying the 'rest' categories and grouping them
    rest_cat = [col for col in df_copy[target_var].unique() if col != cat_ref]
    df_copy[target_var] = df_copy[target_var].replace(rest_cat, "Rest")

    # Calculating the total number of cases for reference and non-reference
    total_ref = df_copy[df_copy[target_var] == cat_ref].shape[0]
    total_non_ref = df_copy[df_copy[target_var] == "Rest"].shape[0]

    # Initializing lists to store results
    categories, woe_values = [], []

    # Calculating WoE for each category
    for cat in df_copy[feature].unique():
        num_ref = df_copy[
            (df_copy[feature] == cat) & (df_copy[target_var] == cat_ref)
        ].shape[0]
        num_non_ref = df_copy[
            (df_copy[feature] == cat) & (df_copy[target_var] == "Rest")
        ].shape[0]

        # Avoiding division by zero
        if num_ref == 0 or num_non_ref == 0:
            woe = 0
        else:
            woe = np.log((num_ref / total_ref) / (num_non_ref / total_non_ref))

        categories.append(cat)
        woe_values.append(woe)

    # Calculating IV for each category and initializing IV total
    iv_values = []
    total_iv = 0

    for cat, woe in zip(categories, woe_values):
        num_ref_cat = df_copy[
            (df_copy[feature] == cat) & (df_copy[target_var] == cat_ref)
        ].shape[0]
        num_non_ref_cat = df_copy[
            (df_copy[feature] == cat) & (df_copy[target_var] == "Rest")
        ].shape[0]

        prop_ref = num_ref_cat / total_ref if total_ref != 0 else 0
        prop_non_ref = num_non_ref_cat / total_non_ref if total_non_ref != 0 else 0

        iv = (prop_ref - prop_non_ref) * woe
        iv_values.append(iv)
        total_iv += iv

    # Creating the final DataFrame
    woe_df = pd.DataFrame({"Category": categories, "WoE": woe_values, "IV": iv_values})

    return woe_df, total_iv

def calculate_cramers_v_matrix(df: pd.DataFrame, sampling: bool = True) -> pd.DataFrame:
    """
    Calculate the Cramér's V correlation matrix for a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame for which the correlation matrix is to be calculated.
    sampling (bool, optional): Whether to sample the data. Defaults to True.

    Returns:
    pd.DataFrame: A DataFrame representing the Cramér's V correlation matrix.
    """
    # Create a copy of a DatFrame
    df_copy = df.copy()
    # Sample the DataFrame if sampling is True
    if sampling:
        sample_size = min(50000, len(df_copy))
        df_sampled = df_copy.sample(n=sample_size, random_state=1)
    else:
        df_sampled = df_copy

    # Ordinal encoding of all columns
    for col in df_sampled.columns:
        df_sampled[col] = df_sampled[col].astype("category").cat.codes

    # Apply cramers_v_bis for each pair of columns in the sampled DataFrame
    corr_matrix = df_sampled.apply(
        lambda col1: df_sampled.apply(lambda col2: cramers_v(col1, col2))
    )

    return corr_matrix


def cluster_corr_matrix(
    corr_matrix: pd.DataFrame,
    threshold: float = 2.0,
    linkage_method: str = "ward",
    criterion: str = "distance",
    depth: int = 2,
    R: int = None,
    monocrit: int = None,
) -> pd.DataFrame:
    """
    Reorganizes a correlation matrix based on hierarchical clustering of columns.

    Parameters:
    corr_matrix (pd.DataFrame): A pandas DataFrame representing the correlation matrix.
    threshold (float, optional): The threshold to apply when forming flat clusters. Defaults to 2.0.
    linkage_method (str, optional): The linkage algorithm to use. Defaults to 'ward'.
    criterion (str, optional): The criterion to use in forming flat clusters. Defaults to 'distance'.
    depth (int, optional): The maximum depth for the calculation. Defaults to 2.
    R (int, optional): The number of inconsistent metrics to consider. Defaults to None.
    monocrit (int, optional): The array for mono-criterion. Defaults to None.

    Returns:
    pd.DataFrame: A reorganized correlation matrix based on the clustering of columns.
    """
    # Calculate the dissimilarity
    dissimilarity = 1 - abs(corr_matrix)

    # Perform hierarchical clustering
    Z = linkage(dissimilarity, method=linkage_method)

    # Form flat clusters
    labels = fcluster(
        Z, t=threshold, criterion=criterion, depth=depth, R=R, monocrit=monocrit
    )

    # Create a mapping of column names to cluster labels
    column_clusters = {name: label for name, label in zip(corr_matrix.columns, labels)}

    # Sort columns by cluster label
    sorted_columns = sorted(column_clusters, key=lambda x: column_clusters[x])

    # Reorganize corr_matrix based on the sorted order of columns
    reorganized_corr_matrix = corr_matrix[sorted_columns].reindex(sorted_columns)

    return reorganized_corr_matrix


def logistic_regression_summary(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: np.ndarray,
    columns: list[str] = None,
    intercept: bool = True,
    multi_class: bool = False,
    cat: int = None,
) -> pd.DataFrame:
    """
    This function takes a trained logistic regression model and training data X as input.
    It returns a table containing:
     - The intercept and all explanatory variables. The columns are: coef, std error, z, 'P>|z|', [0.025, 0.975]
     - If intercept is True, the intercept is added to X.

    Parameters:
    model: A trained logistic regression model.
    X: The training data on which the model was trained.
    y: The set of target values.
    columns: The names of the columns in X, optional.
    intercept: Indicates whether to add an intercept to the X matrix or not. Defaults to True.
    multi_class: Indicates if the model is multi-class. Defaults to False.
    cat: Specifies the category for which to calculate coefficients in a multi-class scenario.

    Returns:
    pd.DataFrame: A table containing the coefficients, standard errors, z-values, p-values,
    and the lower and upper bounds of the 95% confidence interval.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    if intercept:
        X_mat = sm.add_constant(X_standardized)
        if multi_class and cat is not None:
            cat_index = list(model.classes_).index(cat)
            model_coef = model.coef_[cat_index]
            model_coef = np.insert(model_coef, 0, model.intercept_[cat_index])
        else:
            model_coef = np.insert(model.coef_[0], 0, model.intercept_[0])
    else:
        X_mat = X_standardized
        if multi_class and cat is not None:
            cat_index = list(model.classes_).index(cat)
            model_coef = model.coef_[cat_index]
        else:
            model_coef = model.coef_[0]

    p_one_minus_p_probs = np.product(
        model.predict_proba(X_standardized), axis=1
    ).reshape(-1, 1)
    cov_mat = np.linalg.pinv(np.dot((X_mat * p_one_minus_p_probs).T, X_mat))
    std_errors = np.sqrt(np.diag(cov_mat))
    z = model_coef / std_errors
    p_values = stats.norm.sf(np.abs(z)) * 2
    ci_low = stats.norm.ppf(0.025) * std_errors + model_coef
    ci_high = stats.norm.ppf(0.975) * std_errors + model_coef

    if columns and intercept:
        summary = pd.DataFrame(
            index=["intercept"] + columns,
            columns=["coef", "std error", "z", "P>|z|", "[0.025", "0.975]"],
        )
    elif columns:
        summary = pd.DataFrame(
            index=columns,
            columns=["coef", "std error", "z", "P>|z|", "[0.025", "0.975]"],
        )
    else:
        summary = pd.DataFrame(
            columns=["coef", "std error", "z", "P>|z|", "[0.025", "0.975]"]
        )
    summary.index.name = "variable"

    summary["coef"] = np.exp(
        model_coef.reshape(
            -1,
        )
    )
    summary["std error"] = std_errors.reshape(
        -1,
    )
    summary["z"] = z.reshape(
        -1,
    )
    summary["P>|z|"] = p_values.reshape(
        -1,
    ).round(4)
    summary["[0.025"] = np.exp(
        ci_low.reshape(
            -1,
        )
    )
    summary["0.975]"] = np.exp(
        ci_high.reshape(
            -1,
        )
    )
    summary = (
        summary.round(2).reset_index().iloc[(-np.log(summary["coef"]).abs()).argsort()]
    )

    return summary
