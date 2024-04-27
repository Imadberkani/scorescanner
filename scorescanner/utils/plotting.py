"""
plotting.py file includes a collection of functions that will be used in 'scoreinsight'
to create graphs for univariate and bivariate data analysis.
Functions:
bins_stats: calculate statistics for each bins of categorical feature   
generate_bar_plot : Generates a Plotly figure to visualize various statistics for a specified categorical feature 
plot_woe : Generates a Plotly figure to visualize the Weight of Evidence (WoE) for a specified categorical feature 
           in relation to a binary target variable.
plot_js: Generates a Plotly figure to visualize the Jensen-Shannon Distance for each category   
plot_corr_matrix: Generates a Plotly figure to visualize the correlation matrix.         
"""

# importing librairies
import pandas as pd
import numpy as np
from scorescanner.utils.statistical_metrics import (
    calculate_js_distances,
    one_vs_rest_woe,
    cramers_v,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def bins_stats(
    df: pd.DataFrame, feature: str, target_var: str, cat_ref=None
) -> pd.DataFrame:
    """
    For each category of a categorical feature, calculates the total count, the count when crossed with a binary target
    (e.g., number of 0s and 1s), the percentages, and the Jensen-Shannon distances.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    feature (str): The categorical feature.
    target_var (str): The binary target variable.
    cat_ref (str): The reference category for WoE calculation.

    Returns:
    pd.DataFrame: DataFrame with all necessary statistics, Jensen-Shannon distances, and WoE.
    """

    # Calculating bins statistics
    count = df[feature].value_counts().reset_index()
    count.columns = [feature, "count"]

    bins_count = df.groupby(feature)[target_var].value_counts().unstack(fill_value=0)
    bins_count.columns = [f"count_{col}" for col in bins_count.columns]

    stats = count.merge(bins_count, left_on=feature, right_index=True)
    stats["total_percent"] = (stats["count"] / stats["count"].sum()) * 100

    for col in bins_count.columns:
        stats[f'percent_{col.split("_")[1]}'] = (
            stats[col] / stats["count"] * 100
        ).fillna(0)

    # Calculating Jensen-Shannon distances
    js_distances = calculate_js_distances(df, feature, target_var)
    # Calculating WoE
    woe_df = one_vs_rest_woe(df, feature, target_var, cat_ref)[0]
    # Merging bins stats with Jensen-Shannon distances
    stats_js = stats.merge(
        js_distances, left_on=feature, right_on="Category", how="left"
    )
    # Merging bins stats with WoE
    final_stats = stats_js.merge(
        woe_df, left_on=feature, right_on="Category", how="left"
    )
    # Extracting upper bounds from intervals
    final_stats["max_of_interval"] = final_stats[feature].apply(
        lambda x: float(x.split(",")[1].strip(")").strip("]")) if "," in x else np.nan
    )

    return final_stats.sort_values(by="max_of_interval").drop(
        columns=["max_of_interval", "Category_x", "Category_y"]
    )


def generate_bar_plot(
    df: pd.DataFrame,
    feature: str,
    target_var: str,
    cat_ref=None,
    colors=["#82E0AA", "#EC7063", "#5DADE2", "#F7DC6F", "#FDEDEC", "#F4D03F"],
):
    """
    Generates a Plotly figure to visualize various statistics for a specified categorical feature
    in relation to a binary target variable. The function creates a bar plot and a line plot for
    percentages.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    feature (str): The name of the categorical feature to be visualized.
    target_var (str): The name of the binary target variable.
    cat_ref (str, optional): The reference category for calculating Weight of Evidence (WoE).
                             If None, a category is chosen randomly.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure with a bar plot for counts of each category
                                     in 'feature' and a line plot for their corresponding percentages
                                     in relation to 'target'. The line plot uses a secondary y-axis.

    The function uses 'bins_stats' to calculate necessary statistics for 'feature' and 'target',
    and then visualizes these statistics. The visualization includes count, percentage, WoE,
    and Jensen-Shannon Distance for each category in 'feature'.
    """
    # Calculate statistics using the bins_stats function.
    stats_df = bins_stats(df, feature, target_var, cat_ref)

    # Detect columns starting with 'count_' and 'percent_' in stats_df.
    count_cols = [col for col in stats_df.columns if col.startswith("count_")]
    percent_cols = [f'percent_{col.split("_")[1]}' for col in count_cols]

    # Calculate Cramer's V and Information Value
    cramers_v_value = cramers_v(df[feature], df[target_var])
    info_value = one_vs_rest_woe(
        df=df, feature=feature, target_var=target_var, cat_ref=cat_ref
    )[1]

    # Create a Plotly subplot with a secondary y-axis for percentages.
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for count_col, percent_col, color in zip(count_cols, percent_cols, colors):
        modalite = count_col.split("_")[-1]
        # bar plot for each 'count_col'
        fig.add_trace(
            go.Bar(
                x=stats_df[feature],
                y=stats_df[count_col],
                name=modalite,
                marker=dict(color=color),
                hoverinfo="all",
                hovertemplate=stats_df.apply(
                    lambda row: f"<b>{count_col}: {row[count_col]}<br>"
                    f"Percentage: {row[percent_col]:.2f}%<br>"
                    f"WoE: {row['WoE']:.2f}<br>"
                    f"JS Distance: {row['Jensen-Shannon Distance']:.2f}",
                    axis=1,
                ),
                text=stats_df.apply(
                    lambda row: f"<b>{count_col}: {row[count_col]}<br>"
                    f"Percentage: {row[percent_col]:.2f}%<br>",
                    axis=1,
                ),
                customdata=stats_df[percent_col],
            ),
            secondary_y=False,
        )

        # Line for each 'percent_col'
        fig.add_trace(
            go.Scatter(
                x=stats_df[feature],
                y=stats_df[percent_col],
                name=f"Percent {modalite}",
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color),
            ),
            secondary_y=True,
        )

    # Configuring the layout of the chart
    fig.update_layout(
        title={
            "text": f"<b>Distribution of '{feature}'</b>",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20, "color": "black", "family": "Arial, sans-serif"},
        },
        xaxis_title=f"<b>{feature}",
        yaxis_title="<b>Count",
        yaxis2_title="<b>Percentage",
        barmode="stack",
        hovermode="x",
        template="plotly_white",
        legend_title_text="Target Modalities",
    )

    # Adding annotations for Cramer's V and Information Value
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=1.05,
        text=f"<b>Cramer's V: {cramers_v_value:.3f}, Information Value: {info_value:.3f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
        bordercolor="black",
        borderwidth=1,
    )

    return fig


def plot_woe(
    df: pd.DataFrame, feature: str, target_var: str, cat_ref=None, colors=["#9B59B6"]
):
    """
    Generates a Plotly figure to visualize the Weight of Evidence (WoE) for a specified categorical feature
    in relation to a binary target variable.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    feature (str): The name of the categorical feature.
    target_var (str): The name of the binary target variable.
    cat_ref (str, optional): The reference category for WoE calculation. If None, a category is chosen randomly.
    colors (list, optional): List of colors for the plot.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure with a bar plot representing the WoE for each category in 'feature'.
    """
    # Calculate WoE using one_vs_rest_woe function or similar
    woe_df = one_vs_rest_woe(df, feature, target_var, cat_ref)[0]
    # Separating 'Special' and 'Missing' categories
    special_missing_rows = woe_df[woe_df["Category"].isin(["Special", "Missing"])]
    final_stats = woe_df[~woe_df["Category"].isin(["Special", "Missing"])].sort_values(
        by="Category"
    )

    # Sorting the remaining DataFrame by feature and appending 'Special' at the end
    woe_df = pd.concat([final_stats, special_missing_rows])
    # Calculate Information Value
    info_value = one_vs_rest_woe(df, feature, target_var, cat_ref)[1]

    # Create Plotly figure
    fig = go.Figure()

    # Add bar trace for WoE values with improved tooltip and text
    fig.add_trace(
        go.Bar(
            x=woe_df["Category"],
            y=woe_df["WoE"],
            marker=dict(color=colors[0]),
            hoverinfo="text",  # Use text for hover information
            text=woe_df.apply(lambda row: f"<br>WoE: {row['WoE']:.2f}", axis=1),
            hovertext=woe_df.apply(
                lambda row: f"Category: {row['Category']}<br>WoE: {row['WoE']:.2f}",
                axis=1,
            ),
        )
    )

    # Update layout with centered title and improved tooltip
    fig.update_layout(
        title={
            "text": f"<b>Weight of Evidence for {feature} feature",
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title=f"<b>{feature}",
        yaxis_title="<b>WoE",
        template="plotly_white",
        hovermode="closest",  # Improve hover interaction
    )

    # Adding annotation for Information Value
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=1.05,
        text=f"<b>Information Value: {info_value:.3f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
        bordercolor="black",
        borderwidth=1,
    )

    return fig


def plot_js(df: pd.DataFrame, feature: str, target_var: str, colors=["#F7DC6F"]):
    """
    Generates a Plotly figure to visualize the Jensen-Shannon Distance for each category
    of a specified categorical feature in relation to a binary target variable.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    variable (str): The name of the categorical variable.
    target_var (str): The name of the binary target variable.
    colors (list, optional): List of colors for the plot.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure with a bar plot representing the Jensen-Shannon Distance for each category in 'feature'.
    """
    # Calculate JS distances using calculate_js_distances function or similar
    js_df = calculate_js_distances(df, feature, target_var)
    # Separating 'Special' and 'Missing' categories
    special_missing_rows = js_df[js_df["Category"].isin(["Special", "Missing"])]
    final_stats = js_df[~js_df["Category"].isin(["Special", "Missing"])].sort_values(
        by="Category"
    )

    # Sorting the remaining DataFrame by feature and appending 'Special' at the end
    js_df = pd.concat([final_stats, special_missing_rows])

    # Create Plotly figure
    fig = go.Figure()

    # Add bar trace for JS Distance values
    fig.add_trace(
        go.Bar(
            x=js_df["Category"],
            y=js_df["Jensen-Shannon Distance"],
            marker=dict(color=colors[0]),
            hoverinfo="text",  # Use text for hover information
            text=js_df.apply(
                lambda row: f"<b>JS: {row['Jensen-Shannon Distance']:.2f}", axis=1
            ),
            hovertext=js_df.apply(
                lambda row: f"<b>Category: {row['Category']}<br>JS: {row['Jensen-Shannon Distance']:.2f}",
                axis=1,
            ),
        )
    )

    # Update layout with centered title and improved tooltip
    fig.update_layout(
        title={
            "text": f"<b>Divergence from Target Categories: {feature}</b>",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title=f"<b>{feature}",
        yaxis_title="<b>JS Distance",
        template="plotly_white",
    )

    return fig


def plot_corr_matrix(corr_matrix: pd.DataFrame, width: int = 1200, height: int = 700):
    """
    Generates a Plotly figure to visualize the correlation matrix.

    Parameters:
    corr_matrix (pd.DataFrame): A pandas DataFrame representing the correlation matrix.
    width (int, optional): Width of the plot. Defaults to 1200.
    height (int, optional): Height of the plot. Defaults to 700.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure of the correlation heatmap.
    """
    # Create a figure with Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=round(corr_matrix, 2),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="Tealgrn",
            zmin=-1,
            zmax=1,
            hoverongaps=False,
        )
    )

    # Update layout for a better view
    fig.update_layout(
        title={
            "text": "<b>Correlation Heatmap",
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis={"side": "bottom"},
        yaxis_autorange="reversed",
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig
