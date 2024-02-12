"""
utils.py file includes a collection of functions that will be used throughout the scorescanner tutorials.

Functions:

preprocess_credit_score_data : Function to preprocess the credit score classification DataFrame.
"""

# Importing Libraries
import numpy as np


def preprocess_credit_score_data(df):
    """
    Function to preprocess the credit score classification DataFrame. This function performs multiple tasks:
    1. Removes columns that are unlikely to have predictive power ('ID', 'Customer_ID', 'Name', 'SSN').
    2. Cleans specific columns by removing non-numeric characters and converting them to float type.
    3. Processes the 'Type_of_Loan' column to create binary features for each loan type.
    4. Extracts the number of years from 'Credit_History_Age' and converts it to a float.

    Parameters:
    df (DataFrame): The credit score classification DataFrame to be preprocessed.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    # List of columns to remove, typically these are unique identifiers
    columns_to_remove = ["ID", "Customer_ID", "Name", "SSN"]

    # List of columns to clean and convert to float
    columns_to_convert = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Outstanding_Debt",
        "Monthly_Balance",
        "Amount_invested_monthly",
    ]

    # List of loan types for binary feature creation
    loan_types = [
        "Credit-Builder Loan",
        "Personal Loan",
        "Debt Consolidation Loan",
        "Student Loan",
        "Payday Loan",
        "Mortgage Loan",
        "Auto Loan",
        "Home Equity Loan",
    ]

    # Removing the listed columns
    df = df.drop(columns=columns_to_remove)

    # Cleaning and converting specified columns to float
    df[columns_to_convert] = (
        df[columns_to_convert]
        .replace(to_replace=r"[^\d.]+", value="", regex=True)
        .astype(float)
    )
    df["Changed_Credit_Limit"] = (
        df["Changed_Credit_Limit"].replace("_", np.nan).astype(float)
    )

    # Replacing placeholders with 'NAN' in 'Occupation' and 'Credit_Mix' columns
    df["Occupation"] = df["Occupation"].replace("_______", "NAN")
    df["Credit_Mix"] = df["Credit_Mix"].replace("_", "NAN")

    # Creating binary features for each loan type
    for loan_type in loan_types:
        df[loan_type] = df["Type_of_Loan"].apply(
            lambda x: 1 if loan_type in str(x) else 0
        )
    # Droping 'Type_of_Loan' column
    df = df.drop(columns=["Type_of_Loan"])

    # Extracting year from 'Credit_History_Age'
    df["Credit_History_in_Year"] = (
        df["Credit_History_Age"].str.extract(r"(\d+) Years").astype(float)
    )
    # Droping 'Credit_History_Age' column
    df = df.drop(columns=["Credit_History_Age"])

    return df
