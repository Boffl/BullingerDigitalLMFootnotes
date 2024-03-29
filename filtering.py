import pandas as pd

# functions to filter the dataframe
def iqr(series:pd.Series):
    """calculate IQR, return lower and upper limit"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return lower_limit, upper_limit


def filter_df(df:pd.DataFrame, column, by=None):
    """take out anything above 1.5*IQR
    if 'by' is specified, IQR is calculated based on groups"""

    if by:
        groups = df[by].unique()
        
        for group in groups:
            # calculate upper limit for the column, considering the goup
            _, upper_limit = iqr(df[df[by]==group][column])
            # delete values that are in the group and above the upper limit
            condition1 = (df[by] == group) 
            condition2 = (df[column] > upper_limit)
            df = df[~(condition1 & condition2)]      
        df.reset_index()
        return df


    else:  # iqr not based on the groups
        _, upper_limit = iqr(df[column])
        print(f"removing all of {column} higher than: {upper_limit}")
        return df[df[column] <= upper_limit]