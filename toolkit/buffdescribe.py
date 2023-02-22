def buffdescribe(df,  stats=['mean', 'median', 'std']):

    '''
    Function to facilitate a first exploration of a dataframe's data by concentrating the most relevant information

    Params:
    - df: Dataframe
    - stats: Descriptive statistics to calculate. Default: Mean, Median, and Standard Deviation

    Returns: Dataframe with the following columns:
    - Column names from the original df
    - Data type of each column
    - Percentage of null values in each column
    - Total number of non-null values in each column
    - Unique values of each column
    - Percentage of unique values (cardinality)
    - Selected descriptive statistics (default: mean, median, std) of numeric variables (int or float)
    '''

    import pandas as pd

    # Column names
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Data type
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Percentage of Null Values
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Total number of non-null values 
    total_not_null = pd.DataFrame(df.count().values, columns=["NOT_NULL"])

    # Unique values
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    # Percentage of unique values (cardinality)
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, total_not_null, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)

    # Sort values by missing percentage
    describe = concatenado.sort_values(by=["MISSINGS (%)"], ascending=True)

    # Descriptive statistics
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    numeric_stats = df[numeric_cols].agg(stats)
    numeric_stats.columns = [col + '_' + stat for col in numeric_stats.columns for stat in stats]

    # Merge in a new dataframe
    describe = pd.merge(concatenado, numeric_stats, left_index=True, right_index=True, how='left')
    describe.rename(columns={'mean': 'MEAN',
                   'median': 'MEDIAN', 'std': 'STD'}, inplace=True),

    return describe