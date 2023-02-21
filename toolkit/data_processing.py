def extract_date(df, date_column_name):
    '''
    Function to extract the data of a date, from a column that does not have date format.
    The function also creates a column for date and year.

    Parameters
    ----------
    df : dataframe
        It is the dataset where the column that we have to extract the data is located.
    date_column_name : str
        is the column that has data in string. From here the data is extracted and converted to date format.

    
    Return
    ----------
        df : Dataframe with the changes made

    '''
    import re
    from datetime import datetime

    match = re.search(r'\d{4}.\d{2}.\d{2}', date_column_name)
    date = datetime.strptime(match.group(), '%Y-%m-%d').date()

    df['Year'] = df[date_column_name].dt.year
    df['Month'] = df[date_column_name].dt.month
    
    return df

def list_categorical_columns(df):
    '''
    Function that returns a list with the names of the categorical columns of a dataframe.

    Parameters
    ----------
    df : dataframe
    
    Return
    ----------
        features: list of names

    '''
    features = []

    for c in df.columns:
        t = str(df[c].dtype)
        if "object"  in t:
            features.append(c)
    return features



def uniq_value(list_values:list):
    '''
    Function returning the unique values from a list.
    Parameters
    ----------
    list_values:list
    Return
    ----------
    unique: list of unique values
    '''
    unique = []
    for i in list_values:
        if i not in unique:
            unique.extend(list_values)
    return unique