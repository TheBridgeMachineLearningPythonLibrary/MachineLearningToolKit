import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import cv2 
import os
from skimage.io import imread

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

def last_columndf(df,feature):
    '''
    Function will return the dataframe with the column entered at the last position
    ----------
    df: dataframe
    feature: to move at the last position

    Return
    ----------
    df: dataframe
    '''

    lista=list(df.columns.values)
    lista.remove(feature)
    lista.append(feature)
    df=df[lista]
    return df



def log_transform_data(df: pd.DataFrame, col_ignore: List[str]) -> pd.DataFrame:
    '''
    Log transform the numeric columns and recombine back with the non-numeric
    Option to skip specific numeric columns

    Parameters
    ----------
    df : pd.DataFrame
        The dataset we want to carry out log transform on
    col_ignore : List[str]
        A list of numeric column names that we want to skip the log transform for

    
    Return
    ----------
    pd.DataFrame with same dimensions as input
    '''
    df_ignore = df[col_ignore]
    df_rest = df.drop(columns = col_ignore)
    numeric = df_rest.select_dtypes(include=np.number).apply(np.log1p)
    non_numeric = df_rest.select_dtypes(exclude=np.number)
    
    return pd.concat([numeric, non_numeric, df_ignore], axis = 1)



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


def clean_text(df, column:str, language:str, target:str, filename:str='data_processed.csv'):
    
    ''' 
    Function to preprocess and clean a dataframe with text as a preliminary step for Natural Language Processing

    Params:
    - df: Dataframe
    - column: The name of the column in which the text is located (str)
    - language: The language in which the text is written (str) in ENGLISH (e.g. 'spanish', 'english')
    - target: The name of the column in which the target to be predicted is located
    - filename: Name for the processed dataframe to be saved

    Returns:
    - df_processed: Dataframe after cleaning. It contains only the text variable and the target variable
    '''

    # Remove duplicated
    df.drop_duplicates(subset = column, inplace=True)

    # Remove mentions (@)
    df[column] = df[column].str.replace(r'\s*@\w+', '', regex=True)

    # Remove punctuation marks and convert to lowercase
    signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")

    def signs_tweets(tweet):
        return signos.sub('', tweet.lower())

    df[column] = df[column].apply(signs_tweets)

    # Remove links
    def remove_links(df):
        return " ".join(['{link}' if ('http') in word else word for word in df.split()])
    
    df[column] = df[column].apply(remove_links)

    # Remove stopwords
    def remove_stopwords(df):
        stopwords = set(stopwords.words(language))
        return " ".join([word for word in df.split() if word not in stopwords])
    df[column] = df[column].apply(remove_stopwords)

    # Apply Stemmer
    stemmer = SnowballStemmer(language)

    def def_stemmer(x):
        return " ".join([stemmer.stem(word) for word in x.split()])

    df[column] = df[column].apply(def_stemmer)

    # Save processed data
    df_processed = df[[column, target]]
    df_processed.to_csv(filename)

    return df_processed



def load_imgs(path, im_size:int):
    
    ''' 
    Function to load a directory of images and resize them for training a Convolutional Neural Network (CNN) model.
    IMPORTANT: Images must be divided into subdirectories according to the target 
    (e.g. one directory for dog photos and another for cat photos).
    It can be used for both binary and categorical classification.

    Args:
    - path: Path where the subdirectories with the images are located.
    - im_size: Size to which we want to resize the image (e.g. 32).

    Returns:
    - df: Dataframe with the names of the images and the category to which they belong (target).
    - X_train: Array with the image data loaded after resizing.
    - y_train: Array with the target values.
    '''
    

    filenames = []
    X = []
    y = []

    # Create a dictionary with the target values.
    class_names = os.listdir(path)
    class_names_label = {class_name:i for i , class_name in enumerate(class_names)}

    # Iterate over the subdirectories of the given path.
    for subdir, dirs, files in os.walk(path):
            for file in files:
                # Save the names of the files in a list.
                filenames.append(file)
                if file [-4:] == '.jpg' or file [-4:] == '.png':
                    # Read the image in color.
                    image = imread(subdir + '\\' + file)
                    # Resize the image.
                    smallimage = cv2.resize(image, (im_size, im_size)) 
                    # Save the images in the X variable.
                    X.append(smallimage)
            
                    # Save the target values of each image in a list.
                    for i in range(len(class_names_label.keys())):
                        if list(class_names_label.keys())[i] in subdir:
                            y.append(list(class_names_label.values())[i])
    
    # Returns:

    # 1. Dataframe with the names of the images and the category to which they belong.
    df = pd.DataFrame({
    'filename': filenames,
    'category': y
    }) 

    # 2. Array with the image data loaded after resizing.
    X_train = np.array(X)

    # 3. Array with the y values (target).
    y_train = np.array(y)
    
    return df, X_train, y_train
