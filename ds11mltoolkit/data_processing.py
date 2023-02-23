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
import shutil
from keras.preprocessing.image import (
    ImageDataGenerator,
    DataFrameIterator,
    DirectoryIterator
)

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

    Parameters
    ----------
    - df: Dataframe
    - stats: Descriptive statistics to calculate. Default: Mean, Median, and Standard Deviation

    Returns
    ----------
    Dataframe with the following columns:
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
    agg = df.agg(stats).transpose()

    # Merge in a new dataframe
    describe = pd.merge(concatenado, agg, left_index=True, right_index=True, how='left')

    return describe

def clean_text(df, column:str, language:str, target:str, filename:str='data_processed.csv'):
    
    ''' 
    Function to preprocess and clean a dataframe with text as a preliminary step for Natural Language Processing

    Parameters
    ----------
    - df: Dataframe
    - column: The name of the column in which the text is located (str)
    - language: The language in which the text is written (str) in ENGLISH (e.g. 'spanish', 'english')
    - target: The name of the column in which the target to be predicted is located
    - filename: Name for the processed dataframe to be saved

    Returns
    ----------
    - df_processed: Dataframe after cleaning. It contains only the text variable and the target variable
    '''

    # Remove duplicated
    df.drop_duplicates(subset = column, inplace=True)

    # Remove mentions (@)
    df[column] = df[column].str.replace(r'\s*@\w+', '', regex=True)

    # Remove punctuation marks and convert to lowercase
    signos = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")

    def signs_tweets(tweet):
        return signos.sub('', tweet.lower())

    df[column] = df[column].apply(signs_tweets)

    # Remove links
    def remove_links(df):
        return " ".join(['{link}' if ('http') in word else word for word in df.split()])
    
    df[column] = df[column].apply(remove_links)

    # Remove stopwords
    stopwords = set(stopwords.words(language))
    def remove_stopwords(df):
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

    Parameters
    ----------
    - path: Path where the subdirectories with the images are located.
    - im_size: Size to which we want to resize the image (e.g. 32).

    Returns
    ----------
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

def convert_to_numeric(df,column:str):
    '''
    This function convert any number string in that column, to int or float ignoring any NaN value.

    df -> dataframe we are working with

    column -> column which we want to convert to numeric. Must be 'str'

    Return:

    Dataframe with columns already changed

    '''
    df[column] = df[column].apply(lambda x: pd.to_numeric(x, errors = 'coerce'))
    return df

def _exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """

    smoothed_data = data.ewm(alpha=alpha).mean()

    # Check that the first and last values of the smoothed data are the same as the original data
    smoothed_data.iloc[0] = data.iloc[0]
    smoothed_data.iloc[-1] = data.iloc[-1]

    return smoothed_data    


def add_previous(df, n, clas, values):
    """
    Add columns to the dataframe with the values of the last n events for each class.
    
    Parameters:
    df (DataFrame): The input dataframe.
    n (int): The number of previous events to include in the output dataframe.
    clas (str): name of the column you want to obtain the previous values of.
    values (str): name of the column whose previous values you need
    
    Returns:
    None
    """
    # Group the dataframe by the clas column
    grouped = df.groupby(clas)
    
    # Initialize a list to store the shifted values
    shifted_values = []
    
    # Shift the values within each group n times to get the previous values of your clas
    for i in range(1, n + 1):
        shifted_values.append(grouped[values].shift(i))
    
    # Concatenate the shifted values with the original dataframe
    new_cols = [f'Previous_value-{i}' for i in range(1, n + 1)]
    for i, col in enumerate(new_cols):
        df[col] = shifted_values[i]

    return df

def winner_loser(x, df, column):

    """
    Comparator of odd and even rows, checks which one is a bigger value and returns Victory, Loss or Draw
    according to that. Prepared for sports, but appliable to other uses.

    Args:
    x (int): number of the index
    df (df): dataframe to work in
    column (str): name of the column we want to compare

    Return: Victory, Draw or Loss
    
    """
    if (x+2) % 2 == 0:                                  
        if df[column][x] > df[column][x+1]:     
            x = 'Victory'                               
            return x                                    
        elif df[column][x] < df[column][x+1]:   
            x = 'Loss'
            return x
        else:
            x = 'Draw'
            return x
    if (x+2) % 2 != 0:                                 
        if df[column][x] > df[column][x-1]:
            x = 'Victory'
            return x
        elif df[column][x] < df[column][x-1]:
            x = 'Loss'
            return x
        else:
            x = 'Draw'
            return x

class ImageDataGen(ImageDataGenerator):
    def gen_from_array(
        self,
        x,
        y=None,
        max_image_count=None,
        batch_size=32,
        shuffle=True,
        sample_weight=None,
        seed=None,
        save_to_dir='./aug',
        save_prefix='',
        save_format='png',
        ignore_class_split=False,
        subset=None
    ):
        '''Takes data & label arrays, generates new data & label arrays of 
        augmented data by applying random transformations to existing arrays.

        Args:
            x (numpy.ndarray or tuple): Input data. A numpy array of rank 4 or 
                a tuple. If a tuple, the first element should contain the 
                images and the second element another numpy array or a list of 
                numpy arrays that are passed to the output without any 
                modifications. Can be used to feed the model miscellaneous 
                data along with the images. In case of grayscale data, the 
                channels axis of the image array should have value 1, in case 
                of RGB data, it should have value 3, and in case of RGBA data, 
                it should have value 4.

            y (numpy.ndarray, optional): Labels. Defaults to None.

            max_image_count (int, optional): Maximum number of augmented 
                images to generate for each class. If None, generate enough to 
                equal the size of the largest class. Defaults to None.

            batch_size (int, optional): The size of each batch of augmented images. Defaults to 32.

            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

            sample_weight (numpy.ndarray, optional): Sample weights. Defaults to None.

            seed (int, optional): Random seed. Defaults to None.

            save_to_dir (str, optional): If set, this allows you to specify a 
                directory to which to save the augmented pictures being 
                generated (useful for visualizing what you are doing). 
                Defaults to './aug'.

            save_prefix (str, optional): Prefix to use for filenames of saved 
                pictures (only relevant if `save_to_dir` is set). Defaults to ''.

            save_format (str, optional): The format to use for saved pictures, 
                e.g. "png", "jpeg", "bmp", "pdf", "ppm", "gif", "tif", "jpg". 
                Only relevant if `save_to_dir` is set. Defaults to 'png'.

            ignore_class_split (bool, optional): Whether to ignore the 
                difference in number of classes in labels across train and 
                validation split (useful for non-classification tasks). 
                Defaults to False.

            subset (str, optional): The subset of data to use, either 
                `"training"` or `"validation"`, if `validation_split` is set 
                in `ImageDataGenerator`. Defaults to None.

        Returns:
            Tuple of Numpy arrays: `(X_gen, y_gen)` A tuple of two numpy 
                arrays, containing the augmented data and labels, respectively.

            `X_gen`: The generated augmented images.
            `y_gen`: The corresponding labels for the generated images.
        '''
        if y is None:
            y = np.zeros((x.shape[0],))

        unique, counts = np.unique(y, return_counts=True)
        class_sizes = dict(zip(unique, counts))

        if max_image_count is None:
            max_image_count = max(counts)

        X_gen = []
        y_gen = []

        for cls in class_sizes:
            if class_sizes[cls] >= max_image_count: continue

            samples_needed = max_image_count - class_sizes[cls]
            images_generated = 0

            generator = super().flow(
                x=x[y == cls],
                y=y[y == cls],
                batch_size=batch_size,
                shuffle=shuffle,
                sample_weight=sample_weight,
                seed=seed,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format,
                ignore_class_split=ignore_class_split,
                subset=subset
            )

            while images_generated < samples_needed:
                batch = next(generator)

                X_gen.extend(batch[0])
                y_gen.extend(batch[1])

                images_generated += len(batch[1])

        X_gen = np.array(X_gen)
        y_gen = np.array(y_gen)

        return (X_gen, y_gen)
        
    def gen_from_dataframe(
        self,
        dataframe,
        max_image_count=None,
        directory=None,
        x_col='filename',
        y_col='class',
        weight_col=None,
        target_size=(256, 256),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir='./aug',
        save_prefix='',
        save_format='png',
        subset=None,
        interpolation='nearest',
        validate_filenames=True,
        **kwargs
    ) -> DataFrameIterator:
        '''Generates batches of augmented/normalized data from a dataframe and 
        stores them in their respective folders.

        Args:
            dataframe: Pandas dataframe containing the filepaths relative to
                `directory` (or absolute paths if `directory` is None) of the
                images in a string column. It should include other column/s
                depending on the `class_mode`:
                - if `class_mode` is `"categorical"` (default value) it must
                    include the `y_col` column with the class/es of each image.
                    Values in column can be string/list/tuple if a single class
                    or list/tuple if multiple classes.
                - if `class_mode` is `"binary"` or `"sparse"` it must include
                    the given `y_col` column with class values as strings.
                - if `class_mode` is `"raw"` or `"multi_output"` it should
                    contain the columns specified in `y_col`.
                - if `class_mode` is `"input"` or `None` no extra column is needed.

            max_image_count (int, optional): Maximum number of augmented 
                images to generate for each class. If None, generate enough to 
                equal the size of the largest class. Defaults to None.

            directory: string, path to the directory to read images from. If
                `None`, data in `x_col` column should be absolute paths.

            x_col: string, column in `dataframe` that contains the filenames (or
                absolute paths if `directory` is `None`).

            y_col: string or list, column/s in `dataframe` that has the target data.

            weight_col: string, column in `dataframe` that contains the sample
                weights. Default: `None`.

            target_size: tuple of integers `(height, width)`, default: `(256,
                256)`. The dimensions to which all images found will be resized.

            color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to have 1 or 3 color
                channels.

            classes: optional list of classes (e.g. `['dogs', 'cats']`). Default
                is None. If not provided, the list of classes will be
                automatically inferred from the `y_col`, which will map to the
                label indices, will be alphanumeric). The dictionary containing
                the mapping from class names to class indices can be obtained via
                the attribute `class_indices`.

            class_mode: one of "binary", "categorical", "input", "multi_output",
                "raw", sparse" or None. Default: "categorical".
                Mode for yielding the targets:
                - `"binary"`: 1D numpy array of binary labels,
                - `"categorical"`: 2D numpy array of one-hot encoded labels.
                    Supports multi-label output.
                - `"input"`: images identical to input images (mainly used to
                    work with autoencoders),
                - `"multi_output"`: list with the values of the different
                    columns,
                - `"raw"`: numpy array of values in `y_col` column(s),
                - `"sparse"`: 1D numpy array of integer labels,
                - `None`, no targets are returned (the generator will only yield
                    batches of image data, which is useful to use in
                    `model.predict()`).

            batch_size: size of the batches of data (default: 32).

            shuffle: whether to shuffle the data (default: True)

            seed: optional random seed for shuffling and transformations.

            save_to_dir: None or str (default: None). This allows you to
                optionally specify a directory to which to save the augmented
                pictures being generated (useful for visualizing what you are
                doing).

            save_prefix: str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).

            save_format: one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
                "tif", "jpg" (only relevant if `save_to_dir` is set). Default:
                "png".

            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.

            interpolation: Interpolation method used to resample the image if
                the target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed, `"box"` and
                `"hamming"` are also supported. By default, `"nearest"` is used.

            validate_filenames: Boolean, whether to validate image filenames in
                `x_col`. If `True`, invalid images will be ignored. Disabling this
                option can lead to speed-up in the execution of this function.
                Defaults to `True`.
              
            **kwargs: legacy arguments for raising deprecation warnings.

        Returns:
            A `DataFrameIterator` yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        '''
        generator = super().flow_from_dataframe(
            dataframe=dataframe,
            directory=directory,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
            **kwargs
        )

        groups = dataframe.groupby(y_col)
        class_count = pd.concat([
            groups.get_group(label) for label in dataframe[y_col].unique()
        ], axis=0).reset_index(drop=True)

        if os.path.isdir(save_to_dir):
            shutil.rmtree(save_to_dir)
        os.mkdir(save_to_dir)

        for label in class_count[y_col].unique():
            dir_path = os.path.join(save_to_dir, label)
            os.mkdir(dir_path)

            group = groups.get_group(label)
            count = len(group)

            if max_image_count is None or max_image_count < count:
                max_image_count = count

            samples_needed = max_image_count - count
            images_generated = 0

            generator.dataframe = group
            generator.save_to_dir = dir_path
            
            while images_generated < samples_needed:
                next(generator)
                images_generated += batch_size

        return generator

    def gen_from_directory(
        self,
        directory,
        max_image_count=None,
        target_size=(256, 256),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir='./aug',
        save_prefix='',
        save_format='png',
        follow_links=False,
        subset=None,
        interpolation='nearest',
        keep_aspect_ratio=False
    ) -> DirectoryIterator:
        '''Takes the path to a directory & generates batches of augmented data 
        and stores them in their respective folders.

        Args:
            directory: string, path to the target directory. It should contain
                one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree will be included
                in the generator.

            max_image_count (int, optional): Maximum number of augmented 
                images to generate for each class. If None, generate enough to 
                equal the size of the largest class. Defaults to None.

            target_size: Tuple of integers `(height, width)`, defaults to `(256,
                256)`. The dimensions to which all images found will be resized.

            color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to have 1, 3, or 4 channels.

            classes: Optional list of class subdirectories (e.g. `['dogs',
                'cats']`). Default: None. If not provided, the list of classes
                will be automatically inferred from the subdirectory
                names/structure under `directory`, where each subdirectory will be
                treated as a different class (and the order of the classes, which
                will map to the label indices, will be alphanumeric). The
                dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.

            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.

            batch_size: Size of the batches of data (default: 32).

            shuffle: Whether to shuffle the data (default: True) If set to
                False, sorts the data in alphanumeric order.

            seed: Optional random seed for shuffling and transformations.

            save_to_dir: None or str (default: None). This allows you to
                optionally specify a directory to which to save the augmented
                pictures being generated (useful for visualizing what you are
                doing).

            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).

            save_format: one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
                "tif", "jpg" (only relevant if `save_to_dir` is set). Default:
                "png".

            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).

            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.

            interpolation: Interpolation method used to resample the image if
                the target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed, `"box"` and
                `"hamming"` are also supported. By default, `"nearest"` is used.
                
            keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.

        Returns:
            A `DirectoryIterator` yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        '''
        generator = super().flow_from_directory(
            directory=directory,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            keep_aspect_ratio=keep_aspect_ratio
        )

        groups = generator.class_indices

        if os.path.isdir(save_to_dir):
            shutil.rmtree(save_to_dir)
        os.mkdir(save_to_dir)

        for label in groups:
            dir_path = os.path.join(save_to_dir, label)
            os.mkdir(dir_path)

            count = np.count_nonzero(generator.classes == groups[label])

            if max_image_count is None or max_image_count < count:
                max_image_count = count

            samples_needed = max_image_count - count
            images_generated = 0

            generator.save_to_dir = dir_path

            while images_generated < samples_needed:
                next(generator)
                images_generated += batch_size

        return generator