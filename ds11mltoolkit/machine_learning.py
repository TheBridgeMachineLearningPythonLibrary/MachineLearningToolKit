from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import pandas as pd
import numpy as np

from typing import List, Union
from selenium.webdriver.common.by import By
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
import time
import io
from PIL import Image
import pickle
import zipfile


def balance_binary_target(df, strategy='smote', minority_ratio=None, visualize=False):
    """
    This function balances a target binary variable of a dataframe using different oversampling strategies.
    Args:
    - df: dataframe with the target variable to balance.
    - strategy: oversampling strategy to use (default='smote'). The options are: 'smote', 'adasyn' or 'random'.
    - minority_ratio: proportion of the minority class after oversampling (default=None).
    - visualize: if True, visualize the balanced data (default=False).
    Returns:
    - DataFrame: dataframe with the balanced target variable.
    """

    # Automatically detect the target variable column.
    target_col = df.select_dtypes(include=['bool', 'int', 'float']).columns[0]

    # Separate target variable and predictor variables
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Oversampling the minority class using the selected strategy
    if strategy == 'smote':
        sampler = SMOTE(random_state=42)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif strategy == 'random':
        sampler = RandomOverSampler(random_state=42)
    else:
        raise ValueError("Estrategia de sobremuestreo inválida. Las opciones son: 'smote', 'adasyn' o 'random'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Adjust the minority class ratio if specified
    if minority_ratio is not None:
        target_counts = y_resampled.value_counts()
        minority_class = target_counts.idxmin()
        majority_class = target_counts.idxmax()

        minority_count = target_counts[minority_class]
        majority_count = target_counts[majority_class]

        desired_minority_count = int(minority_ratio * (minority_count + majority_count))

        if desired_minority_count < minority_count:
            drop_indices = y_resampled[y_resampled == minority_class].index[:minority_count - desired_minority_count]
            X_resampled = X_resampled.drop(drop_indices)
            y_resampled = y_resampled.drop(drop_indices)
        elif desired_minority_count > minority_count:
            extra_count = desired_minority_count - minority_count
            extra_X, extra_y = sampler.fit_resample(X_resampled[y_resampled == minority_class], y_resampled[y_resampled == minority_class])
            X_resampled = pd.concat([X_resampled, extra_X], axis=0)
            y_resampled = pd.concat([y_resampled, extra_y], axis=0)

    # Display the balanced data if specified
    if visualize:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Distribución de la variable objetivo balanceada')
        y_resampled.value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel(target_col)
        ax.set_ylabel('Frecuencia')

    # Combine the predictor variables and the balanced target variable in a new dataframe.
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    return df_resampled


def ignore_columns_polyfeatures(X: pd.DataFrame, variables_to_ignore: List[str], n: int) -> pd.DataFrame:
    '''
    This function takes a dataframe as input and will create n polynomial features for all columns except those specified to ignore
    It is intended to be used to ignore binary columns for example and to be included in a Pipeline

    Parameters
    ----------
    X : dataframe
        It is the dataset we want to selectively create polynomial features
    variables_to_ignore : List[str]
        a list of column names to ignore in the polynomial feature creation
    n : int
        the degree for the polynomial fearture creation

    
    Return
    ----------
        df : Dataframe with the changes made
    '''
    X_poly_features = X.drop(columns = variables_to_ignore)

    X_ignore = X[variables_to_ignore].reset_index(drop = True)

    poly = PolynomialFeatures(degree = n)

    poly_array = poly.fit_transform(X_poly_features)

    poly_features_names = poly.get_feature_names_out(X_poly_features.columns)

    X_poly_features = pd.DataFrame(poly_array, columns = poly_features_names)

    return pd.concat([X_ignore, X_poly_features], axis = 1)


def create_multiclass_prediction_df(model, class_names: List[str], X_test: Union[pd.DataFrame, np.ndarray], y_test: Union[list, np.ndarray], only_wrong: bool = False) -> pd.DataFrame:
    '''
    This will generate a dataframe from the predictions of a model on X_test for easy analysis of model performance

    Parameters
    ----------
    model : keras model
        It is the dataset we want to selectively create polynomial features
    class_names : List[str]
        a list of the class names which must be in order that relates to the numbers in y_test
    X_test : pd.DataFrame or np.ndarray
        The data for X_test
    y_test : list or np.ndarray
        The y_test with the numeric values of the class names
    only_wrong : bool
        defaults to False but if set to True, only incorrect predictions will be kept
    
    Return
    ----------
        df : Dataframe containing the predictions from model.predict along with the top prediction and actual labels
    
    '''
    # make the predictions with model on X_test
    model_predictions_df = round(pd.DataFrame(model.predict(X_test)), 2)
    # assign class label names as columns
    model_predictions_df.columns = class_names
    # create top prediction column
    model_predictions_df['Top Prediction'] = model_predictions_df.apply(lambda x: x.idxmax(), axis = 1)
    # create class label column
    model_predictions_df['Label'] = [model_predictions_df.columns[y_class] for y_class in y_test]
    # filter for model errors if "only_wrong == True"
    if only_wrong:
        model_predictions_df = model_predictions_df[model_predictions_df['Top Prediction'] != model_predictions_df['Label']]
    # return the dataframe
    return model_predictions_df

def quickregression(name):
    
    """
    Function to save time when doing Machine Learning models. 
    It only asks the name of the model to train and returns the scoring.

    Parameters
    ----------
    name = Name of the ML model.
           Input Example = LinearRegression
    
    Returns
    ----------
    MAE, MAPE, MSE, RMSE and R2 Scores.
    """

    # Fit of the model in the previously split X_train, y_train
    model = name()
    model.fit(X_train, y_train)
    # Predict of the model with X_test
    modpred = model.predict(X_test)
    # Scores of the model with y_test and the predict values.
    print("MAE test:", mean_absolute_error(y_test, modpred))
    print("MAPE test:", mean_absolute_percentage_error(y_test, modpred))
    print("MSE test:", mean_squared_error(y_test, modpred))
    print("RMSE test:", np.sqrt(mean_squared_error(y_test, modpred)))
    return(model.score(X_train, y_train))


def load_model_zip(zip_file, model_file):

    """
    Uploads a model file from a zip file.

    Parameters
    ----------
         zip_file: The name of the zip file where the model file is located.
         model_file: The name of the model file to load.

    Returns:
    ----------
         The model loaded from the file.
    """
    # Abre el archivo zip en modo lectura
    with zipfile.ZipFile(zip_file, "r") as zip:
        # Lee el archivo de modelo del zip y lo carga en la memoria
        with zip.open(model_file, "r") as file:
            model = pickle.load(file)

    return model

def image_scrap(url, n:int):
	'''
	Function to scrap chrome images and get n images we want, and it create a new folder as 'my_images'.

	As we know, we are using selenium, we will need a driver in Chrome.
	Must have driver from Chrome to run it [chrome](https://chromedriver.chromium.org/), file name = 'chromedriver' and dowload in the same path as the scrip or jupyter. 

	Parameters
	----------
	url -> chrome images web link, must be all way long.

	n -> number of images you want to have in the folder. Must be 'int'
	
	Return
	----------

	Folder called 'my_images' with n images, where you can show as much time as you want
	
	'''
	current_dir = os.getcwd()
	driver_path = os.path.join(current_dir, "chromedriver.exe")

	wd = webdriver.Chrome(driver_path)

	def get_images_from_google(url, wd, delay, max_images):
		def scroll_down(wd):
			wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
			time.sleep(delay)

		url = url
		wd.get(url)

		loadMore = wd.find_element(By.XPATH, '/html/body/c-wiz/div/div/div/div[2]/div/div[3]/div/div/form/div/div/button').click()

		image_urls = set()
		skips = 0

		while len(image_urls) + skips < max_images:
			scroll_down(wd)

			thumbnails = wd.find_elements(By.CLASS_NAME, "Q4LuWd")

			for img in thumbnails[len(image_urls) + skips:max_images]:
				try:
					img.click()
					time.sleep(delay)
				except:
					continue

				images = wd.find_elements(By.CLASS_NAME, "n3VNCb")
				for image in images:
					if image.get_attribute('src') in image_urls:
						max_images += 1
						skips += 1
						break

					if image.get_attribute('src') and 'http' in image.get_attribute('src'):
						image_urls.add(image.get_attribute('src'))
						print(f"Found {len(image_urls)}")

		return image_urls


	def download_image(download_path, url, file_name):
		try:
			image_content = requests.get(url).content
			image_file = io.BytesIO(image_content)
			image = Image.open(image_file)
			file_path = download_path + file_name

			with open(file_path, "wb") as f:
				image.save(f, "JPEG")

			print("Success")
		except Exception as e:
			print('FAILED -', e)


	urls = get_images_from_google(url,wd, 1, n)
	
	
	current_dir = os.path.dirname(os.path.abspath(__file__))

	download_dir = os.path.join(current_dir, "my_images")
	

	if not os.path.exists(download_dir):
		os.makedirs(download_dir)

	for i, url in enumerate(urls):
			download_image(download_dir, url, str(i) + ".jpg")

	wd.quit()


def worst_params(gridsearch):
    '''
    Function to obtain the worst params of a gridsearch. In case we need to train a gridsearch multiple times,
    it can be useful to know which parameters are likely to be deleted, in order to make our training faster.

    Args:
    gridsearch: trained gridsearch

    '''
    position = list(gridsearch['rank_test_score']).index(gridsearch['rank_test_score'].max())
    worst_params = gridsearch['params'][position]
    worst_scoring = gridsearch['mean_test_score'][position]

    return str(worst_params), worst_scoring


# Function to calculate a prediction score
def show_scoring(y, y_prediction, label:str, round:int=3, auc_sc:bool=True, roc_auc_sc:bool=True, confusion_matrix_sc:bool=True):
    '''
    Function to calculate a prediction score

    Parameters
    ----------
        y: target values.
        
        y_prediction: prediction values.

        label: value type label.

        round: value rounding.

        auc_sc: compute Area Under the Curve (AUC) using the trapezoidal rule.

        roc_auc_sc: compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

        confusion_matrix_sc: compute confusion matrix to evaluate the accuracy of a classification.

    Return
    ------
        dict(accu --> accuracy_score, 
            auc_s --> roc_curve.auc, 
            roc_auc --> roc_auc_score, 
            conf_mat --> confusion_matrix)
    '''
    try:
        auc_r = None
        roc_auc_r = None
        conf_mat_r = None

        print('-'*30)

        accu_r = accuracy_score(y, y_prediction).round(round)
        print('ACCURACY',label,':',accu_r)

        if auc_sc:
            fpr, tpr, thresh = roc_curve(y, y_prediction)
            auc_r = auc(fpr, tpr).round(round)
            print('AUC',label,':',auc_r)

        if roc_auc_sc:
            roc_auc_r = roc_auc_score(y, y_prediction).round(round)
            print('ROC AUC',label,':',roc_auc_r)
        
        if confusion_matrix_sc:
            c_mat = confusion_matrix(y, y_prediction, normalize='true')
            conf_mat_r = c_mat.round(round)
            print('CONFUSION MATRIX',label,':',conf_mat_r)

        print('-'*30)

        return dict(accu_r=accu_r, auc_r=auc_r, roc_auc_r=roc_auc_r, conf_mat_r=conf_mat_r)

    except SyntaxError:
        print('Fix your syntax')

    except TypeError:
        print('Oh no! A TypeError has occured')
        
    except ValueError:
        print('A ValueError occured!')

    except OSError as err:
        print('OS error:', err)

    except Exception as err:
        print(f'Unexpected {err}, {type(err)}')
    
    except: 
        print('Something went wrong')


# Function to train and predict the model with a classification algorithm
def processing_model_classification(model:object, x, y, test_size_split:float=0.25, shuffle_split:bool=False, random_state_split:int=None, minMaxScaler:bool=False, minMaxScaler_range:tuple=(0,1), standardScaler:bool=False, train_score:bool=False, test_score:bool=True):
    '''
    Function to train and predict the model with a classification algorithm

    Parameters
    ----------
        model: algorithm / model.
        x: {array-like, sparse matrix} of shape (n_samples, n_features).
            training data.

        y: {array-like, sparse matrix} of shape (n_samples,).
            target values.

        test_size_split: if float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
            If int, represents the absolute number of test samples. If None, it will be set to 0.25.

        shuffle_split: whether or not to shuffle the data before splitting.

        minMaxScaler: whether or not to transform features by scaling each feature to a given range.

        minMaxScaler_range: if minMaxScaler is True, desired range of transformed data.

        standardScaler: whether or not to standardize features by removing the mean and scaling to unit variance.

        train_score: compute the score of train.

        test_score: compute the score of test.

    Return
    ------
        model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    '''
    try:
        # Split Train and Test
        if random_state_split == None:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size_split, shuffle=shuffle_split)
        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size_split, shuffle=shuffle_split, random_state=random_state_split)

        # Number of rows to be processed
        print('-'*30)
        print('Rows to process for Train:', len(X_train))
        print('Rows to process for Test:', len(X_test))
        print('-'*30)

        # Data scaling - MinMaxScaler()
        if minMaxScaler:
            X_train = MinMaxScaler(feature_range=minMaxScaler_range).fit_transform(X_train)
            X_test = MinMaxScaler(feature_range=minMaxScaler_range).fit_transform(X_test)

        # Data scaling - StandardScaler()
        if standardScaler:
            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)

        # Training the model
        model.fit(X_train, y_train)

        # Model prediction with Train
        y_pred_train = model.predict(X_train)

        # Model prediction with Test
        y_pred_test = model.predict(X_test)

        # Return the mean accuracy on the given train data and labels
        print('SCORE TRAIN:',model.score(X_train, y_train).round(3))

        # Compute the score of Train
        if train_score:
            show_scoring(y_train, y_pred_train, 'TRAIN', 3)

        # Compute the score of Test
        if test_score:
            show_scoring(y_test, y_pred_test, 'TEST', 3)

        return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test # return model, X_test, y_test, y_pred_test
    
    except SyntaxError:
        print('Fix your syntax')

    except TypeError:
        print('Oh no! A TypeError has occured')
        
    except ValueError:
        print('A ValueError occured!')

    except OSError as err:
        print('OS error:', err)

    except Exception as err:
        print(f'Unexpected {err}, {type(err)}')
    
    except: 
        print('Something went wrong')


# Function to predict the model with a classification algorithm
def predict_model_classification(model:object, X_test, y_test, minMaxScaler:bool=False, minMaxScaler_range:tuple=(0,1), standardScaler:bool=False, test_score:bool=True):
    '''
    Function to predict the model with a classification algorithm

    Parameters
    ----------
        model: algorithm / model.

        X_test: {array-like, sparse matrix} of shape (n_samples, n_features).
            training data.

        y_test: {array-like, sparse matrix} of shape (n_samples,).
            target values.

        minMaxScaler: whether or not to transform features by scaling each feature to a given range.

        minMaxScaler_range: if minMaxScaler is True, desired range of transformed data.

        standardScaler: whether or not to standardize features by removing the mean and scaling to unit variance.

        test_score: calculates the score of test
        
    Return
    ------
        X_test, y_pred_test
    '''
    try:
        # Data scaling - MinMaxScaler()
        if minMaxScaler:
            X_test = MinMaxScaler(feature_range=minMaxScaler_range).fit_transform(X_test)

        # Data scaling - StandardScaler()
        if standardScaler:
            X_test = StandardScaler().fit_transform(X_test)

        # Model prediction with Test
        y_pred_test = model.predict(X_test)

        # Compute the score of Test
        if test_score:
            show_scoring(y_test, y_pred_test, 'TEST', 3)

        return X_test, y_pred_test
    
    except SyntaxError:
        print('Fix your syntax')

    except TypeError:
        print('Oh no! A TypeError has occured')
        
    except ValueError:
        print('A ValueError occured!')

    except OSError as err:
        print('OS error:', err)

    except Exception as err:
        print(f'Unexpected {err}, {type(err)}')
    
    except: 
        print('Something went wrong')


# Function to export a model
def export_model(model:object, dir_model:str, name_model:str, timestamp:bool=False):
    '''
    Function to export a model

    Parameters
    ----------
        model: algorithm / model we want to save.

        dir_model: directory to save the model.

        name_model: name of the model to save.
        
        timestamp: time stamp to rename the model.
    '''
    try:
        # Format the current date and time for the renaming of the file to be exported
        if timestamp:
            now = datetime.now()
            year = now.strftime('%Y')[2:]
            timestamp = '_' + year + now.strftime('%m%d%H%M%S')
        else:
            timestamp = ''

        # Export the model with the renamed model to the specified directory
        filename = os.path.join(dir_model, name_model + timestamp)
        with open(filename, 'wb') as archivo:
            pickle.dump(model, archivo)

        # Show info
        print('-'*46)
        print('Model saved at', dir_model)
        print('-'*46)

    except SyntaxError:
        print('Fix your syntax')

    except TypeError:
        print('Oh no! A TypeError has occured')
        
    except ValueError:
        print('A ValueError occured!')

    except OSError as err:
        print('OS error:', err)

    except Exception as err:
        print(f'Unexpected {err}, {type(err)}')
    
    except: 
        print('Something went wrong')


# Function to import a model
def import_model(dir_model:str, name_model:str):
    '''
    Function to import a model

    Parameters
    ----------
        dir_model: directory to import the model.
        
        name_model: name of the model to be imported.
    
    Return
    ------
        model_import: algorithm / model we have saved.
    '''
    try:
        # Import the model
        filename = os.path.join(dir_model, name_model)
        with open(filename, 'rb') as archivo:
            model_import = pickle.load(archivo)

        return model_import

    except SyntaxError:
        print('Fix your syntax')

    except TypeError:
        print('Oh no! A TypeError has occured')
        
    except ValueError:
        print('A ValueError occured!')

    except OSError as err:
        print('OS error:', err)

    except Exception as err:
        print(f'Unexpected {err}, {type(err)}')
    
    except: 
        print('Something went wrong')


