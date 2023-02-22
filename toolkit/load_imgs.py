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
    
    import numpy as np
    import cv2 
    import os
    from skimage.io import imread
    import pandas as pd

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
