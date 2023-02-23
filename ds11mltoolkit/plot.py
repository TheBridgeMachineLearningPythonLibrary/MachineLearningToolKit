import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Union
import plotly.graph_objs as go
from plotly import subplots
from collections import defaultdict
import plotly.offline as py
from wordcloud import STOPWORDS, WordCloud
import plotly.express as px
from sklearn.metrics import auc, roc_curve
import folium


def plot_multiclass_prediction_image(df, row_index: int, X_test: Union[pd.DataFrame, np.ndarray], prediction_col: str = 'Top Prediction', label_col: str = 'Label'):
    '''
    This will produce two plots side by side relating to a prediction of an image
    The first is the image titled with the label and the predicted label
    The second is a bar plot showing the actual probabilities of predictions for all classes
    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the results of the prediction along with the label and top predicted label
    row_index : int
        the index value for the image we want to plot
    X_test : pd.DataFrame or np.ndarray
        The X_test data
    prediction_col : str
        Defaults to 'Top Prediction' but should be the name of column with the prediction for image
    label_col : str
        Defaults to 'Label' but should be the name of column with the real label
    
    Return
    ----------
    displays plots
    '''

    to_plot = df.loc[[row_index]]

    fig, ax = plt.subplots(1, 2, figsize = (10, 5), constrained_layout = True)

    ax[0].imshow(X_test[row_index])
    ax[0].set_title(f'Label: {df.loc[row_index, prediction_col]}\nPrediction: {df.loc[row_index, label_col]}')

    df.loc[row_index, df.dtypes != object].plot.bar(ax = ax[1])
    plt.xticks(rotation = 45, ha = 'right')
    plt.title('Probabilities of Each Class')
    plt.show()



def plot_ngrams(df, target:str, text:str, n_gram:int):
    ''' 
    This function allows visualizing the frequency of occurrence of words and n-grams (sets of a number of consecutive words) based on the target variable.
    
    Parameters
    ----------
    - df: Dataframe with a text variable.
    - target: Column of the dataframe where the target variable is located.
    - text: Column of the dataframe where the text is located.
    - n_gram: Number of consecutive words whose frequency we want to visualize.
    
    Returns
    ----------
    - Bar chart representing the frequency of words, sorted from highest to lowest, divided by target variable.
    '''
    df1 = df[df[target] ==1]
    df0 = df[df[target] ==0]

    # Custom function for ngram generation 
    def generate_ngrams(text, n_gram):
        token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [" ".join(ngram) for ngram in ngrams]

    # Custom function for horizontal bar chart 
    def horizontal_bar_chart(df, color):
        trace = go.Bar(
            y=df["word"].values[::-1],
            x=df["wordcount"].values[::-1],
            showlegend=False,
            orientation = 'h',
            marker=dict(
                color=color,
            ),
        )
        return trace

    # Get the bar chart from text with label 0 
    freq_dict = defaultdict(int)
    for sent in df0[text]:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

    # Get the bar chart from text with label 1 
    freq_dict = defaultdict(int)
    for sent in df1[text]:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

    # Creating two subplots
    fig = subplots.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words from text with label 0", 
                                          "Frequent words from text with label 1"])
    fig.add_trace(trace0, 1, 1)
    fig.add_trace(trace1, 1, 2)
    fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
    py.iplot(fig, filename='word-plots')

    return fig
    

def sunburst(df, interior:str, exterior:str, col_num:str, title:str):
    '''
    This is a Plotly Graph similar to pie chart but with two levels, interior is for columns which have one or two unique values, and 
    the exterior is for columns which have more values.
    Parameters
	----------
    df -> dataframe we are working with 
    interior -> recommended for columns which have two or so uniques values. Must be 'str'
    exterior -> recommended for columns which have more values, because the graph has more space than inside. Must be 'str'
    col_num -> it,s the column which we want measured, show us the quantity of each value for both column (interior and exterior), must be 'str'
    title -> the title we want to show in the pie, must be 'str'
    Return
	----------
    Return a pie chart with two levels, interior and exterior.
    
    '''

    fig = go.Figure()
    fig = px.sunburst(df, path=[interior, exterior], values=col_num, template = 'plotly_dark')
    fig.update_layout(width=800, height=600, title = title)
    return fig

def wordcloudviz(column):

    """
    Function to create a quick visualization of wordclouds in a given column of a dataframe called df.
    Parameters
    ----------
    column = name of the column of the dataframe. 
             Input example: df['column_name']
    Return
    ---------
        A wordcloud visualization of the words in the column.
    """
    # First, it concatenates the text in a "single" text.
    text = " ".join(comment for comment in column)

    # Creates a wordcloud visualization
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
 
def plot_cumulative_variance_ratio(pca, n_features):

    '''
    Function to visually represent the percentage of variance explained by each PCA component
    Parameters =
    pca: Name of the variable assigned to the PCA
    n_features: Number of PCA components
    Returns: 
    Matplotlib lineplot of the variance explained by each PCA component
    '''
    cumulative_variance_ratio = np.cumsum(pca.explained_varianceratio)[:n_features]

    # Create a bar plot of the cumulative variance ratio
    plt.plot(range(1, n_features + 1), cumulative_variance_ratio)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Ratio')

    # Show the plot
    plt.show()

def heatmap(df, n:int,target:str,columns:None):
    '''
    Heatmap which show us teh correlation of our numerical column of dataset with the target, where you can add specifics numbers 
    
    df -> must be the dataset we are working with
    n -> number of columns we want to correlate with the target
    target -> name of the column of the target, must be 'str'
    columns -> must be all the columns we have in the dataset in previous step, in type object (df.columns)
    Return:
    Heatmap with YlOrBr colour and two decimals, only wiht n number of columns which correlate with our target
    
    '''

    if columns is None:
        columns = df.columns
    
    cols = df[columns].corr().nlargest(n,target)[target].index

    cm = np.corrcoef(df[cols].values.T) 

    plt.figure(figsize=(20,10))
    hm = sns.heatmap(cm, cbar=True, annot=True, cmap='YlOrBr', fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)
    return hm

def plot_roc_curve(y_true, y_pred, pos_label=1, figsize=(8, 8)):
    '''
    Function to plot the ROC curve of a binary classifier
    Parameters:
    y_true: true labels
    y_pred: model predictions
    pos_label: positive label (default: 1)
    figsize: figure size (default: (8, 8))
    Returns: 
    Lineplot of the ROC curve
    
    '''
    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)

    # Compute the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Create the ROC curve plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_map(df, lat_col, lon_col, tooltip_col=None, zoom_start=3, map_type='OpenStreetMap'):
    """
    Function that creates an interactive map using folium from a dataframe with coordinates.
    
    Parameters
    ----------
        - df: dataframe with coordinates.
        - lat_col: name of the column containing latitudes.
        - lon_col: name of the column containing the longitudes.
        - tooltip_col: (optional) name of the column containing the additional information to show in the tooltip of each marker.
        - zoom_start: (optional) initial zoom level of the map.
        - map_type: (optional) type of map to use. Possible values: 'OpenStreetMap', 'Stamen Terrain', 'Stamen Toner', 'Stamen Watercolor', 'CartoDB positron', 'CartoDB dark_matter'.
    
    Returns
    -------
        - map: folium Map object with the added markers
    """
    
    # Create the map with the indicated type and zoom level.
    map = folium.Map(location=[df[lat_col][0], df[lon_col][0]], zoom_start=zoom_start, tiles=map_type)

    # Add markers for each point of the dataframe
    for index, row in df.iterrows():
        location = [row[lat_col], row[lon_col]]
        tooltip = row[tooltip_col] if tooltip_col else None
        folium.Marker(location=location, tooltip=tooltip).add_to(map)

    # Return the map
    return map


def correl_map_max(dataframe):
    """
    Function that, given a dataframe, eliminates the correlations greater than 0.9, visualizes the correlations and returns a new dataframe with columns that meet the condition of being less than 0.9. 
    and returns a new dataframe with the columns that meet the condition of being less than 0.9. 
    Parameters
    ----------
        - DataFrame: set of the data to which you want to apply.
    Returns
    -------
        - dataframe
    """

    # Calculate the correlation matrix
    corr_matrix = dataframe.corr()

    # Eliminate variables with correlation higher than 0.9
    high_corr_vars = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                varname_i = corr_matrix.columns[i]
                varname_j = corr_matrix.columns[j]
                if corr_matrix[varname_i].std() < corr_matrix[varname_j].std():
                    high_corr_vars.add(varname_i)
                else:
                    high_corr_vars.add(varname_j)
    dataframe = dataframe.drop(high_corr_vars, axis=1)

    # Generate the visualization of the correlation map
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,
                square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    return dataframe