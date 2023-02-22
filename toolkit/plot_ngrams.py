def plot_ngrams(df, target:str, text:str, n_gram:int):
    ''' 
    This function allows visualizing the frequency of occurrence of words and n-grams (sets of a number of consecutive words) based on the target variable.
    
    Parameters:
    - df: Dataframe with a text variable.
    - target: Column of the dataframe where the target variable is located.
    - text: Column of the dataframe where the text is located.
    - n_gram: Number of consecutive words whose frequency we want to visualize.
    
    Returns:
    - Bar chart representing the frequency of words, sorted from highest to lowest, divided by target variable.
    '''

    import plotly.graph_objs as go
    from plotly import tools
    from collections import defaultdict
    import pandas as pd
    import plotly.offline as py
    from wordcloud import STOPWORDS

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
    fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words from text with label 0", 
                                          "Frequent words from text with label 1"])
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
    py.iplot(fig, filename='word-plots')
