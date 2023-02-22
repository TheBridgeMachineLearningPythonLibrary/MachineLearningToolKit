def wordcloudviz(column):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
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