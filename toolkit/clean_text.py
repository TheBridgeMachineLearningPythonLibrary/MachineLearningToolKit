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
    
    import pandas as pd 
    import re
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer

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