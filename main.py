import streamlit as st
import pandas as pd
import os
import nltk
import re
import string
from txtai.embeddings import Embeddings
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

x = ['rm','price']
y= ['kk',"residential college","at","in"]
less = ['le','cheaper','<','under']
more = ['more','expensive','pricier']
contains_word = lambda s, l: any(map(lambda x: x in s, l))
sql=''
order=''

def preprocess(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# Extract numerical values from text
def extract_numerical(text):
    # Use regular expression to find numerical values
    numerical_values = re.findall(r'\d+\.?\d*', text)
    # Convert numerical values to float
    numerical_values = [float(val) for val in numerical_values]
    return numerical_values

def load_data_and_embeddings():

    # Embeddings with sentence-transformers backend
    embeddings = Embeddings({"method": "transformers", "path": "sentence-transformers/paraphrase-mpnet-base-v2", "content": True})

    # Index subset of data
    embeddings.load('embeddings.tar.gz')

    return embeddings

embeddings  = st.cache_resource(load_data_and_embeddings)()

st.title('Food Search Engine')

user_query = st.text_input('Search')

if user_query:

    # Preprocess user query
    preprocessed_query = preprocess(user_query)
    #price
    for i in x+less:
        if i in preprocessed_query:
            query_split = preprocessed_query.split(i)[1]
            price = extract_numerical(query_split)[0] if re.search(r'\d', query_split) else None
            if contains_word(preprocessed_query,less) and price:
                sql = sql+" and price <= "+str(price)
            elif contains_word(preprocessed_query,more) and price:
                sql = sql+" and price >= "+str(price)
            elif contains_word(preprocessed_query,"cheap") and not price:
                order = ' order by price asc'
            elif price:
                sql = sql+" and price = "+str(price)

            # if contains_word(preprocessed_query,y) and re.search(r'\d', preprocessed_query):
            #     query_splitl = preprocessed_query.split(i)[1]
            #     location = extract_numerical(query_splitl)[0] 
            #     sql = sql+" and college = "+str(location)
     
    for i in y:
        if i in preprocessed_query and re.search(r'\d', preprocessed_query):
            query_split = preprocessed_query.split(i)[1]
            location = extract_numerical(query_split)[0] 
            sql = sql+" and college = "+str(location) 
            break     

col1, col2,col3,col4 = st.columns([1,1,1,1])

with col1:
    searchbt=st.button('Search')
with col2:
    randombt=st.button('Random')


if searchbt:
    if user_query:
        results = embeddings.search(f"select menu, price, college,description from txtai where similar('{preprocessed_query}'){sql}{order}",limit=5)
        #results = [round(i,2) for i in results['price'] ]
        df = pd.DataFrame.from_dict(results)
        #result = [description[x[0]] for x in results]
        #table = df[df['description'].isin(results)]
        #st.write(location)
        # st.write(price)
        st.write(preprocessed_query)
        # for res in results:
        #     st.write(res)
        #st.dataframe(df.style.format(subset=['price'], formatter="{:.2f}"))
        st.table(df.style.format(subset=['price'], formatter="{:.2f}"))
            
    
    else:
        st.write('stoopit')

if randombt:
    if user_query:
        st.table(embeddings.search(f"select menu, price, college, description from txtai where similar('{preprocessed_query}'){sql} order by random()",limit=5))
    else:
        st.table(embeddings.search(f"select menu, price, college from txtai order by random()",limit=5))
        