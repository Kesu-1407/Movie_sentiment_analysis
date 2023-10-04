import pickle
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
from imdb import IMDb
import matplotlib.pyplot as plt
import streamlit as st
import warnings
from wordcloud import WordCloud
import time

def test_model1(sen, cv, lr):
    s = cv.transform([sen]).toarray()
    res = lr.predict(s)[0]
    if res == 0:
        return "negative"
    else:
        return "positive"


def analyze_movie_sentiment(imdb_id, cv, lr):
    url = f'https://www.imdb.com/title/tt{imdb_id}/reviews'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*text show-more.*')
    results = soup.find_all('div', {'class': regex})
    reviews = [result.text for result in results]
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    df['sentiment'] = df['review'].apply(lambda x: test_model1(x[:512], cv, lr))
    positive_count = df[df['sentiment'] == 'positive'].shape[0]
    total_reviews = df.shape[0]
    percentage_positive = (positive_count / total_reviews) * 100
    return percentage_positive,results,df

# Load the trained Logistic Regression model
with open('logistic_regression.pkl', 'rb') as f:
    lr = pickle.load(f)

# Load the TfidfVectorizer
with open('count-vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

imdb_id_cache = {}
def get_imdb_id(movie_name):
    # Check if IMDb ID is already cached
    if movie_name in imdb_id_cache:
        return imdb_id_cache[movie_name]

    ia = IMDb()
    movies = ia.search_movie(movie_name)
    if movies:
        imdb_id = movies[0].movieID
        # Cache the IMDb ID
        imdb_id_cache[movie_name] = imdb_id
        return imdb_id
    else:
        return None

# df1=None
def main():
    st.title("Movie Sentiment Analysis")
    st.subheader("Enter the movie name:")
    movie_name = st.text_input(" ")
    if st.button("Analyze"):

        progress_bar = st.progress(0)  # Create a progress bar
        status_text = st.empty()  # Create an empty placeholder for status text
        # Update progress bar and status text
        progress_bar.progress(10)
        status_text.text("Scraping comments...")

        imdb_id = get_imdb_id(movie_name)
        if imdb_id is None:
            st.write(" :red[sorry results not found..... check spelling and try again] ")
        else:

            progress_bar.progress(30)
            status_text.text("Analyzing comments...")

            ans,result,df = analyze_movie_sentiment(imdb_id, cv, lr)

            progress_bar.progress(70)
            status_text.text("Generating visualizations...")
            time.sleep(1.5)

            if (ans>50):
                st.subheader(f"The percentage of positive reviews for {movie_name} is: :green[{ans}%]  :smile:") 
            elif (ans<50):
                st.subheader(f"The percentage of positive reviews for {movie_name} is: :red[{ans}%]  :disappointed:")
            else:
                st.subheader(f"The percentage of positive reviews for {movie_name} is: :blue[{ans}%]") 

            labels = ['Positive', 'Negative']
            sizes = [ans, 100 - ans]
            colors = ['green', 'red']
            fig, (ax,ax2) = plt.subplots(1, 2, figsize=(20,15), facecolor='black' ,gridspec_kw={'hspace': 0.1})

            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 30})
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('Sentiment Analysis')

            comments = ' '.join([res.text for res in result])
            wordcloud = WordCloud(font_path='Mefikademo-owEAq.ttf',background_color='black').generate(comments)
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.set_title('Comments Word Cloud')
            ax2.axis('off')

            # progress_bar.progress(100)
            # status_text.text("Analysis complete!")
            # Display the chart and word cloud in Streamlit
            warnings.filterwarnings("ignore", category=UserWarning)
            st.pyplot(fig)

            st.subheader("Top 10 Comments and Sentiments:")
            for i, row in df[['review', 'sentiment']].head(10).iterrows():
                with st.expander(f"Comment {i + 1} - Sentiment: {row['sentiment']}"):
                    st.write(row['review'])

            progress_bar.progress(100)
            status_text.text("Analysis complete!")
        

if __name__ == '__main__':
    main()


