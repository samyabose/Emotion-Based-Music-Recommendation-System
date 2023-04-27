import re
import nltk
import pickle
import requests
import numpy as np
import configparser
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

st.set_page_config(page_title="Search", page_icon=":mag:", layout="wide")

@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

@st.cache_data
def load_scraped_data(filename):
  df = pd.read_csv(filename, lineterminator='\n')
  return df

@st.cache_data
def load_text(filename):
    df = pd.read_csv(filename, names=['Text', 'Emotion'], sep=';')
    return df

# Might also import the data from '1_Explore.py' using importlib
albums = load_data('./data/albums.csv')
artists = load_data('./data/artists.csv')
filteredtracks = load_scraped_data('./data/filteredtracks.csv')
 
st.sidebar.header("Search")
st.sidebar.info("Search Page lets you search for recommendations on popular track, albums, artists or parametric features of the search criteria.")

st.header('Search')

tab1, tab2, tab3, tab4 = st.tabs(['Popularity Based', 'Genre Based', 'NearestNeighbours Based', 'Search'])

def mov(res, ind, dir):

    # Defines the container format

    st.markdown("<iframe src='{}'' width='180' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>".format(dir + res.iloc[ind]['uri']), unsafe_allow_html=True)

def container(desc, dir):

    # Defines the container layout
    col1, col2, col3, col4, col5 = st.columns(5)
    try:
        with col1:
            mov(desc, 0, dir)
        with col2:
            mov(desc, 1, dir)
        with col3:
            mov(desc, 2, dir)
        with col4:
            mov(desc, 3, dir)
        with col5:
            mov(desc, 4, dir)
    except:
        pass

def desc(x):

    artist_name = str(x.artist_name).replace(' ', '').lower()
    album_name = str(x.album_name).replace(' ', '').lower()
    track_name = str(x.track_name).replace(' ', '').lower()
    playlist = str(x.playlist).replace(' ', '').lower()
    genres = ' '.join(i.replace(' ', '').lower() for i in str(x.genres[0])[2:-2].split("', '"))
    lyrics = str(x.lyrics).replace('\r', '').replace('\n', '').lower()
    result = re.sub("[^a-zA-Z0-9 ]", "", artist_name + ' ' + album_name + ' ' + track_name + ' ' + playlist + ' ' + genres + ' ' + lyrics)
    return result

@st.cache_data
def contextBased():
    # Stop word removal for english words
    count = CountVectorizer(stop_words='english', ngram_range=(1,3))

    db = pd.merge(filteredtracks, albums, on='album_id', how='inner')[['artist_name', 'artist_id_x', 'name', 'album_id', 'track_name', 'uri_x', 'playlist', 'genres', 'lyrics']].rename(columns={'name':'album_name', 'artist_id_x':'artist_id', 'uri_x':'track_id'})
    
    # Imputing missing values
    tempdb = db.copy()
    tempdb['description'] = tempdb.apply(desc, axis=1)

    count_matrix = count.fit_transform(tempdb['description'])

    return tempdb, count, count_matrix

tempdb, count, count_matrix = contextBased()

def contextBasedRecommendations(title, mode, num):
  
  if mode=='artist':
    desc = tempdb[tempdb['artist_name'] == title].description.iloc[0]
  if mode=='album':
    desc = tempdb[tempdb['album_name'] == title].description.iloc[0]
  if mode=='track':
    desc = tempdb[tempdb['track_name'] == title].description.iloc[0]



  query_vec = count.transform([desc])                                           # Transforming the modified title into a query vec using the count vectorizer
  similarity = cosine_similarity(query_vec, count_matrix).flatten()             # Computing cosine similarity measure

  inx = np.argsort(similarity)[::-1][:500]                                      # Getting the most relevant recommendations
  res = tempdb.iloc[inx]
  return res.sample(n=num)[['artist_id', 'album_id', 'track_id']]

def popular(mode, n):

        db = contextBasedRecommendations(title, mode.lower(), n)
        return db

def mov2(res, ind, dir, mode):

    # Defines the container format
    if mode.lower() == 'artist':
        st.markdown("<iframe src='{}'' width='180' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>".format(dir + res.iloc[ind]['artist_id']), unsafe_allow_html=True)
    if mode.lower() == 'album':
        st.markdown("<iframe src='{}'' width='180' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>".format(dir + res.iloc[ind]['album_id']), unsafe_allow_html=True)
    if mode.lower() == 'track':
        st.markdown("<iframe src='{}'' width='180' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>".format(dir + res.iloc[ind]['track_id']), unsafe_allow_html=True)

def container2(desc, mode, dir):

    # Defines the container layout
    col1, col2, col3, col4, col5 = st.columns(5)
    try:
        with col1:
            mov2(desc, 0, dir, mode)
        with col2:
            mov2(desc, 1, dir, mode)
        with col3:
            mov2(desc, 2, dir, mode)
        with col4:
            mov2(desc, 3, dir, mode)
        with col5:
            mov2(desc, 4, dir, mode)
    except:
        pass


with tab1:

    # Popularity based recommendation

    def popularTracks(n):

        # Returns random 5 songs from top 500 most popular songs

        db = filteredtracks.sort_values('popularity', ascending=False).head(500).sample(n=n)
        db = pd.merge(db, artists, on='artist_id', how='inner')
        db = pd.merge(db, albums, on='album_id', how='inner')[['track_name', 'artist_name', 'name_y', 'images', 'genres_x', 'release_year', 'uri_x']]
        db = db.rename(columns={'name_y':'album_name', 'genres_x':'genres', 'uri_x':'uri'})
        return db

    st.subheader('Popular Tracks')
    container(popularTracks(5), 'https://open.spotify.com/embed/track/')

    def popularAlbums(n):

        # Returns random 5 songs from top 500 most popular albums

        db = pd.merge(albums, filteredtracks, on='album_id', how='inner').groupby('album_id').sum('popularity').reset_index()
        db = pd.merge(db, albums, on='album_id', how='inner')
        db = pd.merge(db, artists, on='artist_id', how='inner')
        index = db[(db['name_y'] == 'Various Artists') | (~db.name_y.str.isalpha()) | (db['genres'] == '[]')].index
        db.drop(index, inplace = True)
        db = db.sort_values('popularity', ascending=False).head(500).sample(n=n)
        db = db[['name_x', 'name_y', 'images', 'genres', 'release_date', 'uri']]
        db['release_date'] = pd.to_datetime(db.release_date).dt.year
        db = db.rename(columns={'name_x':'album_name', 'name_y':'artist_name', 'release_date':'release_year'})
        db['uri'] = db.uri.apply(lambda x: x[14:])
        return db

    st.subheader('Popular Albums')
    container(popularAlbums(5), 'https://open.spotify.com/embed/album/')

    def popularArtists(n):
        db = artists.sort_values('artist_popularity', ascending=False).head(500).sample(n=n).rename(columns={'artist_id':'uri'})
        return db

    st.subheader('Popular Artists')
    container(popularArtists(5), 'https://open.spotify.com/embed/artist/')

with tab2:

    # Genre based recommendation

    st.subheader('Select A Genre:')
    genres_to_include = sorted(['electronic', 'soul', 'country', 'folk', 'metal', 'hip hop', 'jazz', 'k-pop', 'latin', 'pop', 'grunge', 'r&b', 'classical', 'rock'])
    gen = st.select_slider('Genre', options=genres_to_include, label_visibility='collapsed')

    def genreBasedTracks(gen, n):
        db = pd.read_csv('./data/filteredtracks.csv', lineterminator='\n')
        db['genres'] = db.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        db_exp = db.explode("genres")[db.explode("genres")["genres"].isin(gen)]
        db_exp.loc[db_exp["genres"]=="korean pop", "genres"] = "k-pop"
        db_exp_indices = list(db_exp.index.unique())
        db = db[db.index.isin(db_exp_indices)]
        db = db.reset_index(drop=True)
        db = db.head(500).sample(n=n)
        return db

    st.subheader('Popular ' + gen.capitalize() + ' Tracks')
    container(genreBasedTracks([gen], 5), 'https://open.spotify.com/embed/track/')

    def genreBasedAlbums(gen, n):
        db = pd.merge(albums, artists, on='artist_id', how='inner')
        db['genres'] = db.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        db_exp = db.explode("genres")[db.explode("genres")["genres"].isin(gen)]
        db_exp.loc[db_exp["genres"]=="korean pop", "genres"] = "k-pop"
        db_exp_indices = list(db_exp.index.unique())
        db = db[db.index.isin(db_exp_indices)]
        db = db.reset_index(drop=True)
        db['uri'] = db.uri.apply(lambda x: x[14:])
        db = db.head(500).sample(n=n)
        return db

    st.subheader('Popular ' + gen.capitalize() + ' Albums')
    container(genreBasedAlbums([gen], 5), 'https://open.spotify.com/embed/album/')

    def genreBasedArtists(gen, n):
        db = artists
        db['genres'] = db.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        db_exp = db.explode("genres")[db.explode("genres")["genres"].isin(gen)]
        db_exp.loc[db_exp["genres"]=="korean pop", "genres"] = "k-pop"
        db_exp_indices = list(db_exp.index.unique())
        db = db[db.index.isin(db_exp_indices)]
        db = db.reset_index(drop=True)
        db = db.rename(columns={'artist_id':'uri'})
        db = db.head(500).sample(n=n)
        return db

    st.subheader('Popular ' + gen.capitalize() + ' Artists')
    container(genreBasedArtists([gen], 5), 'https://open.spotify.com/embed/artist/')

with tab3:

    # Nearestneighbour based recommendation

    genres_to_include = sorted(['Electronic', 'Soul', 'Country', 'Folk', 'Metal', 'Hip hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Grunge', 'R&b', 'Classical', 'Rock'])
    st.subheader("Customize Features")
    gen = st.selectbox('Genre', tuple(genres_to_include))
    gen = gen.lower()
    col1, col2, col3 = st.columns((2,0.5,2))
    with col1:
        start_year, end_year = st.slider('Release Year',1935, 2019, (2005, 2019))
        acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
        danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
        energy = st.slider('Energy', 0.0, 1.0, 0.5)
    with col2: 
        pass
    with col3:
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.0)
        valence = st.slider('Valence', 0.0, 1.0, 0.45)
        tempo = st.slider('Tempo', 0.0, 244.0, 118.0)

    audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]
    def n_neighbors_uri_audio(genre, start_year, end_year, test_feat, n=15):
        genre = genre.lower()
        db = pd.read_csv('./data/filteredtracks.csv', lineterminator='\n')
        db['genres'] = db.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        exploded_db = db.explode("genres")
        genre_data = exploded_db[(exploded_db["genres"]==genre) & (exploded_db["release_year"]>=start_year) & (exploded_db["release_year"]<=end_year)]
        genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]
        nn = NearestNeighbors()
        nn.fit(genre_data[audio_feats].to_numpy())
        n_neighbors = nn.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
        uris = genre_data.iloc[n_neighbors]["uri"].tolist()
        db = pd.DataFrame({'uri':uris}).sample(n=n)
        return db

    st.markdown('---')

    try:
        test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
        db = n_neighbors_uri_audio(gen, start_year, end_year, test_feat)
        st.subheader('Recommended ' + gen.capitalize() + ' Tracks')
        container(db.head(5), 'https://open.spotify.com/embed/track/')
        container(db.head(10).tail(5), 'https://open.spotify.com/embed/track/')
        container(db.head(15).tail(5), 'https://open.spotify.com/embed/track/')
    except:
        st.error("Couldn't find results fitting the given criteria! Here are a few popular songs of that genre!")
        st.subheader('Popular ' + gen.capitalize() + ' Tracks')
        container(genreBasedTracks([gen], 5), 'https://open.spotify.com/embed/track/')
        st.subheader('Popular ' + gen.capitalize() + ' Albums')
        container(genreBasedAlbums([gen], 5), 'https://open.spotify.com/embed/album/')
        st.subheader('Popular ' + gen.capitalize() + ' Artists')
        container(genreBasedArtists([gen], 5), 'https://open.spotify.com/embed/artist/')

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox('Enter the search mode:', ('Artist', 'Album', 'Track'))
    with col2:
        if mode.lower() == 'artist':
            title = st.selectbox('Enter the {}:'.format(mode.lower()), tuple(sorted(list(np.unique(tempdb.artist_name.values))[1:])))
        if mode.lower() == 'album':
            title = st.selectbox('Enter the {}:'.format(mode.lower()), tuple(sorted(list(np.unique(tempdb.album_name.values)))))
        if mode.lower() == 'track':
            title = st.selectbox('Enter the {}:'.format(mode.lower()), tuple(sorted(list(np.unique(tempdb.track_name.values)))))

    if mode.lower() == 'artist':
        db = tempdb[tempdb['artist_name'] == title][['artist_id', 'album_id', 'track_id']]
        if len(db)>5:
            db = db.sample(n=5)
        st.subheader('Popular Songs ~ {}'.format(title))
        container2(db, 'track', 'https://open.spotify.com/embed/track/')

        st.subheader('Popular Albums ~ {}'.format(title))
        container2(db, 'album', 'https://open.spotify.com/embed/album/')

        st.subheader('Similar Artists')
        container2(popular(mode, 5), mode, 'https://open.spotify.com/embed/artist/')

    if mode.lower() == 'album':
        db = tempdb[tempdb['album_name'] == title][['artist_id', 'album_id', 'track_id']]
        if len(db)>5:
            db = db.sample(n=5)
        st.subheader('Popular Songs ~ {}'.format(title))
        container2(db, 'track', 'https://open.spotify.com/embed/track/')

        artist = tempdb[tempdb['album_name'] == title]['artist_name'].iloc[0]
        db = tempdb[tempdb['artist_name'] == artist][['artist_id', 'album_id', 'track_id']]
        if len(db)>5:
            db = db.sample(n=5)
        st.subheader('Popular Albums ~ {}'.format(artist))
        container2(db, 'album', 'https://open.spotify.com/embed/album/')

        st.subheader('Similar Albums')
        container2(popular(mode, 5), mode, 'https://open.spotify.com/embed/album/')

    if mode.lower() == 'track':
        artist = tempdb[tempdb['track_name'] == title]['artist_name'].iloc[0]
        db = tempdb[tempdb['artist_name'] == artist][['artist_id', 'album_id', 'track_id']]
        if len(db)>5:
            db = db.sample(n=5)
        st.subheader('Popular Songs ~ {}'.format(artist))
        container2(db, 'track', 'https://open.spotify.com/embed/track/')

        album = tempdb[tempdb['track_name'] == title]['album_name'].iloc[0]
        db = tempdb[tempdb['album_name'] == album][['artist_id', 'album_id', 'track_id']]
        if len(db)>5:
            db = db.sample(n=5)
        st.subheader('Popular Songs ~ {}'.format(album))
        container2(db, 'track', 'https://open.spotify.com/embed/track/')

        st.subheader('Similar Tracks')
        container2(popular(mode, 5), mode, 'https://open.spotify.com/embed/track/')
