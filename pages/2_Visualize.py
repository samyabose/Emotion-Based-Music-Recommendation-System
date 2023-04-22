import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Visualize", page_icon=":chart_with_upwards_trend:", layout="wide")

@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

@st.cache_data
def load_scraped_data(filename):
  df = pd.read_csv(filename, lineterminator='\n')
  return df

# Might also import the data from '1_Explore.py' using importlib
albums = load_data('./data/albums.csv')
artists = load_data('./data/artists.csv')
filteredtracks = load_scraped_data('./data/filteredtracks.csv')
 
st.sidebar.header("Visualization")
st.sidebar.info("Visualization Page lets you visualize various aspects and traits within the datasets.")

st.header('Visualization')

tab1, tab2, tab3, tab4 = st.tabs(["Dominantion", "Popularity", "Distribution", "Co-appearance Network"])
with tab1:

    # Static plots are cached to save time

    @st.cache_data
    def mostRecordedArtists():

        # Top 50 most recorded artists

        db = albums.groupby('artist_id')['total_tracks'].sum('total_tracks').reset_index().sort_values('total_tracks', ascending=False)
        db = pd.merge(db, artists, on='artist_id', how='inner')[['name', 'total_tracks']].rename(columns={'name':'artist_name'})
        fig = px.bar(db.head(51).tail(50), y='artist_name', x='total_tracks', color='total_tracks', color_continuous_scale=px.colors.sequential.Purpor)
        return fig

    @st.cache_data
    def mostPopularTitles():

        # Top 50 most popular titles

        db = filteredtracks[['track_name', 'popularity']].sort_values('popularity', ascending=False).rename(columns={'track_name':'song_title'})
        fig = px.bar(db.head(101).tail(100), y='song_title', x='popularity', color='popularity', color_continuous_scale=px.colors.sequential.RdBu)
        return fig

    
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.write('Top 50 Most Recorded Artists')
        st.plotly_chart(mostRecordedArtists(), use_container_width=True)
    with col2:
        st.write('Top 100 Most Popular Tracks')
        st.plotly_chart(mostPopularTitles(), use_container_width=True)

with tab2:

    # Static plots are cached to save time

    @st.cache_data
    def mostPopularArtist():

        # Top 50 most popular artists
        fig = px.bar(artists.sort_values('artist_popularity', ascending=False).head(51).tail(50), y='name', x='artist_popularity', color='artist_popularity', color_continuous_scale=px.colors.sequential.deep)
        return fig

    @st.cache_data
    def mostFollowedArtist():

        # Top 50 most followed artists
        
        fig = px.bar(artists.sort_values('followers', ascending=False).head(51).tail(50), y='name', x='followers', color='followers', color_continuous_scale=px.colors.sequential.Burg)
        return fig

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.write('Top 50 Most Popular Artists')
        st.plotly_chart(mostPopularArtist(), use_container_width=True)
    with col2:
        st.write('Top 50 Most Followed Artists')
        st.plotly_chart(mostFollowedArtist(), use_container_width=True)

with tab3:

    # Static plots are cached to save time

    @st.cache_data
    def mostPopularGenres():
        
        # Top 50 most popular genres

        genres = {}
        db =filteredtracks.sort_values('popularity', ascending=False).head(1000)[['genres']]
        for i in db.values:
            for j in eval(i[0]):
                if j not in genres:
                    genres[j] = 1
                else:
                    genres[j] += 1

        keys = list(genres.keys())
        values = list(genres.values())
        sorted_value_index = np.argsort(values)[::-1]
        genres = {keys[i]: values[i] for i in sorted_value_index}

        genres = pd.DataFrame({'genre': genres.keys(), 'popularity': genres.values()})
        fig = px.bar(genres.head(50), y='genre', x='popularity', color='popularity', color_continuous_scale=px.colors.sequential.Magenta)
        return fig
    
    @st.cache_data
    def mostPopularAlbums():

        # Top 50 most popular albums

        db = pd.merge(albums, filteredtracks, on='album_id', how='inner').rename(columns={'name':'album_name'}).groupby('album_name').sum('popularity').reset_index()
        db = db[['album_name', 'popularity']].sort_values('popularity', ascending=False)
        fig = px.bar(db.head(51).tail(50), y='album_name', x='popularity', color='popularity', color_continuous_scale=px.colors.sequential.thermal)
        return fig

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.write('Top 50 Most Popular Genres')
        st.plotly_chart(mostPopularGenres(), use_container_width=True)
    with col2:
        st.write('Top 50 Most Popular Albums')
        st.plotly_chart(mostPopularAlbums(), use_container_width=True)

with tab4:

    # Static plots are cached to save time

    @st.cache_data
    def coappearancePopFol():
        
        # Coappearance of popularity and followers

        fig = px.scatter(artists.sort_values(['artist_popularity', 'followers']).sample(n=1000), x='artist_popularity', y='followers', size='followers', color='artist_popularity', color_continuous_scale=px.colors.sequential.Darkmint, marginal_x='rug', marginal_y='rug')
        return fig

    
    st.write('Coappearance of Popularity and Followers')
    st.plotly_chart(coappearancePopFol(), use_container_width=True)