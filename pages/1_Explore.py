import ast
import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Explore", page_icon=":black_nib:", layout="wide")

@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

@st.cache_data
def load_scraped_data(filename):
  df = pd.read_csv(filename, lineterminator='\n')
  return df

albums = load_data('./data/albums.csv')
artists = load_data('./data/artists.csv')
filteredtracks = load_scraped_data('./data/filteredtracks.csv')

st.sidebar.header("Explore")
st.sidebar.info("Exploration Page lets you explore the various datasets provided by Kaggle and Spotify.")

st.header('Exploration')

col1, col2 = st.columns(2)
with col1:
    data = st.selectbox("Select The Dataset You'd Like To Explore:",
                    ('Albums', 'Artists', 'FilteredTracks'))
    generate = st.button('Generate Profile')
with col2:
    if data:
        st.info(data + ':')
        st.info('~ ' + ' • '.join(list(eval(data.lower()).columns)) + ' ~')
st.dataframe(eval(data.lower()).head(1000), 100, 250, use_container_width=True)
if generate:
  if data:
    profile = ProfileReport(eval(data.lower()),
                              explorative = True,
                              dark_mode = True,
                              lazy = False,
                              title = data,
                              dataset = {
                              "description": "This dataset contains information on thousands of albums, artists, and songs that are collected from the Spotify platform using its API. In addition, the dataset also contains lower-level audio features of the songs, as well as their lyrics.",
                              "provider": "Collected from Spotify and Genius API's",
                              "release_year": "2022",
                              "url": "https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset",
                              })
  st_profile_report(profile)