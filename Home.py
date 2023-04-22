import base64
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Home",
    page_icon=":earth_africa:",
)

def load_bootstrap():
    return st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.sidebar.markdown(
    '''
        <div style="border: thin solid black; border-radius: 5px;">
            <div style="background-image: url(data:image/png;base64,{}); background-repeat: no-repeat; height: 125px;">
                <a href='https://github.com/samya-ravenXI' style="position: absolute; z-index: 2; top: 40px; left: 20px;">
                    <img src="https://skillicons.dev/icons?i=github" alt="GitHub"/>
                </a>
                <h2 style="color: rgb(224, 224, 224); position: absolute; z-index: 2; top: 15px; left: 80px; font-family: sans-serif;">Emotion Based Music Recommendation System</h1>
            </div>
            <div style="margin-top: 40px">
                <a href="https://github.com/samya-ravenXI/Movie-Recommendation-System" style="position: absolute; z-index: 2; top: 131px; left: 15px">
                    <img src="https://img.shields.io/badge/github-repo-white" alt="repo"/>
                </a>
                <a href="https://colab.research.google.com/drive/1ahxyp8i9Ngy2nyA5THSOwDzVS99prLMF?usp=sharing" style="position: absolute; z-index: 2; top: 131px; right: 92px">
                    <img src="https://img.shields.io/badge/colab-notebook-orange" alt="repo"/>
                </a>
                <a href="https://huggingface.co/facebook/blenderbot-400M-distill?text=Hi." style="position: absolute; z-index: 2; top: 131px; right: 15px">
                    <img src="https://img.shields.io/badge/API-Key-green" alt="repo"/>
                </a>
            </div>
        </div>
    '''.format(img_to_bytes('./icons/cover.jpg')),
    unsafe_allow_html=True)

with st.container():
    st.title("Emotion Based Music Recommendation System")
    for i in range(2):
        st.markdown('#')
    st.caption('This projects uses Spotify and Genius to recommend music along with Spotify widget.')
    st.caption('It contains emotion based models trained on texts, videos and audios to provided recommendation via those media.')
    st.caption('Tip: To use the chatbot interface in the text based recommendation section, an API Key is required.')

    for i in range(2):
        st.markdown('#')
    st.markdown('#####')
    st.markdown('---')

    col1, col2, col3 = st.columns((1.4,0.85,0.85), gap='large')
    with col1:
        st.empty()
        st.empty()
        st.markdown("<a href='https://docs.streamlit.io/library/get-started'><img src='data:image/png;base64,{}' class='img-fluid' width=80%/></a>".format(img_to_bytes('./icons/streamlit.png')), unsafe_allow_html=True)
    with col2:
        st.markdown("<a href='https://developer.spotify.com/'><img src='data:image/png;base64,{}' class='img-fluid' width=40%/></a>".format(img_to_bytes('./icons/spotify.png')), unsafe_allow_html=True)
    with col3:
        st.markdown("<a href='https://colab.research.google.com/drive/1ahxyp8i9Ngy2nyA5THSOwDzVS99prLMF?usp=sharing'><img src='data:image/png;base64,{}' class='img-fluid' width=50%/></a>".format(img_to_bytes('./icons/colab.png')), unsafe_allow_html=True)