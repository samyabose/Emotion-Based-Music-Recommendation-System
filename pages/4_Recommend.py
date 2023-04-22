import re
import av
import io
import os
import cv2
import nltk
import pickle
import librosa
import requests
import threading
import numpy as np
import pandas as pd
from io import BytesIO
import streamlit as st
import soundfile as sf
from nltk.corpus import stopwords
from streamlit_chat import message
from streamlit_webrtc import webrtc_streamer
import streamlit.components.v1 as components

from sklearn.neural_network import MLPClassifier


from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import model_from_json

from tensorflow.keras.utils import to_categorical
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Recommend", page_icon=":notes:", layout="wide")

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
 
st.sidebar.header("Recommendation")
st.sidebar.info("Recommendation page lets you ask for Text Based, Video Based or Audio Based recommendations.")

st.header('Recommendation')

tab1, tab2, tab3 = st.tabs(['Text Based', 'Video Based', 'Audio Based'])

def mov(res, ind, dir):

    # Defines the container format

    st.markdown("<iframe src='{}'' width='180' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>".format(dir + res.iloc[ind]['uri']), unsafe_allow_html=True)


def container(desc, dir):

    # Defines the container layout
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            mov(desc, 0, dir)
        with col2:
            mov(desc, 1, dir)
        with col3:
            mov(desc, 2, dir)
    except:
        pass    

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

moodToGenre = {
            'joy': ['k-pop', 'pop', 'hip hop', 'rock'],
            'happy': ['k-pop', 'pop', 'hip hop', 'rock'],
            'calm': ['k-pop', 'pop', 'hip hop', 'rock'],
            'sad': ['soul', 'folk', 'classical', 'r&b'],
            'sadness': ['soul', 'folk', 'classical', 'r&b'],
            'anger': ['metal', 'hip hop', 'rock', 'grunge'],
            'angry': ['metal', 'hip hop', 'rock', 'grunge'],
            'fear': ['country', 'soul', 'jazz', 'classical', 'latin'],
            'fearful': ['country', 'soul', 'jazz', 'classical', 'latin'],
            'disgust': ['country', 'soul', 'jazz', 'classical', 'latin'],
            'love': ['electronic', 'jazz', 'k-pop', 'pop', 'r&b'],
            'neutral': ['electronic', 'jazz', 'k-pop', 'pop', 'r&b'],
            'surprise': ['electronic', 'folk', 'hip hop', 'r&b', 'soul', 'pop'],
            'surprised': ['electronic', 'folk', 'hip hop', 'r&b', 'soul', 'pop']
        }

def st_audiorec():

    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join('', "./data/audio_based/st_audiorec/frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # Create an instance of the component: STREAMLIT AUDIO RECORDER
    raw_audio_data = st_audiorec()  # raw_audio_data: stores all the data returned from the streamlit frontend
    wav_bytes = None                # wav_bytes: contains the recorded audio in .WAV format after conversion

    # the frontend returns raw audio data in the form of arraybuffer
    # (this arraybuffer is derived from web-media API WAV-blob data)

    if isinstance(raw_audio_data, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, raw_audio_data = zip(*raw_audio_data['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            raw_audio_data = np.array(raw_audio_data)  # convert to np array
            sorted_ints = raw_audio_data[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            # wav_bytes contains audio data in byte format, ready to be processed further
            wav_bytes = stream.read()

    return wav_bytes

model = model_from_json(open("./data/video_based/fer_sl.json", "r").read())
model.load_weights('./data/video_based/fer_sl.h5')
face_haar_cascade = cv2.CascadeClassifier('./data/video_based/haarcascade_frontalface_default.xml')

with open('./data/audio_based/mlp.pkl', 'rb') as file:
    mlp = pickle.load(file)

def extract_feature(file_name, mfcc, chroma, mel):
  with sf.SoundFile(file_name) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate=sound_file.samplerate
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
  return result

nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame


with tab1:
    st.subheader('Text Based Recommendation') 
    def lemmatization(text):
        lemmatizer= WordNetLemmatizer()
        text = text.split()
        text=[lemmatizer.lemmatize(y) for y in text]
        return " " .join(text)

    def remove_stop_words(text):

        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)

    def removing_numbers(text):
        text=''.join([i for i in text if not i.isdigit()])
        return text

    def lower_case(text):
        text = text.split()
        text=[y.lower() for y in text]
        return " " .join(text)

    def removing_punctuations(text):
        # Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )
        
        # Remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()

    def removing_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_small_sentences(df):
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
                    
    def normalize_text(df):
        df.Text=df.Text.apply(lambda text : lower_case(text))
        df.Text=df.Text.apply(lambda text : remove_stop_words(text))
        df.Text=df.Text.apply(lambda text : removing_numbers(text))
        df.Text=df.Text.apply(lambda text : removing_punctuations(text))
        df.Text=df.Text.apply(lambda text : removing_urls(text))
        df.Text=df.Text.apply(lambda text : lemmatization(text))
        return df

    def normalized_sentence(sentence):
        sentence= lower_case(sentence)
        sentence= remove_stop_words(sentence)
        sentence= removing_numbers(sentence)
        sentence= removing_punctuations(sentence)
        sentence= removing_urls(sentence)
        sentence= lemmatization(sentence)
        return sentence

    with open('./data/text_based/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    with open('./data/text_based/labelencoder.pkl', 'rb') as file:
        le = pickle.load(file)

    adam = Adam(learning_rate=0.005)
    text_based = load_model('./data/text_based/text_based.h5', compile=False)
    text_based.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # API_KEY = st.secrets["API_KEY"]
    API_KEY = "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"

    API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": "Bearer {}".format(API_KEY)}

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def get_text():
        input_text = st.text_input("Enter some text:", "Hey!")
        return input_text 
    col1, col2 = st.columns((1.5,2.5))
    with col1:
        user_input = get_text()
        chatholder = st.empty()
    with col2:
        placeholder = st.empty()
    if user_input:
        output = query({
            "inputs": {
                "past_user_inputs": st.session_state.past,
                "generated_responses": st.session_state.generated,
                "text": user_input,
            },"parameters": {"repetition_penalty": 1.33},
        })

        if len(st.session_state['generated']) >= 3:
            st.session_state['generated'] = st.session_state['generated'][1:]
            st.session_state['past'] = st.session_state['past'][1:]
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["generated_text"])
            
    if st.session_state['generated']:
            with chatholder.container():
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], avatar_style="shapes", key=str(i))
                    message(st.session_state['past'][i], avatar_style="lorelei-neutral", is_user=True, key=str(i) + '_user')

                    sentence = normalized_sentence(st.session_state['past'][i])
                    sentence = tokenizer.texts_to_sequences([sentence])
                    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
                    result = le.inverse_transform(np.argmax(text_based.predict(sentence), axis=-1))[0]
                    prob =  np.max(text_based.predict(sentence))

                    # st.caption(f"Mood:{result}; Probability: {prob}; Genres: {', '.join(moodToGenre[result])}")

            with placeholder.container():
                st.write('Recommended Tracks')
                container(genreBasedTracks(moodToGenre[result], 3), 'https://open.spotify.com/embed/track/')
                st.write('Recommended Albums')
                container(genreBasedAlbums(moodToGenre[result], 3), 'https://open.spotify.com/embed/album/')
                st.write('Recommended Artists')
                container(genreBasedArtists(moodToGenre[result], 3), 'https://open.spotify.com/embed/artist/')

with tab2:
    st.subheader('Video Based Recommendation') 
    col1, col2 = st.columns((1.5,2.5))
    with col2:
        placeholder = st.empty()
        with placeholder.container():
            st.caption("Press 'Start' to record a small video, before clicking on 'Generate Recommendations'")

    with col1: 
        ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
        cap = st.button('Generate Recommendation', use_container_width=True)

        while ctx.state.playing:
            with lock:
                img = img_container["img"]
            if img is None:
                continue
            cv2.imwrite('./data/video_based/input.png', img)
            break

        if cap:
            try:
                img = cv2.imread('./data/video_based/input.png')
                height, width , channel = img.shape
                gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_haar_cascade.detectMultiScale(gray_image)
                for (x,y, w, h) in faces:
                    roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
                    roi_gray=cv2.resize(roi_gray,(48,48))
                    image_pixels = img_to_array(roi_gray)
                    image_pixels = np.expand_dims(image_pixels, axis = 0)
                    image_pixels /= 255
                    predictions = model.predict(image_pixels)
                    max_index = np.argmax(predictions[0])
                    emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    emotion_prediction = emotion_detection[max_index]
                    with placeholder.container():
                        # st.write('Predicted Emotion: ' + emotion_prediction)
                        st.write('Recommended Tracks')
                        container(genreBasedTracks(moodToGenre[emotion_prediction], 3), 'https://open.spotify.com/embed/track/')
                        st.write('Recommended Albums')
                        container(genreBasedAlbums(moodToGenre[emotion_prediction], 3), 'https://open.spotify.com/embed/album/')
                        st.write('Recommended Artists')
                        container(genreBasedArtists(moodToGenre[emotion_prediction], 3), 'https://open.spotify.com/embed/artist/') 
            except:  
                with placeholder.container():
                    st.write('Recommended Tracks')
                    container(genreBasedTracks(['pop', 'hip hop', 'rock', 'folk', 'country', 'r&b'], 3), 'https://open.spotify.com/embed/track/')
                    st.write('Recommended Albums')
                    container(genreBasedAlbums(['pop', 'hip hop', 'rock', 'folk', 'country', 'r&b'], 3), 'https://open.spotify.com/embed/album/')
                    st.write('Recommended Artists')
                    container(genreBasedArtists(['pop', 'hip hop', 'rock', 'folk', 'country', 'r&b'], 3), 'https://open.spotify.com/embed/artist/') 
        
with tab3:  
    st.subheader('Audio Based Recommendation') 
    col1, col2 = st.columns((1.5,2.5))
    with col2:
        placeholder = st.empty()
        with placeholder.container():
            st.write('Recommended Tracks')
            container(genreBasedTracks(['pop', 'hip hop', 'rock', 'folk', 'country', 'r&b'], 3), 'https://open.spotify.com/embed/track/')
            st.write('Recommended Albums')
            container(genreBasedAlbums(['pop', 'hip hop', 'rock', 'folk', 'country', 'r&b'], 3), 'https://open.spotify.com/embed/album/')
            st.write('Recommended Artists')
            container(genreBasedArtists(['pop', 'hip hop', 'rock', 'folk', 'country', 'r&b'], 3), 'https://open.spotify.com/embed/artist/') 
    with col1:
        wav_audio_data = st_audiorec()
        if wav_audio_data!=None:
            data, samplerate = sf.read(io.BytesIO(wav_audio_data))
            sf.write('./data/audio_based/input.wav', data, 16000, 'PCM_24')
            y, s = librosa.load('./data/audio_based/input.wav', sr=16000)
            sf.write('./data/audio_based/input.wav', y, s, 'PCM_24')
            test = extract_feature('./data/audio_based/input.wav', mfcc=True, chroma=True, mel=True)
            pred = mlp.predict([test])[0]
            with placeholder.container():
                # st.write('Predicted Emotion: ' + pred)
                st.write('Recommended Tracks')
                container(genreBasedTracks(moodToGenre[pred], 3), 'https://open.spotify.com/embed/track/')
                st.write('Recommended Albums')
                container(genreBasedAlbums(moodToGenre[pred], 3), 'https://open.spotify.com/embed/album/')
                st.write('Recommended Artists')
                container(genreBasedArtists(moodToGenre[pred], 3), 'https://open.spotify.com/embed/artist/') 
 