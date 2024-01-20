import os
import multiprocessing
import json
import mimetypes
import urllib3
import aiofiles
import tempfile
import re
import asyncio
import aiohttp
import uuid
import logging
import string
import warnings
import numpy as np
import syncedlyrics
import numpy as np
import torch
import time
import string
import warnings
import signal
import sys
import spacy
import nltk

from typing import List, Tuple, Dict, Any
from itertools import product
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from m3u8 import M3U8
from syncedlyrics import search
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, WordNetLemmatizer, ngrams
from bertopic import BERTopic
from quart import Quart, request, redirect, url_for, render_template, flash, jsonify, Response
import gunicorn
import hypercorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Levenshtein import ratio
from bertopic import BERTopic
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from syncedlyrics import search
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
from gensim.models import Word2Vec

#nltk.download("vader_lexicon")
#nltk.download("punkt")
#nltk.download("stopwords")
#nltk.download('wordnet')

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Initialization
GENIUS_API_KEY = "6IJtS4Xta8IPcEPwmC-8YVOXf5Eoc4RHwbhWINDbzomMcFVXQVxbVQapsFxzKewr"
APPLE_MUSIC_API_KEY = "eyJhbGciOiJFUzI1NiIsImtpZCI6IjYyMlcyTVVVV1EiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJVNEdMUUdGTlQzIiwiaWF0IjoxNjk3MjQ4NDQ4LCJleHAiOjE3MTAyMDg0NDh9.XMe-WEuuAJS_LOirXG6yU8CZW1RL6Lw4cwxhc405rvZm_LesEsaLoqNnZ9l_n3SQ0eOqUQEsWXEPNZYJ5wdZXw"

headers = {"Authorization": "Bearer " + APPLE_MUSIC_API_KEY}
warnings.filterwarnings("ignore", category=FutureWarning)
# Create the executor at the module level
executor = ThreadPoolExecutor(max_workers=4)

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
def name_topics(topic_model, topics, n_top_words=10):
    topic_names = {}
    for topic in set(topics):
        # Get the top words for this topic
        top_words = topic_model.get_topic(topic)
        # Use spaCy for NER
        doc = nlp(" ".join(word for word, _ in top_words[:n_top_words]))
        named_entities = [ent.text for ent in doc.ents]
        # If we have any named entities, use them as the topic name
        if named_entities:
            topic_names[topic] = " ".join(named_entities[:2]) + f" {topic}"
        else:
            # Otherwise, use the top 3 words
            topic_names[topic] = " ".join(word for word, _ in top_words[:2]) + f" {topic}"
    return topic_names


# Specify the model for embeddings and tokenization
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

'''
def fit_and_update_topics(docs):
    # Define CountVectorizer
    vectorizer_model = CountVectorizer(ngram_range=(1, 2))
    # Train the model
    topic_model = BERTopic(nr_topics=None, min_topic_size=2, vectorizer_model=vectorizer_model)
    # Check if the number of documents is above a certain threshold
    if len(docs) > 11:  # Set this to a value that you consider as a "decent amount"
        topic_model.nr_topics = 5
    topics, _ = topic_model.fit_transform(docs)
    # Get the current number of topics
    current_nr_topics = len(set(topics))  # No subtraction to exclude the outlier topic (-1)
    # If the number of topics is more than 6, reduce it
    if current_nr_topics > 7:
        topic_model.update_topics(docs, topics, top_n_words=15)
    # If the number of topics is less than 3, retrain the model with a smaller min_topic_size
    elif current_nr_topics < 4:
        topic_model = BERTopic(min_topic_size=2, vectorizer_model=vectorizer_model)
        topics, _ = topic_model.fit_transform(docs)
    return topic_model, topics
'''


def fit_and_update_topics(docs):
    # Preprocess the documents
    preprocessed_docs = [preprocess_text(doc) for doc in docs]

    # Train a Word2Vec model
    model = Word2Vec(preprocessed_docs, min_count=1, sg=1)  # sg=1 means use Skip-Gram

    # Use the trained model to get the vector of each word in each document
    # Then average them to get a vector for each document
    vectorized_docs = [np.mean([model.wv[word] for word in doc if word in model.wv], axis=0) for doc in preprocessed_docs]

    # Remove any documents that could not be vectorized
    vectorized_docs = [doc for doc in vectorized_docs if doc.shape == (model.vector_size,)]

    # Convert vectorized_docs to a numpy array
    vectorized_docs = np.array(vectorized_docs)

    # Train the BERTopic model
    topic_model = BERTopic(nr_topics=None, min_topic_size=2)
    topics, _ = topic_model.fit_transform(docs, embeddings=vectorized_docs)

    # Get the current number of topics
    current_nr_topics = len(set(topics))  # No subtraction to exclude the outlier topic (-1)
    # If the number of topics is more than 6, reduce it
    if current_nr_topics > 7:
        topic_model.update_topics(docs, topics, top_n_words=10)
    # If the number of topics is less than 4, retrain the model with a smaller min_topic_size
    elif current_nr_topics < 4:
        topic_model = BERTopic(min_topic_size=2)
        topics, _ = topic_model.fit_transform(docs, embeddings=vectorized_docs)
    return topic_model, topics

'''
def preprocess_text(text, use_stopwords=False, use_ngrams=True, ngram_range=(1, 2), keep_pos_tags=None):
    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english')) if use_stopwords else set()
    lemmatizer = WordNetLemmatizer()
    # Tokenize and lemmatize text
    tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text) if word.isalpha()]
    # Optionally filter by POS tags
    if keep_pos_tags:
        tagged_tokens = nltk.pos_tag(tokens)
        tokens = [word for word, tag in tagged_tokens if tag in keep_pos_tags]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Create n-grams
    if use_ngrams and ngram_range != (1, 2):
        ngram_tokens = ['_'.join(ng) for n in range(ngram_range[0], ngram_range[1] + 1) for ng in ngrams(tokens, n)]
        tokens = ngram_tokens
    return ' '.join(tokens)
'''

def preprocess_text(text, use_stopwords=False, use_ngrams=False, ngram_range=(1, 2), keep_pos_tags=None):
    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english')) if use_stopwords else set()
    lemmatizer = WordNetLemmatizer()
    # Tokenize and lemmatize text
    tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text) if word.isalpha()]
    # Optionally filter by POS tags
    if keep_pos_tags:
        tagged_tokens = nltk.pos_tag(tokens)
        tokens = [word for word, tag in tagged_tokens if tag in keep_pos_tags]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens









# Function to create a BERTopic model with given parameters
def create_bertopic_model(embedding_model, umap_params, hdbscan_params):
    umap_model = UMAP(**umap_params)
    hdbscan_model = HDBSCAN(**hdbscan_params)
    topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model)
    return topic_model


async def process_annotations(song_id: str, api_key: str, lyrics_data: str, song_duration: int, grace_period: int =15) -> Tuple[Any, Dict, List[Dict], Dict]:
    logging.info("Starting to process annotations for song_id: %s", song_id)
    song_duration = round(song_duration, 2)
    start_time_1 = time.time()
    best_umap_params = {'n_neighbors': 7, 'n_components': 1, 'min_dist': 0.01, 'metric': 'manhattan'}
    best_hdbscan_params = {'min_cluster_size': 2, 'metric': 'euclidean', 'cluster_selection_method': 'leaf', 'min_samples': 1}
    try:
        annotations_data = await get_referents_and_annotations(song_id, api_key, lyrics_data, 30)
        logging.info("Annotations data retrieved successfully.")
    except Exception as e:
        logging.error(f"Failed to get referents and annotations: {e}")
        return None, None, None, None
    if len(annotations_data) < 4:
        return None, annotations_data, annotations_data, None
    annotation_texts = [" ".join(preprocess_text(annotation["annotation"])) for annotation in annotations_data]
    topic_model, topics = fit_and_update_topics(annotation_texts)
    # Name the topics
    topic_names = name_topics(topic_model, topics)
    batch_size = 8
    logging.info("Starting batched summarizations")
    start_time_4 = time.time()
    grouped_annotations = {}
    for annotation, topic in zip(annotations_data, topics):
        grouped_annotations.setdefault(topic, []).append(annotation)
    topic_summaries = []
    used_timestamps = [0, song_duration]
    minimum_length = 75
    maximum_length = 350
    penalty_factor = 2.5
    beam_count = 4
    for topic, annotations in grouped_annotations.items():
        logging.info(f"Processing topic: {topic}")
        combined_texts = [" ".join([annotation['annotation'] for annotation in annotations])]
        try:
            summaries = await summarize_text(combined_texts, minimum_length, maximum_length, penalty_factor, beam_count)
            logging.info(f"Summaries for topic {topic} completed successfully.")
        except Exception as e:
            logging.error(f"Error during summarization of topic {topic}: {e}")
            continue
        annotations.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('-inf'))
        timestamp = find_next_available_timestamp(annotations, used_timestamps, grace_period, song_duration)
        topic_summary = {
            "id": str(uuid.uuid4()),
            "annotation": " ".join(summaries),
            "lyric": f"Topic {topic}",
            "timestamp": timestamp
        }
        logging.info(f"Time for a batch of summaries {time.time() - start_time_4 :.2f} seconds.")
        topic_summaries.append(topic_summary)
    topic_summaries = sorted(topic_summaries, key=lambda x: x['timestamp'])
    logging.info(f"Processing annotations took: {time.time() - start_time_1 :.2f} seconds.")
    return topic_model, grouped_annotations, topic_summaries, topic_names


def find_next_available_timestamp(annotations: List[Dict], used_timestamps: List[int], grace_period: int, song_duration: int) -> int:
    """
    Find the next available timestamp that is not within the grace period of any used timestamp.
    Parameters:
    annotations (List[Dict]): The list of annotations.
    used_timestamps (List[int]): The list of used timestamps.
    grace_period (int): The grace period in seconds.
    song_duration (int): The duration of the song in seconds.
    Returns:
    int: The next available timestamp.
    """
    song_start = 0  # The start of the song in seconds
    song_end = song_duration  # The end of the song in seconds
    timestamp = annotations[0]['timestamp'] if annotations else None
    if timestamp is not None:
        # Check if the timestamp is within the grace period of any used timestamp
        if any(abs(timestamp - other_timestamp) < grace_period for other_timestamp in used_timestamps):
            # Look for the next available timestamp outside the grace period
            for next_annotation in annotations[1:]:
                next_timestamp = next_annotation['timestamp']
                if next_timestamp is not None and all(abs(next_timestamp - other_timestamp) >= grace_period for other_timestamp in used_timestamps):
                    timestamp = next_timestamp
                    break
            else:
                # No available timestamp found, find the midpoint of the largest gap
                timestamp = find_midpoint_of_largest_gap(used_timestamps, song_start, song_end)
    else:
        # No initial timestamp, find the midpoint of the largest gap
        timestamp = find_midpoint_of_largest_gap(used_timestamps, song_start, song_end)
    # Add the new timestamp to the list of used timestamps
    used_timestamps.append(timestamp)
    used_timestamps.sort()  # Keep the list sorted for future calls
    return timestamp


def find_midpoint_of_largest_gap(used_timestamps: List[int], song_start: int, song_end: int) -> int:
    """
    Find the midpoint of the largest gap in the list of used timestamps, including the start and end of the song.
    Parameters:
    used_timestamps (List[int]): The list of used timestamps.
    song_start (int): The start of the song in seconds.
    song_end (int): The end of the song in seconds.
    Returns:
    int: The midpoint of the largest gap.
    """
    # Include the start and end of the song in the list of timestamps
    extended_timestamps = [song_start] + sorted(used_timestamps) + [song_end]
    # Find the largest gap
    gaps = [(extended_timestamps[i+1] - extended_timestamps[i], i) for i in range(len(extended_timestamps)-1)]
    largest_gap = max(gaps, key=lambda x: x[0])
    # Return the midpoint of the largest gap
    return extended_timestamps[largest_gap[1]] + largest_gap[0] // 2


async def summarize_text(input_texts: List[str], minimum_length: int = 75, maximum_length: int = 350, penalty_factor: float = 2.5, beam_count: int = 4) -> List[str]:
    """
    Summarize a list of input texts using the BART CNN summarization model.
    """
    summaries = []
    input_texts = [text.strip().replace("\n", " ") for text in input_texts]
    tokenized_texts = tokenizer(input_texts, truncation=True, padding=True, return_tensors="pt")
    try:
        summary_ids = await asyncio.to_thread(
            model.generate,
            tokenized_texts["input_ids"],
            num_beams=beam_count,
            no_repeat_ngram_size=3,
            length_penalty=penalty_factor,
            min_length=minimum_length,
            max_length=maximum_length,
            early_stopping=True
        )
        summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
    except Exception as e:
        print(f"Error in batch summarization: {e}")
    return summaries


# Genius API Interactions for Song ID
async def get_song_id(search_term, artist_name, api_key):
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": f"{search_term} {artist_name}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            # Check if the request was successful
            if response.status != 200:
                print(f"Failed to get song ID. HTTP status code: {response.status}")
                return None
            response_json = await response.json()
            # Check if the response contains the expected keys
            if "response" not in response_json or "hits" not in response_json["response"] or not response_json["response"]["hits"]:
                print(f"No results found for '{search_term} by {artist_name}'. Please try again with a different search term.")
                return None
            # Return the first song's ID
            song_id = response_json["response"]["hits"][0]["result"]["id"]
            return song_id


# Compile regex pattern for reuse
punctuation_pattern = re.compile(r'\s([?.!",](?:\s|$))')
def parse_description(description):
    readable_description = []
    def parse_element(element):
        if isinstance(element, str):
            readable_description.append(element.strip())
        elif isinstance(element, dict):
            if 'children' in element:
                for child in element['children']:
                    parse_element(child)
            elif 'tag' in element and element['tag'] == 'a':
                if 'children' in element:
                    for child in element['children']:
                        parse_element(child)
                else:
                    readable_description.append(element.get('text', '').strip())
    for item in description:
        parse_element(item)
    full_description = ' '.join(filter(None, readable_description))
    full_description = punctuation_pattern.sub(r'\1', full_description)
    return full_description


async def get_song_details(song_id, api_key):
    url = f"https://api.genius.com/songs/{song_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                print(f"Failed to get song details. HTTP status code: {response.status}")
                return None
            response_json = await response.json()
            if "response" not in response_json or "song" not in response_json["response"]:
                print("Unexpected response format from Genius API.")
                return None
            song_details = response_json["response"]["song"]
            if "description" not in song_details or "dom" not in song_details["description"] or "children" not in song_details["description"]["dom"]:
                print("Unexpected song details format from Genius API.")
                return None
            # Parse the description
            song_description_dom = song_details["description"]["dom"]["children"]
            song_description = parse_description(song_description_dom)
            song_details["description"] = song_description
            return song_details


async def get_referents_and_annotations(song_id, api_key, lyrics_data, limit=25):
    try:
        url = f"https://api.genius.com/referents?song_id={song_id}&text_format=plain"
        headers = {"Authorization": f"Bearer {api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    print(f"Error fetching referents: HTTP Status Code {response.status}")
                    return []
                response_json = await response.json()
                referents = response_json["response"]["referents"]
        processed_annotations = []

        for referent in referents:
            fragment = referent["fragment"]
            annotations = referent["annotations"]
            for annotation in annotations:
                if "body" in annotation and "plain" in annotation["body"]:
                    annotation_text = annotation["body"]["plain"]
                else:
                    continue  # Skip if plain text is not available
                #print(f"Processing fragment: {fragment}")
                timestamp = find_matching_lyric_timestamp(fragment, lyrics_data, threshold=0.4)
                #print(f"Matched timestamp: {timestamp}")
                processed_annotation = {
                    "fragment": fragment,
                    "annotation": annotation_text,
                    "timestamp": timestamp
                }
                processed_annotations.append(processed_annotation)
        # Limit the number of annotations if necessary
        processed_annotations = processed_annotations[:min(limit, len(processed_annotations))]
        return processed_annotations
    except Exception as e:
        print(f"Error in get_referents_and_annotations: {e}")
        return []


def find_matching_lyric_timestamp(fragment, lyrics_data, threshold=0.4, last_timestamp=None):
    # Extract the list of lyrics from the lyrics_data dictionary
    lyrics_list = [lyric_entry for lyric_entry in lyrics_data['lyrics_with_timestamps'] if 'lyric' in lyric_entry]
    # Sort the lyrics list by timestamp
    lyrics_list.sort(key=lambda x: x['timestamp'])
    # Start from the last timestamp if provided
    if last_timestamp is not None:
        lyrics_list = [lyric_entry for lyric_entry in lyrics_list if lyric_entry['timestamp'] > last_timestamp]
    # Compute the Levenshtein ratio between the fragment and each lyric
    levenshtein_ratios = [ratio(fragment, lyric_entry['lyric']) for lyric_entry in lyrics_list]
    # Find the index of the best match
    best_match_index = np.argmax(levenshtein_ratios)
    # Check if the best match score is above the threshold
    if levenshtein_ratios[best_match_index] >= threshold:
        return lyrics_list[best_match_index]['timestamp']
    return None  # Return None if no suitable match is found


async def fetch_lyrics_with_syncedlyrics(artist_name, track_name):
    # Search for the synced lyrics
    lrc = syncedlyrics.search(f"{track_name} {artist_name}")
    # Process the lyrics
    lyrics_data = None
    if lrc:
        parsed_lyrics = [
            {
                "id": str(uuid.uuid4()),  # Add an "id" key with a unique UUID
                "lyric": l,  # 'lyric' comes after 'id'
                "timestamp": round(
                    float(ts[1:].split(":")[0]) * 60 + float(ts[1:].split(":")[1]), 1
                )  # 'timestamp' comes after 'lyric'
            }
            for line in lrc.split("\n")
            if line and "] " in line and len(line.split("] ")) == 2
            for ts, l in [line.split("] ")]
        ]
        lyrics_data = {"lyrics_with_timestamps": parsed_lyrics}
    # Asynchronous file operation (if applicable)
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        temp_name = temp.name
        async with aiofiles.open(temp_name, "w") as file:
            await file.write(lrc)
    return lyrics_data


# Utility Functions
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


async def get_webpage_content(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.ClientError as e:
        print(f"Error occurred while fetching page: {e}")
        return None


async def fetch_variant_playlist_url(playlist_url):
    content = await get_webpage_content(playlist_url)
    if content:
        playlists = M3U8(content).playlists
        if playlists:
            playlists.sort(key=lambda p: abs(p.stream_info.resolution[0] - 720))
            return urljoin(playlist_url, playlists[0].uri)
    print("No variant playlist found.")
    return None


async def fetch_playlist_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.text()
                match = re.search(r'src="h([^"]*)', content)
                if match:
                    return "h" + match.group(1)
            # print("No video URL found.")
            return None


async def fetch_segment_urls(variant_playlist_url):
    content = await get_webpage_content(variant_playlist_url)
    if content:
        return [
            urljoin(variant_playlist_url, segment.uri)
            for segment in M3U8(content).segments
        ]
    return None


# Functions for Apple Music interactions
async def download_image(image_url, image_path):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status == 200:
                # Create the finalOutput directory if it doesn't exist
                output_dir = os.path.join(os.getcwd(), "finalOutput")
                os.makedirs(output_dir, exist_ok=True)
                # Save the image in the finalOutput directory asynchronously
                async with aiofiles.open(os.path.join(output_dir, image_path), mode='wb') as out_file:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        await out_file.write(chunk)


async def search_song(song_title, artist_name, developer_token):
    headers = {"Authorization": "Bearer " + developer_token}
    params = {"term": song_title + " " + artist_name, "limit": "5", "types": "songs"}
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.music.apple.com/v1/catalog/us/search", headers=headers, params=params) as response:
            json_response = await response.json()
    # Check if any song data is returned
    if "songs" not in json_response["results"]:
        print("No songs found.")
        return None, None, None
    song_data = json_response["results"]["songs"]["data"]
    bg_color, text_colors, song_duration = None, None, None
    # Fetch_playlist_url, fetch_variant_playlist_url, fetch_segment_urls are also async
    for song in song_data:
        song_url = song["attributes"]["url"]
        playlist_url = await fetch_playlist_url(song_url)
        if playlist_url:
            variant_playlist_url = await fetch_variant_playlist_url(playlist_url)
            if variant_playlist_url:
                segment_urls = await fetch_segment_urls(variant_playlist_url)
                if segment_urls:
                    await download_video_segments(segment_urls, "video_segments")
                    break  # Stop once a video is downloaded
    # Assuming download_image is also async
    artwork_url = song_data[0]["attributes"]["artwork"]["url"].replace("{w}", "3000").replace("{h}", "3000")
    await download_image(artwork_url, "artwork.jpg")
    bg_color = song_data[0]["attributes"]["artwork"]["bgColor"]
    text_colors = {
        "textColor1": song_data[0]["attributes"]["artwork"]["textColor1"],
        "textColor2": song_data[0]["attributes"]["artwork"]["textColor2"],
        "textColor3": song_data[0]["attributes"]["artwork"]["textColor3"],
        "textColor4": song_data[0]["attributes"]["artwork"]["textColor4"],
    }
    song_duration = song_data[0]["attributes"]["durationInMillis"]
    return bg_color, text_colors, song_duration


async def download_video_segments(segment_urls, video_dir):
    output_dir = os.path.join(os.getcwd(), "finalOutput")
    os.makedirs(output_dir, exist_ok=True)
    segment_url = segment_urls[0]  # Get the first segment URL
    async with aiohttp.ClientSession() as session:
        async with session.get(segment_url) as response:
            if response.status == 200:
                try:
                    content = await response.read()
                    async with aiofiles.open(os.path.join(output_dir, f"AnimatedArt.mp4"), "wb") as file:
                        await file.write(content)
                    print("AnimatedArt downloaded.")
                except aiohttp.client_exceptions.ClientPayloadError:
                    print("The response payload was not fully received.")
            else:
                print(f"No AnimatedArt. Status code: {response.status}")


# Create a global variable for the client session
http_session = None
app = Quart(__name__)

@app.route('/process_song', methods=['POST'])
async def process_song():
    try:
        start_time_f = time.time()
        print(" process_song request received")

        # Delete old files in the output directory at the start of each request
        output_dir = 'finalOutput'
        for filename in ['final_1.json', 'AnimatedArt.mp4', 'artwork.jpg']:
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")
        data = await request.get_json()
        song_title = data.get('song_title')
        artist_name = data.get('artist_name')
        print(f"Processing: song_title: {song_title}, artist_name: {artist_name}")

        # Get the song ID from Genius API
        song_id = await get_song_id(song_title, artist_name, GENIUS_API_KEY)
        if song_id is None:
            print("Error: Could not find the song ID")
            return {"error": "Could not find the song ID."}, 400
        print(f"Genius song ID: {song_id}")

        # Get the song details from Genius API
        song_details = await get_song_details(song_id, GENIUS_API_KEY)

        # Fetch lyrics and process annotations
        lyrics_data = await fetch_lyrics_with_syncedlyrics(artist_name, song_title)
        if lyrics_data is None:
            print("Error: Could not fetch the lyrics")
            return {"error": "Could not fetch the lyrics."}, 400

        # Get referents and annotations
        referents_and_annotations = await get_referents_and_annotations(song_id, GENIUS_API_KEY, lyrics_data)
        if not referents_and_annotations:
            print("Error: Could not get the referents and annotations")
            return {"error": "Could not get the referents and annotations."}, 400

        # Search for the song on Apple Music
        bg_color, text_colors, song_duration = await search_song(song_title, artist_name, APPLE_MUSIC_API_KEY)
        # Convert song_duration from milliseconds to seconds
        song_duration /= 1000

        # Process annotations
        topic_model, grouped_annotations, topic_summaries, topic_names = await process_annotations(song_id, GENIUS_API_KEY, lyrics_data, song_duration)
        logging.info("process_annotations function completed.")   

        # Print out the topic names
        for topic in topic_names:
            if topic != -1:
                print(f"Topic {topic}: {topic_names[topic]}")

        # Additional debugging information
        # Flatten the lyrics_with_timestamps
        flattened_lyrics_with_timestamps = lyrics_data["lyrics_with_timestamps"]

        # Handling the possibility of 'album' being None
        album_name = song_details.get("album", {}).get("name", "") if song_details and song_details.get("album") else ""
        print(f"album_name: {album_name}")
        # Convert song_duration from seconds to milliseconds and round to the nearest integer
        song_duration = round(song_duration * 1000)

        # Generate final JSON
        final_1 = {
            "title": song_details.get("title", ""),
            "artist": song_details.get("primary_artist", {}).get("name", ""),
            "album": song_details.get("album", {}).get("name", ""),
            "release_date": song_details.get("release_date", ""),
            "description": song_details.get("description", ""),
            "bgColor": bg_color,
            "textColors": text_colors,
            "songDuration": song_duration,
            "lyrics_with_timestamps": flattened_lyrics_with_timestamps,
            "annotations_with_timestamps": topic_summaries,
        }

        # Ensure that final_1 does not contain any coroutines
        final_1 = {key: await value if asyncio.iscoroutine(value) else value for key, value in final_1.items()}

        # Serialize and write to file
        json_file_path = os.path.join(output_dir, 'final_1.json')
        async with aiofiles.open(json_file_path, 'w') as json_file:
            await json_file.write(json.dumps(final_1))

        # Check if other files exist and read them into memory asynchronously
        files = {}
        for filename in ['final_1.json', 'AnimatedArt.mp4', 'artwork.jpg']:
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'rb') as file:
                    files[filename] = await file.read()
            else:
                print(f"Warning: File {filename} not found in {output_dir}")

        # Create multipart response
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        response = Response(mimetype='multipart/form-data; boundary=' + boundary)

        # Initialize response_data as a bytes object
        response_data = b""
        for filename, filedata in files.items():
            mimetype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        
            # Convert string parts to bytes and concatenate
            response_data += f"--{boundary}\r\n".encode('utf-8')
            response_data += f"Content-Disposition: form-data; name=\"{filename}\"; filename=\"{filename}\"\r\n".encode('utf-8')
            response_data += f"Content-Type: {mimetype}\r\n\r\n".encode('utf-8')
            # Append the binary data directly
            response_data += filedata
            response_data += b"\r\n"

        # Final boundary
        response_data += f"--{boundary}--\r\n".encode('utf-8')

        # Set the response data
        response.set_data(response_data)
        print(f"Sent response")
        
        print(f"time elapsed: {time.time() - start_time_f :.2f} seconds")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        return {'error': 'An internal server error occurred.'}, 500

    return response



@app.route('/')
async def index():
    return 'Welcome to the Quart Server!'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)