import spacy
from sentence_transformers import SentenceTransformer, util
# from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
import requests
import os
import concurrent.futures
from urllib.parse import urlparse
import feedparser

sentence_encoder = SentenceTransformer("sentence-transformers/sentence-t5-base")
nlp = spacy.load('en_core_web_sm')
# es_client = Elasticsearch('http://localhost:9200')

def get_encoder():
    return sentence_encoder

def get_episode_title(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    podcast_id = path.split('/')[-1][2:]
    episode_title = path.split('/')[3].replace('-', ' ')
    return podcast_id, episode_title

def get_episode_details(podcast_id):
    url = f'https://itunes.apple.com/lookup?id={podcast_id}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('results', [])[0]
    else:
        return "Failed to fetch podcast details. Please try again."

def remove_punctuation(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Create a list of tokens without punctuation
    return ' '.join([token.text.replace('.','').lower() for token in doc if not token.is_punct])

def get_feed_details(feed_url):
    # parse_podcast_feed
    feed = feedparser.parse(feed_url)

    docuemnts = []
    titles = [remove_punctuation(episode['title']) for episode in feed.entries]

    # Batch encode the titles
    title_vectors = sentence_encoder.encode(titles, batch_size=32).tolist()  # You can adjust batch_size for your system

    # Iterate through feed.entries and use precomputed vectors
    for i, episode in enumerate(feed.entries):
        feed_dict = {
            'title': episode['title'],
            'summary': episode['summary'],
            'published_date': episode['published'],
            'audio_urls': episode.enclosures[0]['href'],
            'title_vector': title_vectors[i]
        }

        docuemnts.append(feed_dict)

    return docuemnts

def search_for_episode(episode_title, feed_details):
    """search for episode """
    query_vector = sentence_encoder.encode(remove_punctuation(episode_title)).tolist()

    [d.update({'cos_sim': util.pytorch_cos_sim(d['title_vector'], query_vector)}) for d in feed_details]

    return max(feed_details, key=lambda x: x['cos_sim'])

def search_podcasts(term):
    url = 'https://itunes.apple.com/search'
    params = {
        'term': term,
        'media': 'podcast',
        'limit': 3
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        return "Failed to search for podcasts. Please try again."

def get_podcast_details(podcast_id):
    url = f'https://itunes.apple.com/lookup?id={podcast_id}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('results', [])[0]
    else:
        return f"Error: {response.status_code} - {response.text}"

def fetch_latest_episode(feed_url):
    feed = feedparser.parse(feed_url)
    if feed.entries:
        latest_episode = feed.entries[0]

        feed_dict = {
            'title': latest_episode['title'],
            'summary': latest_episode['summary'],
            'published_date': latest_episode['published'],
            'audio_urls': latest_episode.enclosures[0]['href']
        }

        return feed_dict
    else:
        return "No episodes found in the RSS feed."

def download_audio_file(audio_url, file_name):
    """Download the audio file from the given URL."""
    response = requests.get(audio_url, stream=True)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Audio file downloaded successfully: {file_name}")
        return file_name
    else:
        print(f"Failed to download audio file. Status code: {response.status_code}")
        return None

def download_all(urls, podcast_name):
    # Create directory
    podcast_dir = os.path.join(".",os.path.join('audio'), podcast_name.replace(" ", "_").replace(":","-").replace('.',''))
    os.makedirs(podcast_dir) if not os.path.exists(podcast_dir) else None     
    
    # Using ThreadPoolExecutor to parallelize downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit the download tasks to the executor
        futures = {executor.submit(download_audio_file, url, os.path.join(podcast_dir, f'episode_{i}.mp3')): os.path.join(podcast_dir, f'episode_{i}.mp3') for i, url in enumerate([urls])}
        # Ensure all tasks are completed
        concurrent.futures.wait(futures)
    return list(futures.values())

# encode
def encode_documents(chunks):

    documents = []

    for sentence in chunks:
        temp_dict = {
            'text': sentence['text'],
            'text_vector': sentence_encoder.encode(sentence["text"]).tolist()
        }
        documents.append(temp_dict)
    
    return documents

# create index 
def create_index(es_client, docs):
    chunks = docs['chunks']
    documents = encode_documents(chunks)
    # Setup elasticsearch
    index_name = "podcast-transcriber"

    # Create mapping
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "text_vector": {"type": "dense_vector", "dims": 768},
            }
        }
    }

    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)

    # add documents 
    for doc in documents:
        try:
            # es_client.index(index=index_name, document=doc)
            es_client.index(index=index_name, body=doc)
        except Exception as e:
            print(e)

    return index_name
