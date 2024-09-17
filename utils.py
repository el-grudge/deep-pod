import spacy
from sentence_transformers.util import pytorch_cos_sim
import requests
import os
import concurrent.futures
from urllib.parse import urlparse
import feedparser

nlp = spacy.load('en_core_web_sm')

def remove_punctuation(text):
    doc = nlp(text)
    return ' '.join([token.text.replace('.','').lower() for token in doc if not token.is_punct])

def get_episode_title(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    podcast_id = path.split('/')[-1][2:]
    episode_title = path.split('/')[3].replace('-', ' ')
    return podcast_id, episode_title

def get_podcast_details(podcast_id):
    url = f'https://itunes.apple.com/lookup?id={podcast_id}'
    response = requests.get(url)
    if response.status_code == 200:
        podcast_details = response.json().get('results', [])[0]
        podcast_details['status'] = 'Success'
    else:
        podcast_details = {}
        podcast_details['status'] = 'Fail'
    return podcast_details

def get_feed_details(sentence_encoder, feed_url):
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

def search_for_episode(sentence_encoder, episode_title, feed_details):
    """search for episode """
    query_vector = sentence_encoder.encode(remove_punctuation(episode_title)).tolist()

    [d.update({'cos_sim': pytorch_cos_sim(d['title_vector'], query_vector)}) for d in feed_details]

    return max(feed_details, key=lambda x: x['cos_sim'])

def search_podcasts(term):
    url = 'https://itunes.apple.com/search'
    params = {
        'term': term,
        'media': 'podcast',
        'limit': 3
    }
    response = requests.get(url, params=params)
    if response.status_code == 200 and response.json().get('resultCount') > 0:
        found_podcasts = {'podcasts': response.json().get('results', [])}
        found_podcasts['status'] = 'Success'
    else:
        found_podcasts = {}
        found_podcasts['status'] = 'Fail'
    return found_podcasts

def fetch_latest_episode(feed_url):
    feed = feedparser.parse(feed_url)
    if feed.entries:
        latest_episode = feed.entries[0]

        feed_dict = {
            'title': latest_episode['title'],
            'summary': latest_episode['summary'],
            'published_date': latest_episode['published'],
            'audio_urls': latest_episode.enclosures[0]['href'],
            'status': 'Success'
        }

    else:
        feed_dict = {'status': 'Fail'}
    return feed_dict

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