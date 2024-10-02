from streamlit import spinner
from minsearch import Index as minsearch
from utils import get_episode_title, get_podcast_details, get_feed_details, search_for_episode, fetch_latest_episode, download_all
import json
from pydub import AudioSegment
import os
import io
import time

def create_minsearch_index(index_name):
    return minsearch(
        index_name = index_name,
        text_fields = ['text'],
        keyword_fields = ['']
    )

def create_es_index(client, index_name):
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

    client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(index=index_name, body=index_settings)
    
    return client.indices.get_alias(index=index_name)

def create_chroma_index():
    ...

def create_index(**kwargs):
    vector_db = kwargs['vector_db']

    if vector_db == "1. Minsearch":
        return create_minsearch_index(index_name=kwargs['index_name'])
    elif vector_db == "2. Elasticsearch":
        return create_es_index(client=kwargs['vector_db_client'], index_name=kwargs['index_name'])
    elif vector_db == "3. ChromaDB":
        ... # create_chroma_index()

def download_episode_from_url(sentence_encoder, url):
    try:
        podcast_id, episode_title = get_episode_title(url)
        podcast_details = get_podcast_details(podcast_id)
        feed_details = get_feed_details(sentence_encoder, podcast_details['feedUrl'])
        episode_details = search_for_episode(sentence_encoder, episode_title, feed_details)
        if episode_details['cos_sim'] < 0.95:
            raise Exception
        episode_details['filenames'] = []
        episode_details['filenames'] += download_all(episode_details['audio_urls'], podcast_details['collectionName'])
        episode_details['status'] = 'Success'
        episode_details['status_message'] = f"Podcast {podcast_details['collectionName']} downloaded successfully."
        return episode_details
    except Exception:
        episode_details = {}
        episode_details['status'] = 'Fail'
        episode_details['status_message'] = "Failed to download the podcast. Please try again."
        return episode_details

def download_episode_from_name(id, name):
    try:
        podcast_details = get_podcast_details(id)
        episode_details = fetch_latest_episode(podcast_details['feedUrl'])
        episode_details['filenames'] = []
        episode_details['filenames'] += download_all(episode_details['audio_urls'], name)
        episode_details['status'] = 'Success'
        episode_details['status_message'] = f"Podcast {podcast_details['collectionName']} - {name} downloaded successfully."
        return episode_details
    except Exception:
        episode_details = {}
        episode_details['status'] = 'Fail'
        episode_details['status_message'] = "Failed to download the podcast. Please try again."
        return episode_details

def download_podcast(**kwargs):
    option = kwargs['episode_option']
    if option == "1. Try a sample":
        with open('sample/episode_details.json', 'r') as f:
            episode_details = json.load(f)
    elif option == "2. Provide the iTunes URL for a specific podcast episode":
        episode_url = kwargs['episode_url']
        with spinner('Downloading episode...'):
            episode_details = download_episode_from_url(kwargs['sentence_encoder'], episode_url)
    elif option == "3. Provide a name of a podcast to explore its most recent episode":
        with spinner('Downloading episode...'):
            found_podcasts = kwargs['found_podcasts']
            selected_index = kwargs['selected_index']
            episode_details = download_episode_from_name(found_podcasts[selected_index]['collectionId'], found_podcasts[selected_index]['collectionName'])
    return episode_details

def shrink_mp3(mp3_file):
    # Split the file path into directory, base filename, and extension
    directory, filename = os.path.split(mp3_file)
    basename, ext = os.path.splitext(filename)

    # Create a new filename with the suffix '_smaller'
    new_filename = f"{basename}_smaller{ext}"

    # Create the new file path
    new_file_path = os.path.join(directory, new_filename)

    # decrease size
    audio = AudioSegment.from_mp3(mp3_file)
    # Set desired sample rate and bit depth to control size
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(16 // 8)  # 8 bits = 1 byte    
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(new_file_path, format="mp3", bitrate="128k")

    # Read the local audio file in binary mode
    with open(new_file_path, "rb") as f:
        audio_blob = io.BytesIO(f.read())  # Use BytesIO to create a file-like object

    return audio_blob

def transcribe_with_replicate(replicate_client, mp3_file):
    audio_blob = shrink_mp3(mp3_file)

    # create sepaarte function
    start_time = time.time()
    output = replicate_client.run(
        "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        input={
            "task": "transcribe",
            "audio": audio_blob,
            "language": "None",
            "timestamp": "chunk",
            "batch_size": 64,
            "diarise_audio": False
        }
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time to run the command: {execution_time} seconds")
    print(type(output))

    return output

def transcribe_podcast(**kwargs):
    podcast_option = kwargs['episode_option']
    episode_details = kwargs['episode_details']
    
    if podcast_option == "1. Try a sample":
        chunks, text = episode_details['chunks'], episode_details['text']
    else:
        transcription_method = kwargs['transcription_method']
        if transcription_method == "1. Replicate":
            transcript = transcribe_with_replicate(kwargs['transcription_client'], episode_details['filenames'])
        elif transcription_method == "2. Local transcription":
            ...
    
    return {'chunks': chunks, 'text': text}

def create_oa_embedding(client, model, chunks):
    documents = []

    for sentence in chunks:
        temp_dict = {
            'text': sentence['text'],
            'text_vector': client.embeddings.create(model=model, input=sentence['text'])
        }
        documents.append(temp_dict)

    return documents

def create_t5_embedding(encoder, chunks):
    documents = []

    for sentence in chunks:
        temp_dict = {
            'text': sentence['text'],
            'text_vector': encoder.encode(sentence["text"]).tolist()
        }
        documents.append(temp_dict)
    
    return documents

def encode_podcast(**kwargs):
    episode_details = kwargs['episode_details']
    sentence_encoder = kwargs['sentence_encoder']

    if sentence_encoder == "1. T5":
        documents = create_t5_embedding(kwargs['encoder'], episode_details['chunks'])
    elif sentence_encoder == "2. OpenAI":
        documents = create_oa_embedding(kwargs['oa_embedding_client'], kwargs['embedding_model'], episode_details['chunks'])

    return {'documents': documents}

def populate_minsearch_index(docs, index):
    documents = [{'text': doc['text']} for doc in docs]
    index.fit(documents)

def populate_es_index(documents, index_name, client):
    # add documents 
    for doc in documents:
        try:
            client.index(index=index_name, body=doc)
        except Exception as e:
            print(e)

    return index_name

def index_podcast(**kwargs):
    episode_details = kwargs['episode_details']
    vector_db = kwargs['vector_db']

    if vector_db=="1. Minsearch":
        populate_minsearch_index(episode_details['chunks'], kwargs['index'])
    elif vector_db=="2. Elasticsearch":
        populate_es_index(episode_details['chunks'], kwargs['index_name'], kwargs['vector_db_client'])


#################################



# def populate_es_index(chunks, index_name, client):
# # # encode
# # def encode_documents(sentence_encoder, chunks):
#     documents = []

#     for sentence in chunks:
#         temp_dict = {
#             'text': sentence['text'],
#             'text_vector': sentence_encoder.encode(sentence["text"]).tolist()
#         }
#         documents.append(temp_dict)
    
#     # add documents 
#     for doc in documents:
#         try:
#             client.index(index=index_name, body=doc)
#         except Exception as e:
#             print(e)


######################################################################################################

# from sentence_transformers import SentenceTransformer

# sentence_encoder = SentenceTransformer()

# ###################

#     if llm_option == "1. OpenAI":
#             openai_api_key = st.text_input("OpenAI API Key", key="file_oa_api_key", type="password")
#             if openai_api_key != '':
#                 try:
#                     oa_client = OpenAI(api_key=openai_api_key)
#                     response = oa_client.models.list()
#                     update_session(llm_option_selected=True, llm_option=llm_option, llm_client=oa_client)
#                 except:
#                     st.warning("Invalid API key. Please provide a valid API token.")

# #######################


#         client = st.session_state['llm_client']
#         response = client.embeddings.create(
#             model="text-embedding-3-large",
#             input="The food was delicious and the waiter..."
#         )


# #######################


# # encode
# def encode_documents(sentence_encoder, chunks):
#     documents = []

#     for sentence in chunks:
#         temp_dict = {
#             'text': sentence['text'],
#             'text_vector': sentence_encoder.encode(sentence["text"]).tolist()
#         }
#         documents.append(temp_dict)
    
#     return documents

# # encode
# def encode_text(**kwargs):
#     encoder = kwargs['encoder']
#     chunks = kwargs['docs']

#     documents = []

#     for sentence in chunks:
#         temp_dict = {
#             'text': sentence['text'],
#             'text_vector': encoder.encode(sentence["text"]).tolist()
#         }
#         documents.append(temp_dict)
    
#     return documents

# def index_text(**kwargs):
#     vector_db = kwargs['vector_db']

#     if vector_db == "1. Minsearch":
#         documents = [{'text': doc['text']} for doc in docs]

#         index = minsearch(
#             text_fields = ['text'],
#             keyword_fields = ['']
#         )

#         index.fit(documents)

#         return index
#     elif vector_db == "2. Elasticsearch":
#         # add documents 
#         for doc in documents:
#             try:
#                 es_client.index(index=index_name, body=doc)
#             except Exception as e:
#                 print(e)

#         return index_name
#     elif vector_db == "3. ChromaDB":
#         ...





# # create index 
# def create_index(encoder, es_client, docs):
#     chunks = docs['chunks']
#     documents = encode_documents(encoder, chunks)
#     # Setup elasticsearch
#     index_name = "podcast-transcriber"

#     # Create mapping
#     index_settings = {
#         "settings": {
#             "number_of_shards": 1,
#             "number_of_replicas": 0
#         },
#         "mappings": {
#             "properties": {
#                 "text": {"type": "text"},
#                 "text_vector": {"type": "dense_vector", "dims": 768},
#             }
#         }
#     }

#     es_client.indices.delete(index=index_name, ignore_unavailable=True)
#     es_client.indices.create(index=index_name, body=index_settings)

#     # add documents 
#     for doc in documents:
#         try:
#             # es_client.index(index=index_name, document=doc)
#             es_client.index(index=index_name, body=doc)
#         except Exception as e:
#             print(e)

#     return index_name


# Example usage:
# Assume `client` is an initialized API client (e.g., OpenAI client)

# embedding_response = create_embedding(client, model_name="text-embedding-3-large", input_text="The food was delicious and the waiter...")
# if embedding_response:
#     print(embedding_response)
