import streamlit as st
from openai import OpenAI
import replicate
from bs4 import BeautifulSoup
import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from utils import search_podcasts
from ingest import download_episode_from_url, download_episode_from_name, create_index
from helpers import transcribe, list_topics
from rag import rag
import json


def update_session(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v

def transcribe_episode(encoder, episode_details, replicate_client):
    if not st.session_state.get('transcription_complete', False):
        with st.spinner('Transcribing...'):
            output = transcribe(replicate_client, episode_details['filenames'][0])
            episode_details = {**episode_details, **output}
            index_name = create_index(encoder, es_client, episode_details)
            st.session_state['episode_details'] = episode_details
            st.session_state['index_name'] = index_name
            st.session_stated['index_created'] = True
            st.session_state['es_client'] = es_client
            st.session_state['transcription_complete'] = True
            st.session_state['chat_with_your_podcast'] = True
            st.success(f"Transcription complete.")
    return episode_details

if __name__ == "__main__":
    # ask user to provide the necessary api keys
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="file_oa_api_key", type="password")
        replicate_api_key = st.text_input("Replicate API Key", key="file_replicate_api_key", type="password")
        elasticsearch_api_key = st.text_input("Elasticsearch API Key", key="file_es_api_key", type="password")
        elasticsearch_cloud_id = st.text_input("Elasticsearch Cloud ID", key="file_es_cloud_id", type="password")
        
        st.session_state['openai_api_key_provided'] = True if openai_api_key != '' else False
        st.session_state['replicate_api_key_provided'] = True if replicate_api_key != '' else False
        st.session_state['elasticsearch_api_key_provided'] = True if elasticsearch_api_key != '' else False
        st.session_state['elasticsearch_cloud_id_provided'] = True if elasticsearch_cloud_id != '' else False

    st.title("Podcast Search and Download Tool")

    if st.session_state['openai_api_key_provided'] and st.session_state['replicate_api_key_provided'] and st.session_state['elasticsearch_api_key_provided'] and st.session_state['elasticsearch_cloud_id_provided']:
        # if keys are provided create clients and start the app
        oa_client = OpenAI(api_key=openai_api_key)
        replicate_client = replicate.Client(api_token=replicate_api_key)
        es_client = Elasticsearch(cloud_id=elasticsearch_cloud_id, api_key=elasticsearch_api_key)
        sentence_encoder = SentenceTransformer("sentence-transformers/sentence-t5-base")
        
        option = st.radio(
            "Choose an option:",
            ("1. Try a sample",
             "2. Provide the iTunes URL for a specific podcast episode",
             "3. Provide a name of a podcast to explore its most recent episode"),
            index=None,
        )

        if 'current_option' not in st.session_state or st.session_state['current_option'] != option:
            st.session_state['current_option'] = option
            st.session_state['podcast_downloaded'] = False
            st.session_state['transcription_complete'] = False
            st.session_state['chat_with_your_podcast'] = False
            if 'episode_details' in st.session_state:
                del st.session_state['episode_details']
            if 'index_name' in st.session_state:
                del st.session_state['index_name']
            if 'es_client' in st.session_state:
                del st.session_state['es_client']
            if 'messages' in st.session_state:
                del st.session_state['messages']

        if option == "1. Try a sample":
            with open('sample/episode_details.json', 'r') as f:
                episode_details = json.load(f)
            with st.spinner("Preparing episode..."):
                if not st.session_state.get('index_created', False):
                    index_name = create_index(sentence_encoder, es_client, episode_details)
                    update_session(index_name=index_name, index_created=True)
                podcast_option=None
                user_input=None
                if episode_details['status'] == 'Success':
                    st.success(episode_details['status_message'])
                    st.session_state['episode_details'] = episode_details
                    st.session_state['podcast_downloaded'] = True
                    st.session_state['transcription_complete'] = False
                else:
                    st.warning(episode_details['status_message'])
                    st.session_state['podcast_downloaded'] = False

        elif option == "2. Provide the iTunes URL for a specific podcast episode":
            episode_url = st.text_input("Enter the iTunes URL of the episode you want:")
            if st.button("Search Episode"):
                with st.spinner('Downloading episode...'):
                    episode_details = download_episode_from_url(sentence_encoder, episode_url)
                    if episode_details['status'] == 'Success':
                        st.success(episode_details['status_message'])
                        st.session_state['episode_details'] = episode_details
                        st.session_state['podcast_downloaded'] = True
                        st.session_state['transcription_complete'] = False
                        st.session_stated['index_created'] = False
                    else:
                        st.warning(episode_details['status_message'])
                        st.session_state['podcast_downloaded'] = False

        elif option == "3. Provide a name of a podcast to explore its most recent episode":
            term = st.text_input("Enter a search term for podcasts:")
            if st.button("Search Podcasts"):
                print(term)
                try:
                    found_podcasts = search_podcasts(term)
                    if found_podcasts['status'] == 'Fail':
                        raise Exception
                    st.session_state['found_podcasts'] = found_podcasts['podcasts']
                except Exception:
                    if 'found_podcasts' in st.session_state:
                        del st.session_state['found_podcasts']
                    st.warning("Please enter a valid search term.")

            if 'found_podcasts' in st.session_state:
                found_podcasts = st.session_state['found_podcasts']
                podcast_names = [f"{podcast['collectionName']} by {podcast['artistName']}" for podcast in found_podcasts]
                selected_podcast = st.selectbox("Select a podcast:", podcast_names)
                selected_index = podcast_names.index(selected_podcast)

                if st.button("Get Latest Episode"):
                    with st.spinner('Downloading episode...'):
                        episode_details = download_episode_from_name(found_podcasts[selected_index]['collectionId'], found_podcasts[selected_index]['collectionName'])
                        if episode_details['status'] == 'Success':
                            st.success(episode_details['status_message'])  
                            st.session_state['episode_details'] = episode_details
                            st.session_state['podcast_downloaded'] = True
                            st.session_state['transcription_complete'] = False
                            st.session_stated['index_created'] = False
                        else:
                            st.warning(episode_details['status_message'])
                            st.session_state['podcast_downloaded'] = False

        if st.session_state.get('podcast_downloaded', False):
            if option != "1. Try a sample":
                episode_details = transcribe_episode(sentence_encoder, st.session_state['episode_details'], replicate_client)
            else:
                st.session_state['transcription_complete'] = True

        if st.session_state.get('transcription_complete', False):
            if st.session_state['transcription_complete']:
                chatbox_container = st.container()

                st.session_state['rag_model'] = rag
                st.session_state['topic_model'] = list_topics

                with chatbox_container:
                    if 'chat_with_your_podcast' in st.session_state:
                        st.subheader("Chat with your podcast")
                        episode_details = st.session_state['episode_details']
                        index_name = st.session_state['index_name']

                        if "messages" not in st.session_state:
                            st.session_state.messages = []

                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

                        if prompt := st.chat_input(""):
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            with st.chat_message("user"):
                                st.markdown(prompt)

                            with st.chat_message("assistant"):
                                chat_with_podcast = st.session_state['rag_model']
                                response = st.write_stream(chat_with_podcast(prompt, sentence_encoder, oa_client, index_name, es_client))

                            st.session_state.messages.append({"role": "assistant", "content": response})

                st.header("Need some ideas?")
                fig_col1, fig_col2 = st.columns(2)

                with fig_col1:
                    st.subheader("Podcast summary")
                    if st.button("Show summary"):
                        episode_details = st.session_state['episode_details']
                        soup = BeautifulSoup(episode_details['summary'], "html.parser")
                        for br in soup.find_all("br"):
                            br.replace_with("\n")
                        parsed_text = soup.get_text()
                        st.markdown(parsed_text)

                with fig_col2:
                    st.subheader("Show topics discussed in the episode")
                    episode_details = st.session_state['episode_details']
                    labels = ["Person", "Organization", "Location", "Date", "Money", "Percent", "Event", "Product", "Work of Art", "Book", "Song", "Movie", "Language", "Law", "Facility", "Country", "City", "State", "Nationality", "Religion", "Political group"]
                    selected_labels = st.multiselect("Select or add labels:", labels)
                    new_label = st.text_input("Add a custom label:")
                    if new_label and new_label not in selected_labels:
                        selected_labels.append(new_label)
                    st.write("Selected Labels:", selected_labels)

                    if st.button("Show topics"):
                        with st.spinner('Extracting topics...'):
                            list_my_topics = st.session_state['topic_model']
                            entities = list_my_topics(episode_details, selected_labels)
                            max_length = max(len(words) for words in entities.values())
                            data = {key: words + [''] * (max_length - len(words)) for key, words in entities.items()}
                            df = pd.DataFrame(data)
                            st.dataframe(df, hide_index=True, height=400)
