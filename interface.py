import streamlit as st
from openai import OpenAI
import replicate
from utils import create_index, get_episode_title, get_episode_details, get_feed_details, search_for_episode, download_all, search_podcasts, get_podcast_details, fetch_latest_episode
from helpers import transcribe, list_topics
from rag import rag
from bs4 import BeautifulSoup
import pandas as pd
from elasticsearch import Elasticsearch

def download_episode(episode_details, podcast_name):
    episode_details['filenames'] = []
    episode_details['filenames'] += download_all(episode_details['audio_urls'], podcast_name)
    st.write(f"Downloaded episode {podcast_name}-{episode_details['title']}.")
    return episode_details

def transcribe_episode(episode_details, replicate_client):
    if not st.session_state.get('transcription_complete', False):
        with st.spinner('Transcribing...'):
            output = transcribe(replicate_client, episode_details['filenames'][0])
            episode_details = {**episode_details, **output}
            index_name = create_index(es_client, episode_details)
            st.session_state['episode_details'] = episode_details
            st.session_state['index_name'] = index_name
            st.session_state['es_client'] = es_client
            st.session_state['transcription_complete'] = True
            st.session_state['chat_with_your_podcast'] = True
            st.write(f"Transcription complete.")
    return episode_details

if __name__ == "__main__":
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
        oa_client = OpenAI(api_key=openai_api_key)
        replicate_client = replicate.Client(api_token=replicate_api_key)
        es_client = Elasticsearch(cloud_id=elasticsearch_cloud_id, api_key=elasticsearch_api_key)
        
        option = st.radio(
            "Choose an option:",
            ("1. Provide the iTunes URL for a specific podcast episode",
            "2. Provide a name of a podcast to explore its most recent episode")
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

        if option.startswith("1"):
            episode_url = st.text_input("Enter the iTunes URL of the episode you want:")
            if st.button("Search Episode"):
                if episode_url:
                    try:
                        with st.spinner('Downloading episode...'):
                            podcast_id, episode_title = get_episode_title(episode_url)
                            podcast_details = get_episode_details(podcast_id)
                            if isinstance(podcast_details, str):
                                st.warning(podcast_details)
                            else:
                                podcast_name = podcast_details['collectionName']
                                feed_details = get_feed_details(podcast_details['feedUrl'])
                                episode_details = search_for_episode(episode_title, feed_details)
                                episode_details = download_episode(episode_details, podcast_name)
                        st.session_state['episode_details'] = episode_details
                        st.session_state['podcast_downloaded'] = True
                        st.session_state['transcription_complete'] = False
                    except:
                        st.warning("Please enter a valid URL.")

        elif option.startswith("2"):
            term = st.text_input("Enter a search term for podcasts:")
            if st.button("Search Podcasts"):
                if term:
                    podcasts = search_podcasts(term)
                    if isinstance(podcasts, str):
                        st.warning(podcasts)
                    else:
                        st.session_state['podcasts'] = podcasts
                else:
                    st.warning("Please enter a search term.")

            if 'podcasts' in st.session_state:
                podcasts = st.session_state['podcasts']
                podcast_names = [f"{podcast['collectionName']} by {podcast['artistName']}" for podcast in podcasts]
                selected_podcast = st.selectbox("Select a podcast:", podcast_names)
                selected_index = podcast_names.index(selected_podcast)

                if st.button("Get Latest Episode"):
                    with st.spinner('Downloading episode...'):
                        podcast_id = podcasts[selected_index]['collectionId']
                        podcast_name = podcasts[selected_index]['collectionName']
                        podcast_details = get_podcast_details(podcast_id)
                        if isinstance(podcast_details, str):
                            st.warning(podcast_details)
                        else:
                            episode_details = fetch_latest_episode(podcast_details['feedUrl'])
                            if isinstance(episode_details, str):
                                st.warning(episode_details)
                            else:
                                episode_details = download_episode(episode_details, podcast_name)
                                st.session_state['episode_details'] = episode_details
                                st.session_state['podcast_downloaded'] = True
                                st.session_state['transcription_complete'] = False

        if st.session_state.get('podcast_downloaded', False):
            episode_details = transcribe_episode(st.session_state['episode_details'], replicate_client)

        if st.session_state.get('transcription_complete', False):
            # if st.button("Transcribe Episode"):
            transcribe_episode(st.session_state['episode_details'], replicate_client)

            # if st.session_state.get('transcription_complete', False):
            if st.session_state['transcription_complete']:
                chatbox_container = st.container()

                st.session_state['rag_model'] = rag
                st.session_state['topic_model'] = list_topics

                with chatbox_container:
                    if 'chat_with_your_podcast' in st.session_state:
                        st.subheader("Chat with your podcast")
                        episode_details = st.session_state['episode_details']
                        index_name = st.session_state['index_name']
                        # es_client = st.session_state['es_client']

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
                                response = st.write_stream(chat_with_podcast(prompt, oa_client, index_name, es_client))
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
