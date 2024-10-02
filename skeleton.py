# TODO:
# 2. maybe create a separate file for transcription
# 3. tests (old settings order):
#   3.1. sample / no encoder / no transcription / minsearch / openai - passed
#   3.2. sample / no encoder / no transcription / minsearch / flant5 - passed
#   3.3. sample / t5 / no transcription / elasticsearch / openai
#   3.4. sample / t5 / no transcription / elasticsearch / flant5
#   3.3. sample / openai / no transcription / elasticsearch / openai
#   3.4. sample / openai / no transcription / elasticsearch / flant5
#   3.7. url / t5 / replicate / minsearch / openai
#   3.8. url / t5 / replicate / minsearch / flant5
#   3.9. url / t5 / replicate / elasticsearch / openai
#   3.10. url / t5 / replicate / elasticsearch / flant5
#   3.19. url / openai / replicate / minsearch / openai
#   3.20. url / openai / replicate / minsearch / flant5
#   3.21. url / openai / replicate / elasticsearch / openai
#   3.22. url / openai / replicate / elasticsearch / flant5
#   3.31.  name / t5 / replicate / minsearch / openai
#   3.32.  name / t5 / replicate / minsearch / flant5
#   3.33.  name / t5 / replicate / elasticsearch / openai
#   3.34. name / t5 / replicate / elasticsearch / flant5
#   3.43. name / openai / replicate / minsearch / openai
#   3.44. name / openai / replicate / minsearch / flant5
#   3.45. name / openai / replicate / elasticsearch / openai
#   3.46. name / openai / replicate / elasticsearch / flant5

#   3.13. url / t5 / local / minsearch / openai
#   3.14. url / t5 / local / minsearch / flant5
#   3.15. url / t5 / local / elasticsearch / openai
#   3.16. url / t5 / local / elasticsearch / flant5
#   3.25. url / openai / local / minsearch / openai
#   3.26. url / openai / local / minsearch / flant5
#   3.27. url / openai / local / elasticsearch / openai
#   3.28. url / openai / local / elasticsearch / flant5
#   3.37. name / t5 / local / minsearch / openai
#   3.38. name / t5 / local / minsearch / flant5
#   3.39. name / t5 / local / elasticsearch / openai
#   3.40. name / t5 / local / elasticsearch / flant5
#   3.49. name / openai / local / minsearch / openai
#   3.50. name / openai / local / minsearch / flant5
#   3.51. name / openai / local / elasticsearch / openai
#   3.52. name / openai / local / elasticsearch / flant5

#   3.5. sample / no encoder / no transcription / chromadb / openai
#   3.6. sample / no encoder / no transcription / chromadb / flant5
#   3.11. url / t5 / replicate / chromadb / openai
#   3.12. url / t5 / replicate / chromadb / flant5
#   3.17. url / t5 / local / chromadb / openai
#   3.18. url / t5 / local / chromadb / flant5
#   3.23. url / openai / replicate / chromadb / openai
#   3.24. url / openai / replicate / chromadb / flant5
#   3.29. url / openai / local / chromadb / openai
#   3.31. url / openai / local / chromadb / flant5
#   3.35. name / t5 / replicate / chromadb / openai
#   3.36. name / t5 / replicate / chromadb / flant5
#   3.41. name / t5 / local / chromadb / openai
#   3.42. name / t5 / local / chromadb / flant5
#   3.47. name / openai / replicate / chromadb / openai
#   3.48. name / openai / replicate / chromadb / flant5
#   3.53. name / openai / local / chromadb / openai
#   3.54. name / openai / local / chromadb / flant5


import streamlit as st
from streamlit_functions import *
from ingest import download_podcast, transcribe_podcast, encode_podcast, index_podcast
from rag import rag
import time


if __name__ == "__main__":
    # ask user to provide the necessary api keys
    with st.sidebar:

        # podcast option
        choose_podcast_option()

        # sentence encoder
        if st.session_state.get('episode_option_selected', False):
            choose_encoder()

        # transcription method
        if st.session_state.get('sentence_encoder_selected', False):
            choose_transcription_method()

        # vector database
        if st.session_state.get('transcription_method_selected', False):
            choose_vector_db()

        # llm
        if st.session_state.get('vector_db_selected', False):
            choose_llm()

        # Check if all settings are selected to enable/disable the button
        if st.session_state['episode_option_selected'] and st.session_state['sentence_encoder_selected'] and st.session_state['transcription_method_selected'] and st.session_state['vector_db_selected'] and st.session_state['llm_option_selected']:
            update_session(settings_ready=True)
        else:
            update_session(settings_ready=False)
        
        # Create two columns
        col1, col2 = st.columns(2)

        # Add a button to each column
        col1.button("Apply Settings", on_click=apply_settings, disabled=not st.session_state.get('settings_ready', False))

        col2.button("Reset Settings", on_click=reset_settings, disabled=not st.session_state.get('settings_applied', False))

    st.title("Podcast Search and Download Tool")

    if st.session_state.get('settings_applied', False):
        st.write("Hello!")
        with st.expander("Interact with the podcast"):       

            # download
            if not st.session_state.get('interaction_started', False):
                with st.spinner('Downloading...'):
                    episode_details = download_podcast(**st.session_state)
                    st.write(episode_details.keys())
                    if episode_details['status'] == 'Success':
                        st.success(episode_details['status_message'])
                        update_session(episode_details=episode_details, podcast_downloaded=True)
                    else:
                        st.warning(episode_details['status_message'])
                        update_session(podcast_downloaded=False)

            # transcribe
            if st.session_state['podcast_downloaded'] and not st.session_state.get('interaction_started', False):
                with st.spinner('Transcribing...'):
                    st.session_state['episode_details'].update(transcribe_podcast(**st.session_state))
                    st.write(st.session_state['episode_details']['text'][:100])
                    update_session(podcast_transcribed=True)

            # encode
            if st.session_state['podcast_transcribed'] and not st.session_state.get('interaction_started', False):
                with st.spinner('Encoding...'):
                    if st.session_state['vector_db'] != "1. Minsearch":
                        try:
                            st.session_state['episode_details'].update(encode_podcast(**st.session_state))
                            update_session(podcast_embedded=True)
                        except:
                            update_session(podcast_embedded=False)
                    else:
                        update_session(podcast_embedded=True)

            # populate index
            if st.session_state['podcast_embedded'] and not st.session_state.get('interaction_started', False):
                with st.spinner('Indexing...'):
                    index_podcast(**st.session_state)
                    update_session(podcast_indexed=True)

            # interact
            if st.session_state['podcast_indexed']:
                update_session(interaction_started=True)
                chatbox_container = st.container()
                with chatbox_container:
                    st.subheader("Chat with your podcast")
                    episode_details = st.session_state['episode_details']
                    index_name = st.session_state['index_name']

                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    if query := st.chat_input(""):
                        st.session_state.messages.append({"role": "user", "content": query})
                        with st.chat_message("user"):
                            st.markdown(query)

                        with st.chat_message("assistant"):
                            response = st.write_stream(rag(query, **st.session_state))

                        st.session_state.messages.append({"role": "assistant", "content": response})
